#!/usr/bin/env python3
import json
import time
from pathlib import Path

import numpy as np
import yaml
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, CameraInfo
import tf2_ros
from PIL import Image as PILImage
from PIL import ImageDraw
from ultralytics import YOLO


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def stamp_to_sec(stamp):
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def quat_to_rot(qx, qy, qz, qw):
    xx = qx * qx
    yy = qy * qy
    zz = qz * qz
    xy = qx * qy
    xz = qx * qz
    yz = qy * qz
    wx = qw * qx
    wy = qw * qy
    wz = qw * qz

    return np.array([
        [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz),       2.0 * (xz + wy)],
        [2.0 * (xy + wz),       1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
        [2.0 * (xz - wy),       2.0 * (yz + wx),       1.0 - 2.0 * (xx + yy)],
    ], dtype=np.float64)


def transform_point(tf_msg, p_source):
    t = tf_msg.transform.translation
    q = tf_msg.transform.rotation
    R = quat_to_rot(q.x, q.y, q.z, q.w)
    p = np.asarray(p_source, dtype=np.float64).reshape(3, 1)
    trans = np.array([[t.x], [t.y], [t.z]], dtype=np.float64)
    out = R @ p + trans
    return out.flatten()


def ros_rgb_to_numpy(msg):
    if msg.encoding not in ["rgb8", "bgr8", "rgba8", "bgra8"]:
        raise ValueError(f"Unsupported RGB encoding: {msg.encoding}")

    channels = 4 if msg.encoding in ["rgba8", "bgra8"] else 3
    raw = np.frombuffer(msg.data, dtype=np.uint8)

    row_stride = msg.step
    useful = msg.width * channels
    arr = raw.reshape(msg.height, row_stride)[:, :useful]
    arr = arr.reshape(msg.height, msg.width, channels)

    if msg.encoding == "rgb8":
        rgb = arr[:, :, :3]
    elif msg.encoding == "bgr8":
        rgb = arr[:, :, :3][:, :, ::-1]
    elif msg.encoding == "rgba8":
        rgb = arr[:, :, :3]
    elif msg.encoding == "bgra8":
        rgb = arr[:, :, :3][:, :, ::-1]
    else:
        raise ValueError(f"Unsupported RGB encoding: {msg.encoding}")

    return rgb.copy()


def ros_depth_to_meters(msg):
    if msg.encoding == "32FC1":
        raw = np.frombuffer(msg.data, dtype=np.float32)
        row_stride = msg.step // 4
        depth = raw.reshape(msg.height, row_stride)[:, :msg.width]
        return depth.astype(np.float32)

    if msg.encoding == "16UC1":
        raw = np.frombuffer(msg.data, dtype=np.uint16)
        row_stride = msg.step // 2
        depth = raw.reshape(msg.height, row_stride)[:, :msg.width]
        return depth.astype(np.float32) / 1000.0

    raise ValueError(f"Unsupported depth encoding: {msg.encoding}")


def choose_best_box(result, model_names, target_class_name):
    # 优先处理 OBB
    if hasattr(result, "obb") and result.obb is not None and len(result.obb) > 0:
        obb = result.obb
        conf = obb.conf.cpu().numpy()
        cls = obb.cls.cpu().numpy().astype(int)

        if hasattr(obb, "xyxy"):
            xyxy = obb.xyxy.cpu().numpy()
        else:
            xyxy = result.boxes.xyxy.cpu().numpy()

        valid = list(range(len(conf)))

        if target_class_name:
            name_to_id = {v: k for k, v in model_names.items()}
            if target_class_name not in name_to_id:
                raise ValueError(f"target_class_name={target_class_name} not in model names={model_names}")
            target_id = name_to_id[target_class_name]
            valid = [i for i in valid if cls[i] == target_id]

        if not valid:
            return None

        best = max(valid, key=lambda i: conf[i])
        x1, y1, x2, y2 = xyxy[best].tolist()

        angle = None
        if hasattr(obb, "xywhr"):
            try:
                angle = float(obb.xywhr[best, 4].cpu().numpy())
            except Exception:
                angle = None

        return {
            "type": "obb",
            "x1": int(round(x1)),
            "y1": int(round(y1)),
            "x2": int(round(x2)),
            "y2": int(round(y2)),
            "conf": float(conf[best]),
            "class_id": int(cls[best]),
            "class_name": str(model_names.get(int(cls[best]), str(cls[best]))),
            "angle_rad": angle,
        }

    # 普通 bbox
    if result.boxes is None or len(result.boxes) == 0:
        return None

    boxes = result.boxes
    xyxy = boxes.xyxy.cpu().numpy()
    conf = boxes.conf.cpu().numpy()
    cls = boxes.cls.cpu().numpy().astype(int)

    valid = list(range(len(conf)))

    if target_class_name:
        name_to_id = {v: k for k, v in model_names.items()}
        if target_class_name not in name_to_id:
            raise ValueError(f"target_class_name={target_class_name} not in model names={model_names}")
        target_id = name_to_id[target_class_name]
        valid = [i for i in valid if cls[i] == target_id]

    if not valid:
        return None

    best = max(valid, key=lambda i: conf[i])
    x1, y1, x2, y2 = xyxy[best].tolist()

    return {
        "type": "bbox",
        "x1": int(round(x1)),
        "y1": int(round(y1)),
        "x2": int(round(x2)),
        "y2": int(round(y2)),
        "conf": float(conf[best]),
        "class_id": int(cls[best]),
        "class_name": str(model_names.get(int(cls[best]), str(cls[best]))),
        "angle_rad": None,
    }


def depth_from_bbox(depth, box, depth_cfg):
    h, w = depth.shape[:2]

    x1 = max(0, min(w - 1, box["x1"]))
    y1 = max(0, min(h - 1, box["y1"]))
    x2 = max(0, min(w - 1, box["x2"]))
    y2 = max(0, min(h - 1, box["y2"]))

    if x2 <= x1 or y2 <= y1:
        return None, None

    ratio = float(depth_cfg["center_crop_ratio"])
    ratio = max(0.1, min(1.0, ratio))

    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    bw = (x2 - x1) * ratio
    bh = (y2 - y1) * ratio

    rx1 = int(max(0, round(cx - bw / 2.0)))
    rx2 = int(min(w, round(cx + bw / 2.0)))
    ry1 = int(max(0, round(cy - bh / 2.0)))
    ry2 = int(min(h, round(cy + bh / 2.0)))

    crop = depth[ry1:ry2, rx1:rx2]
    valid = crop[np.isfinite(crop)]

    dmin = float(depth_cfg["min_m"])
    dmax = float(depth_cfg["max_m"])
    valid = valid[(valid >= dmin) & (valid <= dmax)]

    if valid.size == 0:
        return None, [rx1, ry1, rx2, ry2]

    z = float(np.median(valid))
    return z, [rx1, ry1, rx2, ry2]


def backproject(u, v, z, info):
    fx = float(info.k[0])
    fy = float(info.k[4])
    cx = float(info.k[2])
    cy = float(info.k[5])

    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    return [float(x), float(y), float(z)]


class MethodANode(Node):
    def __init__(self, cfg):
        super().__init__("method_a_yolo_rgbd_once")

        self.cfg = cfg
        self.rgb_msg = None
        self.depth_msg = None
        self.info_msg = None

        self.create_subscription(Image, cfg["topics"]["rgb"], self.rgb_cb, qos_profile_sensor_data)
        self.create_subscription(Image, cfg["topics"]["depth"], self.depth_cb, qos_profile_sensor_data)
        self.create_subscription(CameraInfo, cfg["topics"]["camera_info"], self.info_cb, qos_profile_sensor_data)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

    def rgb_cb(self, msg):
        self.rgb_msg = msg

    def depth_cb(self, msg):
        self.depth_msg = msg

    def info_cb(self, msg):
        self.info_msg = msg

    def wait_bundle(self):
        timeout = float(self.cfg["runtime"]["timeout_sec"])
        slop = float(self.cfg["runtime"]["sync_slop_sec"])
        start = time.time()

        while time.time() - start < timeout:
            rclpy.spin_once(self, timeout_sec=0.1)

            if self.rgb_msg is None or self.depth_msg is None or self.info_msg is None:
                continue

            tr = stamp_to_sec(self.rgb_msg.header.stamp)
            td = stamp_to_sec(self.depth_msg.header.stamp)
            ti = stamp_to_sec(self.info_msg.header.stamp)

            if max(abs(tr - td), abs(tr - ti), abs(td - ti)) <= slop:
                return self.rgb_msg, self.depth_msg, self.info_msg

        raise TimeoutError("Failed to receive synchronized RGB / Depth / CameraInfo")

    def lookup_base_transform(self, info_frame_id):
        base_candidates = self.cfg["frames"]["base_frame_candidates"]
        camera_candidates = []

        if info_frame_id:
            camera_candidates.append(info_frame_id)

        for c in self.cfg["frames"]["camera_frame_candidates"]:
            if c not in camera_candidates:
                camera_candidates.append(c)

        end = time.time() + 3.0
        while time.time() < end:
            rclpy.spin_once(self, timeout_sec=0.1)

            for base in base_candidates:
                for cam in camera_candidates:
                    try:
                        tf_msg = self.tf_buffer.lookup_transform(base, cam, rclpy.time.Time())
                        return base, cam, tf_msg
                    except Exception:
                        pass

        return None, None, None


def draw_debug(rgb, box, roi, p_cam, p_base, out_path):
    img = PILImage.fromarray(rgb)
    draw = ImageDraw.Draw(img)

    x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
    draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=3)

    u = int(round(0.5 * (x1 + x2)))
    v = int(round(0.5 * (y1 + y2)))
    draw.ellipse([u - 4, v - 4, u + 4, v + 4], fill=(0, 0, 255))

    if roi is not None:
        rx1, ry1, rx2, ry2 = roi
        draw.rectangle([rx1, ry1, rx2, ry2], outline=(255, 255, 0), width=2)

    lines = [
        f"YOLO: {box['class_name']} conf={box['conf']:.2f}",
        f"cam: x={p_cam[0]:.3f}, y={p_cam[1]:.3f}, z={p_cam[2]:.3f}",
    ]

    if p_base is not None:
        lines.append(f"base: x={p_base[0]:.3f}, y={p_base[1]:.3f}, z={p_base[2]:.3f}")
    else:
        lines.append("base: TF not found")

    y = 10
    for line in lines:
        draw.text((10, y), line, fill=(255, 255, 0))
        y += 22

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


def main():
    cfg_path = "/home/ubuntu/ros2_ws/day5_method_a/config/method_a_config.yaml"
    cfg = load_yaml(cfg_path)

    print("[INFO] Loading YOLO model:", cfg["yolo"]["model_path"])
    model = YOLO(cfg["yolo"]["model_path"])
    print("[INFO] YOLO model names:", model.names)

    rclpy.init()
    node = MethodANode(cfg)

    try:
        print("[INFO] Waiting for RGB-D-CameraInfo bundle...")
        rgb_msg, depth_msg, info_msg = node.wait_bundle()

        rgb = ros_rgb_to_numpy(rgb_msg)
        depth = ros_depth_to_meters(depth_msg)

        print("[INFO] Running YOLO inference...")
        pil = PILImage.fromarray(rgb)
        results = model.predict(
            source=pil,
            conf=float(cfg["yolo"]["conf_thres"]),
            imgsz=int(cfg["yolo"]["imgsz"]),
            verbose=False,
        )

        if len(results) == 0:
            raise RuntimeError("YOLO returned no result")

        box = choose_best_box(
            results[0],
            model.names,
            cfg["yolo"].get("target_class_name", ""),
        )

        output = {
            "rgb_topic": cfg["topics"]["rgb"],
            "depth_topic": cfg["topics"]["depth"],
            "camera_info_topic": cfg["topics"]["camera_info"],
            "rgb_frame_id": rgb_msg.header.frame_id,
            "depth_frame_id": depth_msg.header.frame_id,
            "camera_info_frame_id": info_msg.header.frame_id,
            "detected": False,
        }

        if box is None:
            output["reason"] = "no_yolo_detection"
            out_json = cfg["runtime"]["output_json"]
            Path(out_json).parent.mkdir(parents=True, exist_ok=True)
            Path(out_json).write_text(json.dumps(output, indent=2))
            print(json.dumps(output, indent=2))
            print("[WARN] YOLO did not detect the target.")
            return

        z, roi = depth_from_bbox(depth, box, cfg["depth"])

        output["detected"] = True
        output["yolo"] = {
            "box_type": box["type"],
            "class_id": box["class_id"],
            "class_name": box["class_name"],
            "confidence": box["conf"],
            "bbox_xyxy": [box["x1"], box["y1"], box["x2"], box["y2"]],
            "bbox_center_uv": [
                0.5 * (box["x1"] + box["x2"]),
                0.5 * (box["y1"] + box["y2"]),
            ],
            "angle_rad": box["angle_rad"],
        }

        if z is None:
            output["depth_valid"] = False
            output["reason"] = "no_valid_depth_in_bbox"
            out_json = cfg["runtime"]["output_json"]
            Path(out_json).parent.mkdir(parents=True, exist_ok=True)
            Path(out_json).write_text(json.dumps(output, indent=2))
            print(json.dumps(output, indent=2))
            print("[WARN] No valid depth in bbox ROI.")
            return

        u = 0.5 * (box["x1"] + box["x2"])
        v = 0.5 * (box["y1"] + box["y2"])
        p_cam = backproject(u, v, z, info_msg)

        base_frame, camera_frame_used, tf_msg = node.lookup_base_transform(info_msg.header.frame_id)

        p_base = None
        if tf_msg is not None:
            p_base = transform_point(tf_msg, p_cam).tolist()

        output["depth_valid"] = True
        output["depth"] = {
            "median_m": z,
            "roi_xyxy": roi,
        }

        output["estimate_camera_frame"] = {
            "frame": info_msg.header.frame_id,
            "x": p_cam[0],
            "y": p_cam[1],
            "z": p_cam[2],
        }

        output["tf"] = {
            "transform_found": tf_msg is not None,
            "base_frame": base_frame,
            "camera_frame_used": camera_frame_used,
        }

        output["estimate_base_frame"] = None
        if p_base is not None:
            output["estimate_base_frame"] = {
                "frame": base_frame,
                "x": p_base[0],
                "y": p_base[1],
                "z": p_base[2],
            }

        out_json = cfg["runtime"]["output_json"]
        out_img = cfg["runtime"]["output_debug_image"]

        Path(out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(out_json).write_text(json.dumps(output, indent=2))

        draw_debug(rgb, box, roi, p_cam, p_base, out_img)

        print(json.dumps(output, indent=2))
        print("[INFO] Saved JSON:", out_json)
        print("[INFO] Saved debug image:", out_img)

    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
