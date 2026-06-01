#!/usr/bin/env python3
import json
import time
from pathlib import Path

import cv2
import numpy as np
import yaml
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, CameraInfo
import tf2_ros


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


def detect_red_block(rgb, cfg):
    detector_cfg = cfg["color_detector"]

    # OpenCV 用 HSV 时一般输入 BGR，这里先从 RGB 转 BGR
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    low1 = np.array(detector_cfg["hsv_low_1"], dtype=np.uint8)
    high1 = np.array(detector_cfg["hsv_high_1"], dtype=np.uint8)
    low2 = np.array(detector_cfg["hsv_low_2"], dtype=np.uint8)
    high2 = np.array(detector_cfg["hsv_high_2"], dtype=np.uint8)

    mask1 = cv2.inRange(hsv, low1, high1)
    mask2 = cv2.inRange(hsv, low2, high2)
    mask = cv2.bitwise_or(mask1, mask2)

    k = int(detector_cfg.get("morph_kernel", 5))
    if k > 1:
        kernel = np.ones((k, k), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    min_area = float(detector_cfg.get("min_area_px", 80))

    best_cnt = None
    best_area = -1.0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue

        if area > best_area:
            best_area = area
            best_cnt = cnt

    if best_cnt is None:
        return None, mask

    rect = cv2.minAreaRect(best_cnt)
    (cx, cy), (rw, rh), angle = rect

    x, y, w, h = cv2.boundingRect(best_cnt)

    contour_mask = np.zeros(mask.shape, dtype=np.uint8)
    cv2.drawContours(contour_mask, [best_cnt], -1, 255, thickness=-1)

    box_pts = cv2.boxPoints(rect)
    box_pts = np.round(box_pts).astype(int)

    detection = {
        "center_u": float(cx),
        "center_v": float(cy),
        "bbox_xywh": [int(x), int(y), int(w), int(h)],
        "area_px": float(best_area),
        "angle_deg": float(angle),
        "rotated_box_points": box_pts.tolist(),
        "contour_mask": contour_mask,
    }

    return detection, mask


def depth_from_detection(depth, detection, cfg):
    depth_cfg = cfg["depth"]

    dmin = float(depth_cfg["min_m"])
    dmax = float(depth_cfg["max_m"])

    h_img, w_img = depth.shape[:2]

    # 1. 优先使用轮廓 mask 内的深度
    if bool(depth_cfg.get("use_contour_mask", True)):
        mask = detection["contour_mask"]
        valid = depth[mask > 0]
        valid = valid[np.isfinite(valid)]
        valid = valid[(valid >= dmin) & (valid <= dmax)]

        if valid.size > 0:
            return float(np.median(valid)), "contour_mask", None

    # 2. 如果 mask 失败，则使用 bbox 中心区域
    x, y, w, h = detection["bbox_xywh"]

    ratio = float(depth_cfg.get("fallback_center_crop_ratio", 0.6))
    ratio = max(0.1, min(1.0, ratio))

    cx = x + w / 2.0
    cy = y + h / 2.0

    cw = w * ratio
    ch = h * ratio

    x1 = int(max(0, round(cx - cw / 2.0)))
    x2 = int(min(w_img, round(cx + cw / 2.0)))
    y1 = int(max(0, round(cy - ch / 2.0)))
    y2 = int(min(h_img, round(cy + ch / 2.0)))

    crop = depth[y1:y2, x1:x2]
    valid = crop[np.isfinite(crop)]
    valid = valid[(valid >= dmin) & (valid <= dmax)]

    if valid.size == 0:
        return None, "none", [x1, y1, x2, y2]

    return float(np.median(valid)), "bbox_center_crop", [x1, y1, x2, y2]


def backproject(u, v, z, info):
    fx = float(info.k[0])
    fy = float(info.k[4])
    cx = float(info.k[2])
    cy = float(info.k[5])

    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    return [float(x), float(y), float(z)]


class ColorRGBDNode(Node):
    def __init__(self, cfg):
        super().__init__("method_a_color_rgbd_once")

        self.cfg = cfg

        self.rgb_msg = None
        self.depth_msg = None
        self.info_msg = None

        self.create_subscription(
            Image,
            cfg["topics"]["rgb"],
            self.rgb_cb,
            qos_profile_sensor_data
        )

        self.create_subscription(
            Image,
            cfg["topics"]["depth"],
            self.depth_cb,
            qos_profile_sensor_data
        )

        self.create_subscription(
            CameraInfo,
            cfg["topics"]["camera_info"],
            self.info_cb,
            qos_profile_sensor_data
        )

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
                        tf_msg = self.tf_buffer.lookup_transform(
                            base,
                            cam,
                            rclpy.time.Time()
                        )
                        return base, cam, tf_msg
                    except Exception:
                        pass

        return None, None, None


def draw_debug(rgb, detection, z, depth_method, crop_roi, p_cam, p_base, out_path):
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    x, y, w, h = detection["bbox_xywh"]
    cv2.rectangle(bgr, (x, y), (x + w, y + h), (0, 255, 0), 2)

    u = int(round(detection["center_u"]))
    v = int(round(detection["center_v"]))
    cv2.circle(bgr, (u, v), 5, (255, 0, 0), -1)

    pts = np.array(detection["rotated_box_points"], dtype=np.int32)
    cv2.polylines(bgr, [pts], isClosed=True, color=(0, 255, 255), thickness=2)

    if crop_roi is not None:
        x1, y1, x2, y2 = crop_roi
        cv2.rectangle(bgr, (x1, y1), (x2, y2), (255, 255, 0), 2)

    lines = [
        f"color block detected",
        f"z={z:.3f}m method={depth_method}",
        f"cam: x={p_cam[0]:.3f}, y={p_cam[1]:.3f}, z={p_cam[2]:.3f}",
    ]

    if p_base is not None:
        lines.append(f"base: x={p_base[0]:.3f}, y={p_base[1]:.3f}, z={p_base[2]:.3f}")
    else:
        lines.append("base: TF not found")

    y_text = 25
    for line in lines:
        cv2.putText(
            bgr,
            line,
            (10, y_text),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2
        )
        y_text += 24

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), bgr)


def main():
    cfg_path = "/home/ubuntu/ros2_ws/day5_method_a/config/method_a_color_config.yaml"
    cfg = load_yaml(cfg_path)

    rclpy.init()
    node = ColorRGBDNode(cfg)

    try:
        print("[INFO] Waiting for RGB-D-CameraInfo bundle...")
        rgb_msg, depth_msg, info_msg = node.wait_bundle()

        rgb = ros_rgb_to_numpy(rgb_msg)
        depth = ros_depth_to_meters(depth_msg)

        print("[INFO] Running red color block detection...")
        detection, mask = detect_red_block(rgb, cfg)

        output = {
            "method": "color_block_rgbd",
            "rgb_topic": cfg["topics"]["rgb"],
            "depth_topic": cfg["topics"]["depth"],
            "camera_info_topic": cfg["topics"]["camera_info"],
            "rgb_frame_id": rgb_msg.header.frame_id,
            "depth_frame_id": depth_msg.header.frame_id,
            "camera_info_frame_id": info_msg.header.frame_id,
            "detected": False,
        }

        mask_path = cfg["runtime"]["output_mask_image"]
        Path(mask_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(mask_path, mask)

        if detection is None:
            output["reason"] = "no_red_block_detected"

            out_json = cfg["runtime"]["output_json"]
            Path(out_json).parent.mkdir(parents=True, exist_ok=True)
            Path(out_json).write_text(json.dumps(output, indent=2))

            print(json.dumps(output, indent=2))
            print("[WARN] No red block detected.")
            print("[INFO] Saved mask image:", mask_path)
            return

        z, depth_method, crop_roi = depth_from_detection(depth, detection, cfg)

        output["detected"] = True
        output["color_detection"] = {
            "center_uv": [detection["center_u"], detection["center_v"]],
            "bbox_xywh": detection["bbox_xywh"],
            "area_px": detection["area_px"],
            "angle_deg": detection["angle_deg"],
            "rotated_box_points": detection["rotated_box_points"],
        }

        if z is None:
            output["depth_valid"] = False
            output["reason"] = "no_valid_depth_for_color_block"

            out_json = cfg["runtime"]["output_json"]
            Path(out_json).parent.mkdir(parents=True, exist_ok=True)
            Path(out_json).write_text(json.dumps(output, indent=2))

            print(json.dumps(output, indent=2))
            print("[WARN] Red block detected, but no valid depth.")
            print("[INFO] Saved mask image:", mask_path)
            return

        u = detection["center_u"]
        v = detection["center_v"]

        p_cam = backproject(u, v, z, info_msg)

        base_frame, camera_frame_used, tf_msg = node.lookup_base_transform(
            info_msg.header.frame_id
        )

        p_base = None
        if tf_msg is not None:
            p_base = transform_point(tf_msg, p_cam).tolist()

        output["depth_valid"] = True
        output["depth"] = {
            "median_m": z,
            "method": depth_method,
            "fallback_crop_roi_xyxy": crop_roi,
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

        draw_debug(
            rgb=rgb,
            detection=detection,
            z=z,
            depth_method=depth_method,
            crop_roi=crop_roi,
            p_cam=p_cam,
            p_base=p_base,
            out_path=out_img
        )

        print(json.dumps(output, indent=2))
        print("[INFO] Saved JSON:", out_json)
        print("[INFO] Saved debug image:", out_img)
        print("[INFO] Saved mask image:", mask_path)

    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
