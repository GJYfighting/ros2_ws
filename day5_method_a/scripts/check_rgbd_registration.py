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


CONFIG_PATH = "/home/ubuntu/ros2_ws/day5_method_a/config/method_a_color_config.yaml"

OUT_JSON = "/home/ubuntu/ros2_ws/day5_method_a/results/rgbd_registration_check.json"
OUT_IMAGE = "/home/ubuntu/ros2_ws/day5_method_a/results/debug/rgbd_registration_check.png"
OUT_MASK = "/home/ubuntu/ros2_ws/day5_method_a/results/debug/rgbd_registration_mask.png"


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def stamp_to_sec(stamp):
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


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
    color_cfg = cfg["color_detector"]

    # RGB -> BGR -> HSV
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    low1 = np.array(color_cfg["hsv_low_1"], dtype=np.uint8)
    high1 = np.array(color_cfg["hsv_high_1"], dtype=np.uint8)
    low2 = np.array(color_cfg["hsv_low_2"], dtype=np.uint8)
    high2 = np.array(color_cfg["hsv_high_2"], dtype=np.uint8)

    mask1 = cv2.inRange(hsv, low1, high1)
    mask2 = cv2.inRange(hsv, low2, high2)
    mask = cv2.bitwise_or(mask1, mask2)

    k = int(color_cfg.get("morph_kernel", 5))
    if k > 1:
        kernel = np.ones((k, k), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    min_area = float(color_cfg.get("min_area_px", 80))

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
        return None, mask, None

    x, y, w, h = cv2.boundingRect(best_cnt)

    rect = cv2.minAreaRect(best_cnt)
    (cx, cy), (rw, rh), angle = rect
    box_pts = cv2.boxPoints(rect)
    box_pts = np.round(box_pts).astype(int)

    contour_mask = np.zeros(mask.shape, dtype=np.uint8)
    cv2.drawContours(contour_mask, [best_cnt], -1, 255, thickness=-1)

    detection = {
        "center_uv": [float(cx), float(cy)],
        "bbox_xywh": [int(x), int(y), int(w), int(h)],
        "area_px": float(best_area),
        "angle_deg": float(angle),
        "rotated_box_points": box_pts.tolist(),
    }

    return detection, mask, contour_mask


def valid_depth_values(arr, dmin=0.02, dmax=10.0):
    flat = arr.reshape(-1)
    finite = flat[np.isfinite(flat)]
    valid = finite[(finite >= dmin) & (finite <= dmax)]
    return finite, valid


def stats_dict(prefix, arr, dmin=0.02, dmax=10.0):
    finite, valid = valid_depth_values(arr, dmin=dmin, dmax=dmax)

    out = {
        f"{prefix}_total": int(arr.size),
        f"{prefix}_finite": int(finite.size),
        f"{prefix}_valid": int(valid.size),
    }

    if finite.size > 0:
        out[f"{prefix}_finite_min"] = float(np.min(finite))
        out[f"{prefix}_finite_max"] = float(np.max(finite))
        out[f"{prefix}_finite_median"] = float(np.median(finite))

    if valid.size > 0:
        out[f"{prefix}_valid_min"] = float(np.min(valid))
        out[f"{prefix}_valid_max"] = float(np.max(valid))
        out[f"{prefix}_valid_median"] = float(np.median(valid))

    return out


class RGBDRegistrationCheck(Node):
    def __init__(self, cfg):
        super().__init__("rgbd_registration_check")

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

        raise TimeoutError("Timeout waiting for RGB-D-CameraInfo bundle")


def make_depth_visual(depth):
    vis = depth.copy()
    finite = vis[np.isfinite(vis)]

    if finite.size == 0:
        return np.zeros((depth.shape[0], depth.shape[1], 3), dtype=np.uint8)

    lo = np.percentile(finite, 2)
    hi = np.percentile(finite, 98)

    if hi <= lo:
        hi = lo + 1e-6

    vis = np.nan_to_num(vis, nan=0.0, posinf=hi, neginf=lo)
    vis = np.clip(vis, lo, hi)
    vis = ((vis - lo) / (hi - lo) * 255.0).astype(np.uint8)
    color = cv2.applyColorMap(vis, cv2.COLORMAP_JET)
    return color


def main():
    cfg = load_yaml(CONFIG_PATH)

    dmin = float(cfg["depth"]["min_m"])
    dmax = float(cfg["depth"]["max_m"])

    rclpy.init()
    node = RGBDRegistrationCheck(cfg)

    try:
        print("[INFO] Waiting for RGB-D-CameraInfo bundle...")
        rgb_msg, depth_msg, info_msg = node.wait_bundle()

        print("[INFO] Got messages:")
        print("  rgb:", rgb_msg.width, rgb_msg.height, rgb_msg.encoding, rgb_msg.header.frame_id)
        print("  depth:", depth_msg.width, depth_msg.height, depth_msg.encoding, depth_msg.header.frame_id)
        print("  camera_info:", info_msg.width, info_msg.height, info_msg.header.frame_id)

        rgb = ros_rgb_to_numpy(rgb_msg)
        depth = ros_depth_to_meters(depth_msg)

        detection, mask, contour_mask = detect_red_block(rgb, cfg)

        result = {
            "rgb_topic": cfg["topics"]["rgb"],
            "depth_topic": cfg["topics"]["depth"],
            "camera_info_topic": cfg["topics"]["camera_info"],
            "rgb_frame": rgb_msg.header.frame_id,
            "depth_frame": depth_msg.header.frame_id,
            "camera_info_frame": info_msg.header.frame_id,
            "rgb_encoding": rgb_msg.encoding,
            "depth_encoding": depth_msg.encoding,
            "rgb_size": [int(rgb_msg.width), int(rgb_msg.height)],
            "depth_size": [int(depth_msg.width), int(depth_msg.height)],
            "camera_info_size": [int(info_msg.width), int(info_msg.height)],
            "detected": detection is not None,
        }

        result.update(stats_dict("whole_depth", depth, dmin=dmin, dmax=dmax))

        Path(OUT_MASK).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(OUT_MASK, mask)

        debug_rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        depth_color = make_depth_visual(depth)

        if detection is None:
            result["registration_pass"] = False
            result["reason"] = "no_red_block_detected"
        else:
            x, y, w, h = detection["bbox_xywh"]
            cx, cy = detection["center_uv"]

            result["color_detection"] = detection

            bbox_depth = depth[y:y+h, x:x+w]
            result.update(stats_dict("bbox_depth", bbox_depth, dmin=dmin, dmax=dmax))

            if contour_mask is not None:
                contour_depth = depth[contour_mask > 0]
                result.update(stats_dict("contour_depth", contour_depth, dmin=dmin, dmax=dmax))

            bbox_valid = int(result.get("bbox_depth_valid", 0))
            contour_valid = int(result.get("contour_depth_valid", 0))

            result["registration_pass"] = bool(bbox_valid > 0 or contour_valid > 0)

            if result["registration_pass"]:
                result["reason"] = "rgb_bbox_has_valid_depth"
            else:
                result["reason"] = "rgb_bbox_has_no_valid_depth"

            cv2.rectangle(debug_rgb, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.circle(debug_rgb, (int(round(cx)), int(round(cy))), 5, (255, 0, 0), -1)

            cv2.rectangle(depth_color, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.circle(depth_color, (int(round(cx)), int(round(cy))), 5, (255, 255, 255), -1)

            txt = (
                f"registration_pass={result['registration_pass']} "
                f"bbox_valid={bbox_valid} contour_valid={contour_valid}"
            )

            cv2.putText(
                debug_rgb,
                txt,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2
            )

            cv2.putText(
                depth_color,
                txt,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2
            )

        # 拼接 RGB debug 和 depth debug，方便对比
        combined = np.hstack([debug_rgb, depth_color])

        Path(OUT_JSON).parent.mkdir(parents=True, exist_ok=True)
        Path(OUT_IMAGE).parent.mkdir(parents=True, exist_ok=True)

        Path(OUT_JSON).write_text(json.dumps(result, indent=2))
        cv2.imwrite(OUT_IMAGE, combined)

        print(json.dumps(result, indent=2))
        print("[INFO] Saved JSON:", OUT_JSON)
        print("[INFO] Saved debug image:", OUT_IMAGE)
        print("[INFO] Saved mask image:", OUT_MASK)

    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
