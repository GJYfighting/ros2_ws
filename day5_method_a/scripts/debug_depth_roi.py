#!/usr/bin/env python3
import json
import time
from pathlib import Path

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image


RESULT_JSON = "/home/ubuntu/ros2_ws/day5_method_a/results/method_a_color_once.json"
DEPTH_TOPIC = "/depth_cam/depth/image_raw"
OUT_IMAGE = "/home/ubuntu/ros2_ws/day5_method_a/results/debug/depth_roi_debug.png"


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


def print_stats(name, arr):
    flat = arr.reshape(-1)

    finite = flat[np.isfinite(flat)]
    nonzero = finite[finite > 0.0]

    print(f"\n===== {name} =====")
    print("shape:", arr.shape)
    print("total pixels:", flat.size)
    print("finite count:", finite.size)
    print("nonzero finite count:", nonzero.size)

    if finite.size > 0:
        print("finite min:", float(np.min(finite)))
        print("finite max:", float(np.max(finite)))
        print("finite median:", float(np.median(finite)))

    if nonzero.size > 0:
        print("nonzero min:", float(np.min(nonzero)))
        print("nonzero max:", float(np.max(nonzero)))
        print("nonzero median:", float(np.median(nonzero)))

    valid_002_5 = finite[(finite >= 0.02) & (finite <= 5.0)]
    valid_002_10 = finite[(finite >= 0.02) & (finite <= 10.0)]

    print("valid [0.02, 5.0] count:", valid_002_5.size)
    if valid_002_5.size > 0:
        print("valid [0.02, 5.0] median:", float(np.median(valid_002_5)))

    print("valid [0.02, 10.0] count:", valid_002_10.size)
    if valid_002_10.size > 0:
        print("valid [0.02, 10.0] median:", float(np.median(valid_002_10)))


class DepthOnce(Node):
    def __init__(self):
        super().__init__("debug_depth_roi")
        self.msg = None
        self.create_subscription(
            Image,
            DEPTH_TOPIC,
            self.cb,
            qos_profile_sensor_data
        )

    def cb(self, msg):
        self.msg = msg


def main():
    result = json.loads(Path(RESULT_JSON).read_text())

    if "color_detection" not in result:
        raise RuntimeError("No color_detection in result JSON. Run method_a_color_rgbd_once.py first.")

    bbox = result["color_detection"]["bbox_xywh"]
    x, y, w, h = bbox

    rclpy.init()
    node = DepthOnce()

    print("[INFO] Waiting for one depth image...")
    start = time.time()
    while time.time() - start < 8.0:
        rclpy.spin_once(node, timeout_sec=0.1)
        if node.msg is not None:
            break

    if node.msg is None:
        raise RuntimeError("No depth message received")

    msg = node.msg
    print("[INFO] depth encoding:", msg.encoding)
    print("[INFO] depth size:", msg.width, msg.height)
    print("[INFO] depth frame:", msg.header.frame_id)

    depth = ros_depth_to_meters(msg)

    H, W = depth.shape[:2]

    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(W, x + w)
    y2 = min(H, y + h)

    bbox_depth = depth[y1:y2, x1:x2]

    cx = x + w / 2.0
    cy = y + h / 2.0

    ratio = 0.60
    cw = w * ratio
    ch = h * ratio

    rx1 = int(max(0, round(cx - cw / 2.0)))
    rx2 = int(min(W, round(cx + cw / 2.0)))
    ry1 = int(max(0, round(cy - ch / 2.0)))
    ry2 = int(min(H, round(cy + ch / 2.0)))

    roi_depth = depth[ry1:ry2, rx1:rx2]

    print_stats("whole depth image", depth)
    print_stats("bbox depth region", bbox_depth)
    print_stats("center crop depth region", roi_depth)

    # 保存 depth 可视化图
    vis = depth.copy()
    finite = vis[np.isfinite(vis)]

    if finite.size > 0:
        lo = np.percentile(finite, 2)
        hi = np.percentile(finite, 98)
        if hi <= lo:
            hi = lo + 1e-6

        vis = np.nan_to_num(vis, nan=0.0, posinf=hi, neginf=lo)
        vis = np.clip(vis, lo, hi)
        vis = ((vis - lo) / (hi - lo) * 255.0).astype(np.uint8)
    else:
        vis = np.zeros_like(vis, dtype=np.uint8)

    color = cv2.applyColorMap(vis, cv2.COLORMAP_JET)

    cv2.rectangle(color, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.rectangle(color, (rx1, ry1), (rx2, ry2), (255, 255, 0), 2)
    cv2.circle(color, (int(round(cx)), int(round(cy))), 4, (255, 255, 255), -1)

    Path(OUT_IMAGE).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(OUT_IMAGE, color)

    print("\n[INFO] Saved depth ROI debug image:", OUT_IMAGE)

    node.destroy_node()
    try:
        rclpy.shutdown()
    except Exception:
        pass


if __name__ == "__main__":
    main()
