#!/usr/bin/env python3
import argparse
import csv
import time
from pathlib import Path

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image


RGB_TOPIC = "/depth_cam/rgbd/image"
OUT_DIR = Path("/home/ubuntu/ros2_ws/day5_method_a/results/observe_search")
CSV_PATH = OUT_DIR / "observe_search_results.csv"


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


def detect_red(rgb):
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    # 稍微宽松一些的红色阈值
    low1 = np.array([0, 25, 15], dtype=np.uint8)
    high1 = np.array([20, 255, 255], dtype=np.uint8)
    low2 = np.array([150, 25, 15], dtype=np.uint8)
    high2 = np.array([180, 255, 255], dtype=np.uint8)

    mask1 = cv2.inRange(hsv, low1, high1)
    mask2 = cv2.inRange(hsv, low2, high2)
    mask = cv2.bitwise_or(mask1, mask2)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best = None
    best_area = 0.0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > best_area:
            best_area = area
            best = cnt

    if best is None or best_area < 20:
        return False, best_area, None, mask

    x, y, w, h = cv2.boundingRect(best)
    M = cv2.moments(best)

    if abs(M["m00"]) > 1e-6:
        u = M["m10"] / M["m00"]
        v = M["m01"] / M["m00"]
    else:
        u = x + w / 2.0
        v = y + h / 2.0

    return True, best_area, {
        "bbox": [x, y, w, h],
        "center": [u, v]
    }, mask


class ImageCapture(Node):
    def __init__(self):
        super().__init__("save_and_score_rgbd_observe")
        self.msg = None
        self.create_subscription(
            Image,
            RGB_TOPIC,
            self.cb,
            qos_profile_sensor_data
        )

    def cb(self, msg):
        self.msg = msg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pose", required=True)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rclpy.init()
    node = ImageCapture()

    start = time.time()

    while time.time() - start < 8.0:
        rclpy.spin_once(node, timeout_sec=0.1)
        if node.msg is not None:
            break

    if node.msg is None:
        raise RuntimeError("No RGBD image received")

    msg = node.msg
    rgb = ros_rgb_to_numpy(msg)

    detected, area, det, mask = detect_red(rgb)

    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    img_std = float(rgb.std())
    img_mean = float(rgb.mean())

    if det is not None:
        x, y, w, h = det["bbox"]
        u, v = det["center"]

        cv2.rectangle(bgr, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.circle(bgr, (int(round(u)), int(round(v))), 5, (255, 0, 0), -1)

        center_score = 1.0 - min(
            1.0,
            abs(u - rgb.shape[1] / 2.0) / (rgb.shape[1] / 2.0)
        )
    else:
        x = y = w = h = -1
        u = v = -1
        center_score = 0.0

    text = f"{args.pose} detected={detected} area={area:.1f} std={img_std:.2f}"
    cv2.putText(
        bgr,
        text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2
    )

    img_path = OUT_DIR / f"{args.pose}.png"
    mask_path = OUT_DIR / f"{args.pose}_mask.png"

    cv2.imwrite(str(img_path), bgr)
    cv2.imwrite(str(mask_path), mask)

    new_file = not CSV_PATH.exists()

    with open(CSV_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow([
                "pose",
                "detected",
                "red_area_px",
                "bbox_x",
                "bbox_y",
                "bbox_w",
                "bbox_h",
                "center_u",
                "center_v",
                "center_score",
                "image_mean",
                "image_std",
                "rgb_frame"
            ])

        writer.writerow([
            args.pose,
            int(detected),
            area,
            x,
            y,
            w,
            h,
            u,
            v,
            center_score,
            img_mean,
            img_std,
            msg.header.frame_id
        ])

    print(f"pose={args.pose}")
    print(f"detected={detected}")
    print(f"red_area_px={area}")
    print(f"image_mean={img_mean:.3f}")
    print(f"image_std={img_std:.3f}")
    print(f"saved={img_path}")
    print(f"mask={mask_path}")
    print(f"csv={CSV_PATH}")

    node.destroy_node()
    try:
        rclpy.shutdown()
    except Exception:
        pass


if __name__ == "__main__":
    main()
