#!/usr/bin/env python3
import time
from pathlib import Path

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image


RGB_TOPIC = "/depth_cam/rgbd/image"
OUT_IMAGE = "/home/ubuntu/ros2_ws/day5_method_a/results/debug/rgbd_image_raw_check.png"


def ros_rgb_to_numpy(msg):
    if msg.encoding not in ["rgb8", "bgr8", "rgba8", "bgra8"]:
        raise ValueError(f"Unsupported encoding: {msg.encoding}")

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
        raise ValueError(f"Unsupported encoding: {msg.encoding}")

    return rgb.copy()


class ImageOnce(Node):
    def __init__(self):
        super().__init__("check_rgbd_image_stats")
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
    rclpy.init()
    node = ImageOnce()

    print("[INFO] Waiting for one RGBD image...")
    start = time.time()

    while time.time() - start < 8.0:
        rclpy.spin_once(node, timeout_sec=0.1)
        if node.msg is not None:
            break

    if node.msg is None:
        print("[ERROR] No image received.")
        node.destroy_node()
        rclpy.shutdown()
        return

    msg = node.msg
    print("[INFO] width:", msg.width)
    print("[INFO] height:", msg.height)
    print("[INFO] encoding:", msg.encoding)
    print("[INFO] frame_id:", msg.header.frame_id)
    print("[INFO] step:", msg.step)

    rgb = ros_rgb_to_numpy(msg)

    print("[INFO] image min:", rgb.min(axis=(0, 1)))
    print("[INFO] image max:", rgb.max(axis=(0, 1)))
    print("[INFO] image mean:", rgb.mean(axis=(0, 1)))
    print("[INFO] image std:", rgb.std(axis=(0, 1)))

    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    Path(OUT_IMAGE).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(OUT_IMAGE, bgr)
    print("[INFO] saved:", OUT_IMAGE)

    node.destroy_node()
    try:
        rclpy.shutdown()
    except Exception:
        pass


if __name__ == "__main__":
    main()
