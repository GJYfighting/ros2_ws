#!/usr/bin/env python3
import time
import yaml
import rclpy
from rclpy.node import Node
import tf2_ros

class TFProbe(Node):
    def __init__(self):
        super().__init__("tf_probe")
        self.buffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.buffer, self)

def main():
    rclpy.init()
    node = TFProbe()

    print("Waiting for TF...")
    end = time.time() + 3.0
    while time.time() < end:
        rclpy.spin_once(node, timeout_sec=0.1)

    frames_yaml = node.buffer.all_frames_as_yaml()
    print("\n===== TF FRAMES =====")
    print(frames_yaml)

    base_candidates = ["base_link", "robot/base_link"]
    camera_candidates = [
        "robot/link4/robot_cam_rgb",
        "robot/link4/robot_cam_depth",
        "depth_cam_frame",
        "depth_cam_link",
        "robot/depth_cam_frame",
        "robot/depth_cam_link",
    ]

    print("\n===== TRY TRANSFORMS: base <- camera =====")
    found = False
    for base in base_candidates:
        for cam in camera_candidates:
            try:
                tf = node.buffer.lookup_transform(base, cam, rclpy.time.Time())
                t = tf.transform.translation
                q = tf.transform.rotation
                print(f"[OK] target={base}, source={cam}")
                print(f"     translation=({t.x:.4f}, {t.y:.4f}, {t.z:.4f})")
                print(f"     rotation=({q.x:.4f}, {q.y:.4f}, {q.z:.4f}, {q.w:.4f})")
                found = True
            except Exception:
                pass

    if not found:
        print("[WARN] No base<-camera transform found.")

    node.destroy_node()
    try:
        rclpy.shutdown()
    except Exception:
        pass

if __name__ == "__main__":
    main()
