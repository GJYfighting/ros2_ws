#!/usr/bin/env python3
import math
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from rclpy.qos import qos_profile_sensor_data


class CameraInfoRepublisher(Node):
    def __init__(self):
        super().__init__('camera_info_republisher')

        self.hfov = 1.047
        self.default_width = 640
        self.default_height = 400

        self.rgb_topic = '/depth_cam/rgbd/image'
        self.info_topic = '/depth_cam/rgbd/camera_info'

        self.pub = self.create_publisher(
            CameraInfo,
            self.info_topic,
            10
        )

        self.sub = self.create_subscription(
            Image,
            self.rgb_topic,
            self.image_cb,
            qos_profile_sensor_data
        )

        self.get_logger().info(
            f'camera_info republisher started: {self.rgb_topic} -> {self.info_topic}'
        )

    def image_cb(self, msg: Image):
        width = int(msg.width) if msg.width > 0 else self.default_width
        height = int(msg.height) if msg.height > 0 else self.default_height

        fx = width / (2.0 * math.tan(self.hfov / 2.0))
        fy = fx
        cx = width / 2.0
        cy = height / 2.0

        ci = CameraInfo()
        ci.header.stamp = msg.header.stamp

        # 关键：统一用 URDF 里的 optical-like frame
        # 后续 3D 回投得到的点就按 depth_cam_frame 解释。
        ci.header.frame_id = 'depth_cam_frame'

        ci.width = width
        ci.height = height
        ci.distortion_model = 'plumb_bob'
        ci.d = [0.0, 0.0, 0.0, 0.0, 0.0]

        ci.k = [
            fx, 0.0, cx,
            0.0, fy, cy,
            0.0, 0.0, 1.0
        ]

        ci.r = [
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0
        ]

        ci.p = [
            fx, 0.0, cx, 0.0,
            0.0, fy, cy, 0.0,
            0.0, 0.0, 1.0, 0.0
        ]

        self.pub.publish(ci)


def main():
    rclpy.init()
    node = CameraInfoRepublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    try:
        rclpy.shutdown()
    except Exception:
        pass


if __name__ == '__main__':
    main()
