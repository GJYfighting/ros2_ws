#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, CameraInfo

class CameraProbe(Node):
    def __init__(self):
        super().__init__('camera_probe')

        self.rgb_count = 0
        self.depth_count = 0
        self.info_count = 0

        self.last_rgb = None
        self.last_depth = None
        self.last_info = None

        self.create_subscription(
            Image,
            '/depth_cam/rgb/image_raw',
            self.rgb_cb,
            qos_profile_sensor_data
        )
        self.create_subscription(
            Image,
            '/depth_cam/depth/image_raw',
            self.depth_cb,
            qos_profile_sensor_data
        )
        self.create_subscription(
            CameraInfo,
            '/depth_cam/rgb/camera_info',
            self.info_cb,
            qos_profile_sensor_data
        )

        self.create_timer(2.0, self.report)
        self.get_logger().info('camera probe started')

    def rgb_cb(self, msg):
        self.rgb_count += 1
        self.last_rgb = (msg.width, msg.height, msg.encoding, msg.header.frame_id)

    def depth_cb(self, msg):
        self.depth_count += 1
        self.last_depth = (msg.width, msg.height, msg.encoding, msg.header.frame_id)

    def info_cb(self, msg):
        self.info_count += 1
        self.last_info = (msg.width, msg.height, msg.header.frame_id)

    def report(self):
        self.get_logger().info(
            f'RGB count={self.rgb_count}, last={self.last_rgb}; '
            f'DEPTH count={self.depth_count}, last={self.last_depth}; '
            f'INFO count={self.info_count}, last={self.last_info}'
        )
        self.rgb_count = 0
        self.depth_count = 0
        self.info_count = 0

def main():
    rclpy.init()
    node = CameraProbe()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
