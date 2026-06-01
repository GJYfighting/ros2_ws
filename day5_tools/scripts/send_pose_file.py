#!/usr/bin/env python3
import argparse
import yaml
import rclpy

from rclpy.node import Node
from rclpy.action import ActionClient
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


ARM_JOINTS = ["joint1", "joint2", "joint3", "joint4", "joint5"]
GRIPPER_JOINTS = ["r_joint"]


class PoseFileSender(Node):
    def __init__(self):
        super().__init__("pose_file_sender")

        self.arm_client = ActionClient(
            self,
            FollowJointTrajectory,
            "/arm_controller/follow_joint_trajectory"
        )

        self.gripper_client = ActionClient(
            self,
            FollowJointTrajectory,
            "/gripper_controller/follow_joint_trajectory"
        )

    def send_trajectory(self, client, joint_names, positions, duration_sec):
        self.get_logger().info(f"Waiting for action server: {client._action_name}")

        if not client.wait_for_server(timeout_sec=10.0):
            raise RuntimeError(f"Action server not available: {client._action_name}")

        goal = FollowJointTrajectory.Goal()

        traj = JointTrajectory()
        traj.joint_names = joint_names

        point = JointTrajectoryPoint()
        point.positions = [float(x) for x in positions]

        sec = int(duration_sec)
        nanosec = int((float(duration_sec) - sec) * 1e9)
        point.time_from_start = Duration(sec=sec, nanosec=nanosec)

        traj.points = [point]
        goal.trajectory = traj

        self.get_logger().info(
            f"Sending goal to {client._action_name}: "
            f"joints={joint_names}, positions={point.positions}"
        )

        send_future = client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_future)

        goal_handle = send_future.result()

        if goal_handle is None:
            raise RuntimeError(f"Failed to send goal to {client._action_name}")

        if not goal_handle.accepted:
            raise RuntimeError(f"Goal rejected by {client._action_name}")

        self.get_logger().info("Goal accepted, waiting for result...")

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)

        result = result_future.result()
        self.get_logger().info(f"Action result received from {client._action_name}")

        return result


def load_pose_from_yaml(yaml_path, key):
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)

    if data is None:
        raise RuntimeError(f"YAML file is empty: {yaml_path}")

    # 支持两种格式：
    # 格式1：
    # observe_pose:
    #   arm: [...]
    #   gripper: 1.4
    #
    # 格式2：
    # poses:
    #   observe_pose:
    #     arm: [...]
    #     gripper: 1.4

    if key in data:
        pose = data[key]
    elif "poses" in data and key in data["poses"]:
        pose = data["poses"][key]
    else:
        raise KeyError(
            f"Cannot find key '{key}' in {yaml_path}. "
            f"Available top-level keys: {list(data.keys())}"
        )

    if "arm" not in pose:
        raise KeyError(f"Pose '{key}' does not contain 'arm' field")

    if "gripper" not in pose:
        raise KeyError(f"Pose '{key}' does not contain 'gripper' field")

    arm = pose["arm"]
    gripper = pose["gripper"]
    duration = float(pose.get("duration", 3.0))

    if len(arm) != 5:
        raise ValueError(f"'arm' must contain 5 joint values, got {len(arm)}")

    return arm, gripper, duration


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml", required=True, help="pose yaml file")
    parser.add_argument("--key", required=True, help="pose key, e.g. observe_pose")
    parser.add_argument("--arm-only", action="store_true")
    parser.add_argument("--gripper-only", action="store_true")
    args = parser.parse_args()

    arm, gripper, duration = load_pose_from_yaml(args.yaml, args.key)

    rclpy.init()
    node = PoseFileSender()

    try:
        if not args.gripper_only:
            node.send_trajectory(
                node.arm_client,
                ARM_JOINTS,
                arm,
                duration
            )

        if not args.arm_only:
            node.send_trajectory(
                node.gripper_client,
                GRIPPER_JOINTS,
                [float(gripper)],
                1.0
            )

        node.get_logger().info("Pose command finished.")

    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
