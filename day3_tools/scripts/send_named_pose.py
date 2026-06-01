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

class NamedPoseSender(Node):
    def __init__(self):
        super().__init__("named_pose_sender")
        self.arm_client = ActionClient(
            self, FollowJointTrajectory, "/arm_controller/follow_joint_trajectory"
        )
        self.gripper_client = ActionClient(
            self, FollowJointTrajectory, "/gripper_controller/follow_joint_trajectory"
        )

    def send_goal(self, client, joint_names, positions, duration_sec):
        if not client.wait_for_server(timeout_sec=5.0):
            raise RuntimeError(f"Action server not available: {client._action_name}")

        goal = FollowJointTrajectory.Goal()
        traj = JointTrajectory()
        traj.joint_names = joint_names

        point = JointTrajectoryPoint()
        point.positions = list(positions)
        sec = int(duration_sec)
        nanosec = int((duration_sec - sec) * 1e9)
        point.time_from_start = Duration(sec=sec, nanosec=nanosec)

        traj.points = [point]
        goal.trajectory = traj

        send_future = client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_future)
        goal_handle = send_future.result()

        if goal_handle is None or not goal_handle.accepted:
            raise RuntimeError(f"Goal rejected by {client._action_name}")

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        return result_future.result()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml", required=True, help="path to named_poses.yaml")
    parser.add_argument("--pose", required=True, help="pose name in YAML")
    parser.add_argument("--arm-only", action="store_true")
    parser.add_argument("--gripper-only", action="store_true")
    args = parser.parse_args()

    with open(args.yaml, "r") as f:
        data = yaml.safe_load(f)

    poses = data["poses"]
    if args.pose not in poses:
        raise KeyError(f"pose '{args.pose}' not found in YAML")

    pose = poses[args.pose]
    arm = pose["arm"]
    gripper = pose["gripper"]
    duration = float(pose.get("duration", 3.0))

    rclpy.init()
    node = NamedPoseSender()

    try:
        if not args.gripper_only:
            node.get_logger().info(f"Sending arm pose: {args.pose} -> {arm}")
            node.send_goal(node.arm_client, ARM_JOINTS, arm, duration)

        if not args.arm_only:
            node.get_logger().info(f"Sending gripper pose: {args.pose} -> {gripper}")
            node.send_goal(node.gripper_client, GRIPPER_JOINTS, [gripper], 1.5)

        node.get_logger().info("Done.")
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass

if __name__ == "__main__":
    main()
