#!/usr/bin/env python3
import argparse
import json
import sys
import time
from pathlib import Path

import rclpy
from sensor_msgs.msg import JointState

SCRIPT_DIR = Path.home() / "ros2_ws/day7_baseline/scripts"
sys.path.insert(0, str(SCRIPT_DIR))

import day7_rule_baseline_once as base


def read_joint_position(node, joint_name="r_joint", timeout_s=3.0):
    result = {"value": None}

    def cb(msg):
        if joint_name in msg.name:
            idx = list(msg.name).index(joint_name)
            result["value"] = float(msg.position[idx])

    sub = node.create_subscription(JointState, "/joint_states", cb, 10)

    t0 = time.time()
    while time.time() - t0 < timeout_s:
        rclpy.spin_once(node, timeout_sec=0.1)
        if result["value"] is not None:
            break

    node.destroy_subscription(sub)
    return result["value"]


def send_gripper_timed(node, q, duration=3.0, settle=4.0):
    if not node.gripper_client.wait_for_server(timeout_sec=5.0):
        raise RuntimeError("gripper action server not available")

    goal = base.FollowJointTrajectory.Goal()

    traj = base.JointTrajectory()
    traj.joint_names = list(base.GRIPPER_JOINTS)

    pt = base.JointTrajectoryPoint()
    pt.positions = base.gripper_positions_from_r(q)
    pt.time_from_start = base.duration_msg(duration)

    traj.points = [pt]
    goal.trajectory = traj
    goal.goal_time_tolerance = base.duration_msg(2.0)

    fut = node.gripper_client.send_goal_async(goal)
    rclpy.spin_until_future_complete(node, fut, timeout_sec=3.0)

    if not fut.done():
        raise RuntimeError(f"send goal timeout: q={q}")

    gh = fut.result()
    if gh is None or not gh.accepted:
        raise RuntimeError(f"goal rejected: q={q}")

    t0 = time.time()
    while time.time() - t0 < duration + settle:
        rclpy.spin_once(node, timeout_sec=0.05)

    return read_joint_position(node, "r_joint", timeout_s=2.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", default="1.4,1.2,1.1,1.0,0.9,0.8,0.7")
    parser.add_argument("--duration", type=float, default=3.0)
    parser.add_argument("--settle", type=float, default=4.0)
    parser.add_argument("--tol", type=float, default=0.08)
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    targets = [float(x.strip()) for x in args.targets.split(",") if x.strip()]

    rclpy.init()
    node = base.MotionAndIK()

    rows = []

    try:
        for q in targets:
            actual = send_gripper_timed(
                node,
                q,
                duration=args.duration,
                settle=args.settle,
            )

            err = None if actual is None else actual - q
            reached = False if actual is None else abs(err) <= args.tol

            row = {
                "target": q,
                "actual": actual,
                "error": err,
                "reached": reached,
            }
            rows.append(row)

            print(
                f"target={q:+.3f} "
                f"actual={actual if actual is not None else None} "
                f"error={err if err is not None else None} "
                f"reached={reached}"
            )

    finally:
        node.destroy_node()
        rclpy.shutdown()

    result = {
        "mode": "gripper_free_reach_test",
        "targets": targets,
        "duration": args.duration,
        "settle": args.settle,
        "tol": args.tol,
        "rows": rows,
    }

    if args.out:
        p = Path(args.out)
    else:
        p = Path.home() / "ros2_ws/day7_baseline/results/day7_gripper_free_reach_test.json"

    p.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\n[DAY7] saved: {p}")


if __name__ == "__main__":
    main()
