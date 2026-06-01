#!/usr/bin/env python3
import argparse
import json
import math
import sys
import time
from pathlib import Path

import rclpy
from sensor_msgs.msg import JointState

SCRIPT_DIR = Path.home() / "ros2_ws/day7_baseline/scripts"
sys.path.insert(0, str(SCRIPT_DIR))

import day7_rule_baseline_once as base


def delta(before, after):
    dx = float(after["x"]) - float(before["x"])
    dy = float(after["y"]) - float(before["y"])
    dz = float(after["z"]) - float(before["z"])
    return {
        "dx": dx,
        "dy": dy,
        "dz": dz,
        "abs_xy": math.sqrt(dx * dx + dy * dy),
    }


def get_truth_retry(tag, attempts=8, wait_s=0.4):
    last = None
    for _ in range(attempts):
        try:
            return base.get_truth(tag)
        except Exception as e:
            last = e
            time.sleep(wait_s)
    raise RuntimeError(f"get_truth_retry_failed tag={tag}: {last}")


def read_joint_position(node, joint_name="r_joint", timeout_s=2.0):
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


def send_gripper_hold_goal(node, close_values, step_dt=0.8, hold_s=25.0):
    """
    发送一个长时间夹持轨迹：
      q1 -> q2 -> ... -> q_end
      然后在 q_end 保持 hold_s 秒。
    不等待 result，让 gripper goal 在 arm lift 过程中继续 active。
    """
    if not node.gripper_client.wait_for_server(timeout_sec=5.0):
        raise RuntimeError("gripper action server not available")

    goal = base.FollowJointTrajectory.Goal()

    traj = base.JointTrajectory()
    traj.joint_names = list(base.GRIPPER_JOINTS)

    points = []
    t = 0.0

    for q in close_values:
        t += step_dt
        pt = base.JointTrajectoryPoint()
        pt.positions = base.gripper_positions_from_r(q)
        pt.time_from_start = base.duration_msg(t)
        points.append(pt)

    # 关键：最后一个 close 目标保持很久，保证抬升时仍然在夹。
    t += hold_s
    pt = base.JointTrajectoryPoint()
    pt.positions = base.gripper_positions_from_r(close_values[-1])
    pt.time_from_start = base.duration_msg(t)
    points.append(pt)

    traj.points = points
    goal.trajectory = traj

    # 不让 action 很快因为到不了目标而结束。
    goal.goal_time_tolerance = base.duration_msg(hold_s + 5.0)

    fut = node.gripper_client.send_goal_async(goal)
    rclpy.spin_until_future_complete(node, fut, timeout_sec=3.0)

    if not fut.done():
        raise RuntimeError("gripper_hold_goal_send_timeout")

    gh = fut.result()
    if gh is None or not gh.accepted:
        raise RuntimeError("gripper_hold_goal_rejected")

    return gh


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trial-id", default=f"hold_close_lift_{int(time.time())}")
    parser.add_argument("--skip-home", action="store_true")

    parser.add_argument("--pre-z-base", type=float, default=0.22)
    parser.add_argument("--down-z-base", type=float, default=0.020)
    parser.add_argument("--lift-z-base", type=float, default=0.060)

    parser.add_argument("--x-offset", type=float, default=0.0)
    parser.add_argument("--y-offset", type=float, default=0.0)

    parser.add_argument("--close-seq", default="1.2,1.1,1.0,0.95,0.9")
    parser.add_argument("--gripper-step-dt", type=float, default=0.8)
    parser.add_argument("--gripper-hold-s", type=float, default=25.0)
    parser.add_argument("--wait-before-lift", type=float, default=4.0)

    parser.add_argument("--down-duration", type=float, default=3.0)
    parser.add_argument("--lift-duration", type=float, default=4.0)
    parser.add_argument("--hold-after-lift", type=float, default=1.0)

    parser.add_argument("--clamp-dz", type=float, default=0.012)
    parser.add_argument("--success-dz", type=float, default=0.035)
    parser.add_argument("--max-clamp-xy", type=float, default=0.035)

    args = parser.parse_args()

    close_values = [float(x.strip()) for x in args.close_seq.split(",") if x.strip()]

    result_dir = Path.home() / "ros2_ws/day7_baseline/results"
    result_dir.mkdir(parents=True, exist_ok=True)

    log = {
        "trial_id": args.trial_id,
        "mode": "hold_close_lift_probe",
        "params": vars(args),
        "close_values": close_values,
        "success": False,
        "clamped": False,
        "failure_type": "unknown",
        "diagnosis": "unknown",
        "t_start": time.time(),
    }

    rclpy.init()
    node = base.MotionAndIK()

    try:
        try:
            if not args.skip_home:
                log["stage"] = "home"
                node.arm(base.HOME, duration=3.0)

            log["stage"] = "observe"
            node.gripper(base.GRIPPER_OPEN, duration=0.8)
            node.arm(base.OBSERVE, duration=3.0)
            node.gripper(base.GRIPPER_OPEN, duration=0.8)

            truth_before = get_truth_retry(args.trial_id + "_before")
            log["truth_before"] = truth_before

            p_world = [
                float(truth_before["x"]),
                float(truth_before["y"]),
                float(truth_before["z"]),
            ]
            p_base = base.world_to_base(p_world)
            p_base[0] += args.x_offset
            p_base[1] += args.y_offset

            waypoints = {
                "pregrasp": [p_base[0], p_base[1], args.pre_z_base],
                "down": [p_base[0], p_base[1], args.down_z_base],
                "lift": [p_base[0], p_base[1], args.lift_z_base],
            }

            log["target_world_xyz"] = p_world
            log["target_base_xy"] = [p_base[0], p_base[1]]
            log["waypoints_base"] = waypoints

            q_targets = {}
            seed = base.OBSERVE

            for name in ["pregrasp", "down", "lift"]:
                q, status = node.compute_ik(
                    waypoints[name],
                    seed,
                    ik_link_name="end_effector_link",
                )

                if q is None:
                    q, status = node.compute_ik(
                        waypoints[name],
                        seed,
                        ik_link_name="link5",
                    )

                if q is None:
                    raise RuntimeError(f"ik_fail_{name}_{status}")

                q_targets[name] = q
                seed = q

            log["q_targets"] = q_targets

            log["stage"] = "pregrasp"
            node.arm(q_targets["pregrasp"], duration=2.5)

            log["stage"] = "down"
            node.arm(q_targets["down"], duration=args.down_duration)
            time.sleep(0.5)

            truth_down = get_truth_retry(args.trial_id + "_after_down")
            log["truth_after_down"] = truth_down
            log["delta_after_down"] = delta(truth_before, truth_down)

            log["stage"] = "send_gripper_hold_goal"
            send_gripper_hold_goal(
                node,
                close_values,
                step_dt=args.gripper_step_dt,
                hold_s=args.gripper_hold_s,
            )

            # 等待夹爪进入接触状态，但不等 gripper action 完成。
            t0 = time.time()
            while time.time() - t0 < args.wait_before_lift:
                rclpy.spin_once(node, timeout_sec=0.05)

            actual_r_before_lift = read_joint_position(node, "r_joint", timeout_s=1.0)
            truth_close = get_truth_retry(args.trial_id + "_after_hold_close")
            close_delta = delta(truth_before, truth_close)

            log["actual_r_joint_before_lift"] = actual_r_before_lift
            log["truth_after_hold_close"] = truth_close
            log["delta_after_hold_close"] = close_delta

            log["stage"] = "lift_while_gripper_goal_active"
            node.arm(q_targets["lift"], duration=args.lift_duration)

            t1 = time.time()
            while time.time() - t1 < args.hold_after_lift:
                rclpy.spin_once(node, timeout_sec=0.05)

            actual_r_after_lift = read_joint_position(node, "r_joint", timeout_s=1.0)
            truth_lift = get_truth_retry(args.trial_id + "_after_lift")
            lift_delta = delta(truth_before, truth_lift)

            log["actual_r_joint_after_lift"] = actual_r_after_lift
            log["truth_after_lift"] = truth_lift
            log["delta_after_lift"] = lift_delta
            log["truth_after"] = truth_lift
            log["delta_final"] = lift_delta

            clamped = (
                lift_delta["dz"] >= args.clamp_dz
                and lift_delta["abs_xy"] <= args.max_clamp_xy
            )

            success = (
                lift_delta["dz"] >= args.success_dz
                or float(truth_lift["z"]) > 0.825
            )

            log["clamped"] = bool(clamped)
            log["success"] = bool(success)

            if success:
                log["failure_type"] = "success"
                log["diagnosis"] = "grasp_success"
            elif clamped:
                log["failure_type"] = "clamped_but_not_high_enough"
                log["diagnosis"] = "hold-close works; improve lift height or duration"
            else:
                log["failure_type"] = "not_clamped"
                log["diagnosis"] = (
                    "gripper stayed in contact but block did not follow. "
                    "If actual_r remains around contact value, remaining issue is contact geometry/friction, not free gripper motion."
                )

            log["stage"] = "done"

        except Exception as e:
            log["error"] = str(e)
            if "ik_fail" in str(e):
                log["failure_type"] = "ik_fail"
            else:
                log["failure_type"] = "runtime_error"
            log["diagnosis"] = "inspect error field"

        finally:
            log["t_end"] = time.time()
            log["duration_s"] = log["t_end"] - log["t_start"]

            out = result_dir / f"{args.trial_id}.json"
            out.write_text(json.dumps(log, indent=2, ensure_ascii=False))

            print(json.dumps(log, indent=2, ensure_ascii=False))
            print(f"\n[DAY7] saved: {out}")

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
