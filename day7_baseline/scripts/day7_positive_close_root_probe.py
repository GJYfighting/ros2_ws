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


def get_truth_retry(tag, attempts=6, wait_s=0.5):
    last_error = None
    for i in range(attempts):
        try:
            return base.get_truth(tag)
        except Exception as e:
            last_error = e
            time.sleep(wait_s)
    raise RuntimeError(f"get_truth_retry_failed tag={tag}: {last_error}")


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


def send_gripper_timed(node, q, duration=0.8, settle=0.8):
    """
    发送夹爪目标，但不等待 FollowJointTrajectory result。
    原因：夹爪接触物体后可能无法精确到达目标位置，等待 result 会卡很久。
    """
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
    goal.goal_time_tolerance = base.duration_msg(0.5)

    send_future = node.gripper_client.send_goal_async(goal)
    rclpy.spin_until_future_complete(node, send_future, timeout_sec=2.0)

    if not send_future.done():
        raise RuntimeError(f"gripper_goal_send_timeout q={q}")

    goal_handle = send_future.result()
    if goal_handle is None or not goal_handle.accepted:
        raise RuntimeError(f"gripper_goal_rejected q={q}")

    t0 = time.time()
    while time.time() - t0 < duration + settle:
        rclpy.spin_once(node, timeout_sec=0.05)

    return read_joint_position(node, "r_joint", timeout_s=1.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trial-id", default=f"root_probe_{int(time.time())}")
    parser.add_argument("--skip-home", action="store_true")

    parser.add_argument("--pre-z-base", type=float, default=0.22)
    parser.add_argument("--down-z-base", type=float, default=0.020)
    parser.add_argument("--lift-z-base", type=float, default=0.060)

    parser.add_argument("--x-offset", type=float, default=0.0)
    parser.add_argument("--y-offset", type=float, default=0.0)

    parser.add_argument("--close-seq", default="1.1,1.0,0.9,0.8,0.7,0.6")
    parser.add_argument("--gripper-duration", type=float, default=0.8)
    parser.add_argument("--gripper-settle", type=float, default=0.8)

    parser.add_argument("--down-duration", type=float, default=3.0)
    parser.add_argument("--lift-duration", type=float, default=3.0)
    parser.add_argument("--hold-after-lift", type=float, default=1.0)

    parser.add_argument("--clamp-dz", type=float, default=0.012)
    parser.add_argument("--success-dz", type=float, default=0.035)
    parser.add_argument("--max-clamp-xy", type=float, default=0.035)

    args = parser.parse_args()

    result_dir = Path.home() / "ros2_ws/day7_baseline/results"
    result_dir.mkdir(parents=True, exist_ok=True)

    close_values = [float(x.strip()) for x in args.close_seq.split(",") if x.strip()]
    close_end = close_values[-1]

    log = {
        "trial_id": args.trial_id,
        "mode": "positive_close_root_probe",
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
            send_gripper_timed(node, base.GRIPPER_OPEN, duration=0.8, settle=0.5)
            node.arm(base.OBSERVE, duration=3.0)
            send_gripper_timed(node, base.GRIPPER_OPEN, duration=0.8, settle=0.5)

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

            log["close_results"] = []

            for i, cv in enumerate(close_values):
                log["stage"] = f"close_{i}_{cv}"
                actual_r = send_gripper_timed(
                    node,
                    cv,
                    duration=args.gripper_duration,
                    settle=args.gripper_settle,
                )

                truth_c = get_truth_retry(args.trial_id + f"_after_close_{i}")
                d = delta(truth_before, truth_c)

                log["close_results"].append({
                    "i": i,
                    "target_r_joint": cv,
                    "actual_r_joint": actual_r,
                    "truth": truth_c,
                    "delta": d,
                })

            truth_close = log["close_results"][-1]["truth"]
            close_delta = log["close_results"][-1]["delta"]
            actual_close = log["close_results"][-1]["actual_r_joint"]

            log["truth_after_close"] = truth_close
            log["delta_after_close"] = close_delta
            log["actual_r_joint_after_close"] = actual_close

            log["stage"] = "small_lift"

            # 抬升前再发一次最终 close 目标，但仍然不等待 result。
            send_gripper_timed(
                node,
                close_end,
                duration=0.5,
                settle=0.2,
            )

            node.arm(q_targets["lift"], duration=args.lift_duration)
            time.sleep(args.hold_after_lift)

            truth_lift = get_truth_retry(args.trial_id + "_after_lift")
            lift_delta = delta(truth_before, truth_lift)

            log["truth_after_lift"] = truth_lift
            log["delta_after_lift"] = lift_delta
            log["truth_after"] = truth_lift
            log["delta_final"] = lift_delta

            reached_close_target = None
            if actual_close is not None:
                reached_close_target = abs(actual_close - close_end) <= 0.08

            close_moved_block = (
                close_delta["abs_xy"] >= 0.003
                or abs(close_delta["dz"]) >= 0.003
            )

            clamped = (
                lift_delta["dz"] >= args.clamp_dz
                and lift_delta["abs_xy"] <= args.max_clamp_xy
            )

            success = (
                lift_delta["dz"] >= args.success_dz
                or float(truth_lift["z"]) > 0.825
            )

            log["reached_close_target"] = reached_close_target
            log["close_moved_block"] = close_moved_block
            log["clamped"] = bool(clamped)
            log["success"] = bool(success)

            if success:
                log["failure_type"] = "success"
                log["diagnosis"] = "grasp_success"
            elif clamped:
                log["failure_type"] = "clamped_but_not_high_enough"
                log["diagnosis"] = "gripper_can_clamp; improve lift trajectory or hold"
            elif close_moved_block:
                log["failure_type"] = "contact_but_not_clamped"
                log["diagnosis"] = "contact_exists; tune close endpoint, x/y offset, or friction"
            else:
                log["failure_type"] = "not_clamped"

                if reached_close_target is True:
                    log["diagnosis"] = (
                        "r_joint reached target but block did not move. "
                        "If visual mesh penetrates the block, gripper/block collision contact is ineffective."
                    )
                elif reached_close_target is False:
                    log["diagnosis"] = (
                        "r_joint did not reach target, but block still did not move. "
                        "Likely gripper contact/controller/contact model abnormal."
                    )
                else:
                    log["diagnosis"] = (
                        "cannot read r_joint; check /joint_states and gripper_controller."
                    )

            log["stage"] = "done"

        except Exception as e:
            log["error"] = str(e)
            if "get_truth_retry_failed" in str(e):
                log["failure_type"] = "truth_topic_fail"
                log["diagnosis"] = "Gazebo dynamic_pose topic failed or simulation paused/frozen."
            elif "ik_fail" in str(e):
                log["failure_type"] = "ik_fail"
                log["diagnosis"] = "IK target unreachable."
            else:
                log["failure_type"] = "runtime_error"
                log["diagnosis"] = "runtime error; inspect error field."

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
