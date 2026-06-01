#!/usr/bin/env python3
import argparse
import json
import math
import sys
import time
from pathlib import Path

import rclpy

SCRIPT_DIR = Path.home() / "ros2_ws/day7_baseline/scripts"
sys.path.insert(0, str(SCRIPT_DIR))

import day7_rule_baseline_once as base
import day7_hold_close_lift_probe as hold


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trial-id", default=f"hold_close_multilift_{int(time.time())}")
    parser.add_argument("--skip-home", action="store_true")

    parser.add_argument("--pre-z-base", type=float, default=0.22)
    parser.add_argument("--down-z-base", type=float, default=0.020)
    parser.add_argument("--lift-seq", default="0.035,0.045,0.060,0.090")

    parser.add_argument("--x-offset", type=float, default=0.0)
    parser.add_argument("--y-offset", type=float, default=0.0)

    parser.add_argument("--close-seq", default="1.2,1.1,1.0,0.95,0.9")
    parser.add_argument("--gripper-step-dt", type=float, default=0.8)
    parser.add_argument("--gripper-hold-s", type=float, default=45.0)
    parser.add_argument("--wait-before-lift", type=float, default=4.0)
    parser.add_argument("--wait-for-close-timeout", type=float, default=45.0)
    parser.add_argument("--wait-for-close-tol", type=float, default=0.03)

    parser.add_argument("--down-duration", type=float, default=3.0)
    parser.add_argument("--lift-duration", type=float, default=2.5)
    parser.add_argument("--hold-after-each-lift", type=float, default=0.8)

    parser.add_argument("--clamp-dz", type=float, default=0.012)
    parser.add_argument("--success-dz", type=float, default=0.035)
    parser.add_argument("--max-clamp-xy", type=float, default=0.035)

    args = parser.parse_args()

    close_values = [float(x.strip()) for x in args.close_seq.split(",") if x.strip()]
    lift_zs = [float(x.strip()) for x in args.lift_seq.split(",") if x.strip()]

    result_dir = Path.home() / "ros2_ws/day7_baseline/results"
    result_dir.mkdir(parents=True, exist_ok=True)

    log = {
        "trial_id": args.trial_id,
        "mode": "hold_close_multilift_probe",
        "params": vars(args),
        "close_values": close_values,
        "lift_zs": lift_zs,
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

            truth_before = hold.get_truth_retry(args.trial_id + "_before")
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
            }

            lift_names = []
            for i, z in enumerate(lift_zs):
                name = f"lift_{i+1}"
                lift_names.append(name)
                waypoints[name] = [p_base[0], p_base[1], z]

            log["target_world_xyz"] = p_world
            log["target_base_xy"] = [p_base[0], p_base[1]]
            log["waypoints_base"] = waypoints

            q_targets = {}
            seed = base.OBSERVE

            for name in ["pregrasp", "down"] + lift_names:
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

            truth_down = hold.get_truth_retry(args.trial_id + "_after_down")
            log["truth_after_down"] = truth_down
            log["delta_after_down"] = hold.delta(truth_before, truth_down)

            log["stage"] = "send_gripper_hold_goal"
            hold.send_gripper_hold_goal(
                node,
                close_values,
                step_dt=args.gripper_step_dt,
                hold_s=args.gripper_hold_s,
            )

            t0 = time.time()
            actual_r_before_lift = None
            target_r = close_values[-1]
            close_reached = False
            while time.time() - t0 < args.wait_for_close_timeout:
                rclpy.spin_once(node, timeout_sec=0.05)
                actual_r_before_lift = hold.read_joint_position(node, "r_joint", timeout_s=0.1)
                if actual_r_before_lift is not None and actual_r_before_lift <= target_r + args.wait_for_close_tol:
                    close_reached = True
                    break

            t_wait = time.time()
            while time.time() - t_wait < args.wait_before_lift:
                rclpy.spin_once(node, timeout_sec=0.05)

            actual_r_before_lift = hold.read_joint_position(node, "r_joint", timeout_s=1.0)
            truth_close = hold.get_truth_retry(args.trial_id + "_after_hold_close")
            close_delta = hold.delta(truth_before, truth_close)

            log["actual_r_joint_before_lift"] = actual_r_before_lift
            log["close_reached_before_lift"] = close_reached
            log["waited_for_close_wall_s"] = time.time() - t0
            log["truth_after_hold_close"] = truth_close
            log["delta_after_hold_close"] = close_delta

            log["lift_results"] = []

            for name in lift_names:
                log["stage"] = name

                node.arm(q_targets[name], duration=args.lift_duration)

                t1 = time.time()
                while time.time() - t1 < args.hold_after_each_lift:
                    rclpy.spin_once(node, timeout_sec=0.05)

                actual_r = hold.read_joint_position(node, "r_joint", timeout_s=1.0)
                truth_lift = hold.get_truth_retry(args.trial_id + f"_after_{name}")
                d = hold.delta(truth_before, truth_lift)

                item = {
                    "name": name,
                    "target_z_base": waypoints[name][2],
                    "actual_r_joint": actual_r,
                    "truth": truth_lift,
                    "delta": d,
                    "clamped_at_step": (
                        d["dz"] >= args.clamp_dz
                        and d["abs_xy"] <= args.max_clamp_xy
                    ),
                    "success_at_step": (
                        d["dz"] >= args.success_dz
                        or float(truth_lift["z"]) > 0.825
                    ),
                }

                log["lift_results"].append(item)

            final = log["lift_results"][-1]
            final_delta = final["delta"]

            log["actual_r_joint_after_lift"] = final["actual_r_joint"]
            log["truth_after"] = final["truth"]
            log["delta_final"] = final_delta

            log["clamped"] = any(x["clamped_at_step"] for x in log["lift_results"])
            log["success"] = any(x["success_at_step"] for x in log["lift_results"])

            if log["success"]:
                log["failure_type"] = "success"
                log["diagnosis"] = "grasp_success"
            elif log["clamped"]:
                log["failure_type"] = "clamped_but_slipped_later"
                log["diagnosis"] = "object followed at early lift step, then slipped; use smaller staged lift or stronger close"
            else:
                log["failure_type"] = "partial_contact_not_clamped"
                log["diagnosis"] = "object moved slightly but did not reach clamp threshold; tune close endpoint and x/y offset"

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
