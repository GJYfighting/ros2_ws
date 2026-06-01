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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trial-id", default=f"clamp_lift_{int(time.time())}")
    parser.add_argument("--skip-home", action="store_true")

    parser.add_argument("--pre-z-base", type=float, default=0.22)
    parser.add_argument("--down-z-base", type=float, default=0.020)
    parser.add_argument("--lift-seq", default="0.060,0.100,0.160")

    parser.add_argument("--x-offset", type=float, default=0.0)
    parser.add_argument("--y-offset", type=float, default=0.0)

    parser.add_argument("--close-seq", default="1.0,0.8,0.6,0.4")
    parser.add_argument("--close-wait", type=float, default=1.0)
    parser.add_argument("--gripper-close-duration", type=float, default=1.2)

    parser.add_argument("--down-duration", type=float, default=3.0)
    parser.add_argument("--lift-duration", type=float, default=3.0)
    parser.add_argument("--hold-after-lift", type=float, default=1.0)

    parser.add_argument("--clamp-dz", type=float, default=0.012)
    parser.add_argument("--max-clamp-xy", type=float, default=0.035)

    args = parser.parse_args()

    result_dir = Path.home() / "ros2_ws/day7_baseline/results"
    result_dir.mkdir(parents=True, exist_ok=True)

    close_values = [float(x.strip()) for x in args.close_seq.split(",") if x.strip()]
    lift_zs = [float(x.strip()) for x in args.lift_seq.split(",") if x.strip()]

    log = {
        "trial_id": args.trial_id,
        "mode": "clamp_lift_probe",
        "params": vars(args),
        "close_values": close_values,
        "lift_zs": lift_zs,
        "success": False,
        "clamped": False,
        "failure_type": "unknown",
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

            truth_before = base.get_truth(args.trial_id + "_before")
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
            truth_down = base.get_truth(args.trial_id + "_after_down")
            log["truth_after_down"] = truth_down
            log["delta_after_down"] = delta(truth_before, truth_down)

            log["stage"] = "close"
            for i, cv in enumerate(close_values):
                log["stage"] = f"close_{i}_{cv}"
                node.gripper(cv, duration=args.gripper_close_duration)
                time.sleep(args.close_wait)

            time.sleep(0.5)
            truth_close = base.get_truth(args.trial_id + "_after_close")
            log["truth_after_close"] = truth_close
            log["delta_after_close"] = delta(truth_before, truth_close)

            log["lift_results"] = []

            for name in lift_names:
                log["stage"] = name

                # 再发一次最终 close 值，确保抬升前夹爪保持闭合状态
                node.gripper(close_values[-1], duration=0.4)
                time.sleep(0.2)

                node.arm(q_targets[name], duration=args.lift_duration)
                time.sleep(args.hold_after_lift)

                truth_lift = base.get_truth(args.trial_id + f"_after_{name}")
                d = delta(truth_before, truth_lift)

                log[f"truth_after_{name}"] = truth_lift
                log[f"delta_after_{name}"] = d
                log["lift_results"].append({
                    "name": name,
                    "target_z_base": waypoints[name][2],
                    "truth": truth_lift,
                    "delta": d,
                })

            truth_final = log["lift_results"][-1]["truth"] if log["lift_results"] else truth_close
            final_delta = delta(truth_before, truth_final)

            log["truth_after"] = truth_final
            log["delta_final"] = final_delta

            first_lift_delta = log["lift_results"][0]["delta"] if log["lift_results"] else final_delta

            clamped = (
                first_lift_delta["dz"] >= args.clamp_dz
                and first_lift_delta["abs_xy"] <= args.max_clamp_xy
            )

            success = (
                final_delta["dz"] >= 0.035
                or float(truth_final["z"]) > 0.825
            )

            log["clamped"] = bool(clamped)
            log["success"] = bool(success)

            if success:
                log["failure_type"] = "success"
            elif clamped:
                log["failure_type"] = "clamped_but_slipped_or_not_high_enough"
            else:
                log["failure_type"] = "not_clamped"

            log["stage"] = "done"

        except RuntimeError as e:
            msg = str(e)
            log["error"] = msg
            if "ik_fail" in msg:
                log["failure_type"] = "ik_fail"
            else:
                log["failure_type"] = "runtime_error"

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
