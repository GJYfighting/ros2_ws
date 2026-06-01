#!/usr/bin/env python3
import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import rclpy
import tf2_ros

SCRIPT_DIR = Path.home() / "ros2_ws/day7_baseline/scripts"
sys.path.insert(0, str(SCRIPT_DIR))

import day7_rule_baseline_once as base


WS = Path.home() / "ros2_ws"
RESULT_DIR = WS / "day7_tools/results"
SET_BLOCK = WS / "day4_tools/scripts/set_block_pose.py"


PAD_LOCAL = {
    "l_out_link": np.array([0.0100, -0.0120, 0.0110], dtype=float),
    "r_out_link": np.array([0.0100, 0.0120, 0.0110], dtype=float),
}


def q_to_R(qx, qy, qz, qw):
    x, y, z, w = qx, qy, qz, qw
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=float)


def transform_point(tf, p_local):
    t = tf.transform.translation
    q = tf.transform.rotation
    R = q_to_R(q.x, q.y, q.z, q.w)
    return R @ p_local + np.array([t.x, t.y, t.z], dtype=float)


def run_checked(cmd, timeout=30):
    out = subprocess.run(
        cmd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
        check=True,
    ).stdout
    return out


def set_block(x, y, z, yaw):
    return run_checked([
        "python3",
        str(SET_BLOCK),
        "--x", str(x),
        "--y", str(y),
        "--z", str(z),
        "--yaw", str(yaw),
    ], timeout=20)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trial-id", default=f"pad_alignment_{int(time.time())}")
    parser.add_argument("--block-x", type=float, default=0.12)
    parser.add_argument("--block-y", type=float, default=0.0)
    parser.add_argument("--block-z", type=float, default=0.775)
    parser.add_argument("--block-yaw", type=float, default=0.0)
    parser.add_argument("--pre-z-base", type=float, default=0.220)
    parser.add_argument("--down-z-base", type=float, default=0.020)
    parser.add_argument("--x-offset", type=float, default=0.0)
    parser.add_argument("--y-offset", type=float, default=0.0)
    parser.add_argument("--close", type=float, default=None)
    args = parser.parse_args()

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    log = {
        "trial_id": args.trial_id,
        "params": vars(args),
        "t_start": time.time(),
    }

    rclpy.init()
    node = base.MotionAndIK()
    tf_buffer = tf2_ros.Buffer()
    tf_listener = tf2_ros.TransformListener(tf_buffer, node)

    try:
        node.gripper(base.GRIPPER_OPEN, duration=1.0)
        node.arm(base.HOME, duration=3.0)
        node.arm(base.OBSERVE, duration=3.0)
        node.gripper(base.GRIPPER_OPEN, duration=1.0)
        time.sleep(0.5)
        set_block(args.block_x, args.block_y, args.block_z, args.block_yaw)
        time.sleep(3.0)

        truth_before = base.get_truth(args.trial_id + "_before")
        log["truth_before"] = truth_before

        p_world = np.array([truth_before["x"], truth_before["y"], truth_before["z"]], dtype=float)
        p_base = np.array(base.world_to_base(p_world), dtype=float)
        p_base[0] += args.x_offset
        p_base[1] += args.y_offset

        waypoints = {
            "pregrasp": [p_base[0], p_base[1], args.pre_z_base],
            "down": [p_base[0], p_base[1], args.down_z_base],
        }
        log["target_base"] = p_base.tolist()
        log["waypoints_base"] = waypoints

        seed = base.OBSERVE
        q_targets = {}
        for name in ["pregrasp", "down"]:
            q, status = node.compute_ik(waypoints[name], seed, ik_link_name="end_effector_link")
            if q is None:
                q, status = node.compute_ik(waypoints[name], seed, ik_link_name="link5")
            if q is None:
                raise RuntimeError(f"ik_fail_{name}_{status}")
            q_targets[name] = q
            seed = q
        log["q_targets"] = q_targets

        node.arm(q_targets["pregrasp"], duration=2.5)
        node.arm(q_targets["down"], duration=2.5)
        time.sleep(1.0)

        if args.close is not None:
            node.gripper(args.close, duration=2.0)
            time.sleep(1.0)

        for _ in range(20):
            rclpy.spin_once(node, timeout_sec=0.1)

        frames = ["grasp_tcp", "end_effector_link", "link5", "l_out_link", "r_out_link"]
        frame_positions = {}
        for frame in frames:
            tf = tf_buffer.lookup_transform("base_link", frame, rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=2.0))
            p = tf.transform.translation
            frame_positions[frame] = [float(p.x), float(p.y), float(p.z)]
        log["frame_positions_base"] = frame_positions

        pad_centers = {}
        for frame, local in PAD_LOCAL.items():
            tf = tf_buffer.lookup_transform("base_link", frame, rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=2.0))
            pad_centers[frame] = transform_point(tf, local).tolist()
        log["pad_centers_base"] = pad_centers

        l = np.array(pad_centers["l_out_link"], dtype=float)
        r = np.array(pad_centers["r_out_link"], dtype=float)
        midpoint = 0.5 * (l + r)
        axis = r - l
        axis_norm = float(np.linalg.norm(axis))
        axis_unit = axis / axis_norm if axis_norm > 1e-9 else axis
        block_base = p_base
        err = midpoint - block_base
        lateral_err_along_clamp = float(np.dot(block_base - midpoint, axis_unit))
        perpendicular_err = (block_base - midpoint) - lateral_err_along_clamp * axis_unit

        log["pad_midpoint_base"] = midpoint.tolist()
        log["pad_axis_l_to_r_base"] = axis.tolist()
        log["pad_gap_m"] = axis_norm
        log["block_center_base"] = block_base.tolist()
        log["midpoint_error_base"] = err.tolist()
        log["block_offset_along_pad_axis_m"] = lateral_err_along_clamp
        log["block_offset_perpendicular_to_pad_axis_base"] = perpendicular_err.tolist()
        log["block_offset_perpendicular_norm_m"] = float(np.linalg.norm(perpendicular_err))
        log["truth_after_inspect"] = base.get_truth(args.trial_id + "_after_inspect")

    except Exception as e:
        log["error"] = str(e)
    finally:
        log["t_end"] = time.time()
        log["duration_s"] = log["t_end"] - log["t_start"]
        out = RESULT_DIR / f"{args.trial_id}.json"
        out.write_text(json.dumps(log, indent=2, ensure_ascii=False))
        print(json.dumps(log, indent=2, ensure_ascii=False))
        print(f"\n[DAY7] saved: {out}")
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
