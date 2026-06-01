#!/usr/bin/env python3
import argparse
import csv
import json
import math
import random
import shutil
import subprocess
import time
from pathlib import Path

import yaml
import numpy as np


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def run_cmd(cmd, timeout=None):
    print("[RUN]", " ".join(cmd))
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout
    )

    if proc.stdout.strip():
        print(proc.stdout)

    if proc.returncode != 0:
        print(proc.stderr)
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")

    return proc


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def quat_to_yaw(qx, qy, qz, qw):
    # yaw around z axis
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


def angle_diff(a, b):
    d = a - b
    while d > math.pi:
        d -= 2.0 * math.pi
    while d < -math.pi:
        d += 2.0 * math.pi
    return d


def get_nested(d, keys, default=None):
    cur = d
    for k in keys:
        if cur is None:
            return default
        if not isinstance(cur, dict):
            return default
        if k not in cur:
            return default
        cur = cur[k]
    return cur


def is_not_none(v):
    return v is not None


def safe_float(v):
    if v is None:
        return ""
    try:
        return float(v)
    except Exception:
        return ""


def compute_edge_flag(bbox, width=640, height=400, margin=5):
    if bbox is None:
        return 0

    if len(bbox) == 4:
        # bbox_xywh
        x, y, w, h = bbox
        x2 = x + w
        y2 = y + h
    else:
        return 0

    if x <= margin or y <= margin or x2 >= width - margin or y2 >= height - margin:
        return 1
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="/home/ubuntu/ros2_ws/day6_tools/config/perception_error_config.yaml")
    parser.add_argument("--n", type=int, default=30)
    parser.add_argument("--seed", type=int, default=6)
    parser.add_argument("--start-index", type=int, default=0)
    args = parser.parse_args()

    cfg = load_yaml(args.config)

    rng = random.Random(args.seed)

    results_dir = Path(cfg["paths"]["results_dir"])
    samples_dir = results_dir / "samples"
    results_dir.mkdir(parents=True, exist_ok=True)
    samples_dir.mkdir(parents=True, exist_ok=True)

    csv_path = Path(cfg["paths"]["dataset_csv"])
    new_file = not csv_path.exists()

    fieldnames = [
        "sample_id",

        "x_req", "y_req", "z_req", "yaw_req",

        "detected", "depth_valid", "tf_found",
        "note",

        "bbox_x", "bbox_y", "bbox_w", "bbox_h",
        "edge_flag",
        "area_px",
        "angle_deg",
        "depth_median_m",

        "x_est_base", "y_est_base", "z_est_base",
        "x_est_world", "y_est_world", "z_est_world",

        "x_gt_world", "y_gt_world", "z_gt_world",
        "qx_gt", "qy_gt", "qz_gt", "qw_gt", "yaw_gt",

        "ex", "ey", "ez", "enorm",
        "yaw_err_req_gt",

        "perception_json",
        "truth_json"
    ]

    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if new_file:
            writer.writeheader()

        for i in range(args.start_index, args.start_index + args.n):
            sample_id = f"s{i:04d}"
            print("\n" + "=" * 70)
            print(f"[SAMPLE] {sample_id}")
            print("=" * 70)

            x = rng.uniform(cfg["randomization"]["x_min"], cfg["randomization"]["x_max"])
            y = rng.uniform(cfg["randomization"]["y_min"], cfg["randomization"]["y_max"])
            yaw = rng.uniform(cfg["randomization"]["yaw_min"], cfg["randomization"]["yaw_max"])
            z = float(cfg["block"]["z"])

            # 1. 随机放置木块
            run_cmd([
                "python3",
                cfg["paths"]["set_block_pose_script"],
                "--x", str(x),
                "--y", str(y),
                "--z", str(z),
                "--yaw", str(yaw)
            ])

            time.sleep(float(cfg["timing"]["settle_after_block_reset_sec"]))

            # 2. 发送观察位
            run_cmd([
                "python3",
                cfg["paths"]["send_pose_script"],
                "--yaml", cfg["paths"]["observe_pose_file"],
                "--key", "observe_pose"
            ])

            time.sleep(float(cfg["timing"]["settle_after_observe_pose_sec"]))

            # 3. 运行第5天感知脚本
            try:
                run_cmd([
                    "python3",
                    cfg["paths"]["perception_script"]
                ])
                perception = load_json(cfg["paths"]["perception_json"])
            except Exception as e:
                print(f"[WARN] perception script failed: {e}")
                perception = {
                    "detected": False,
                    "depth_valid": False,
                    "note": f"perception_script_failed: {e}"
                }

            # 4. 读取 Gazebo 真值
            truth_json_path = str(samples_dir / f"{sample_id}_truth.json")
            try:
                run_cmd([
                    "python3",
                    cfg["paths"]["get_block_truth_script"],
                    "--out", truth_json_path
                ])
                truth = load_json(truth_json_path)
            except Exception as e:
                print(f"[WARN] truth read failed: {e}")
                truth = {
                    "x": None, "y": None, "z": None,
                    "qx": None, "qy": None, "qz": None, "qw": None
                }

            # 5. 归档 perception json
            perception_archive_path = str(samples_dir / f"{sample_id}_perception.json")

            try:
                if Path(cfg["paths"]["perception_json"]).exists():
                    shutil.copyfile(cfg["paths"]["perception_json"], perception_archive_path)
            except Exception:
                pass

            detected = bool(perception.get("detected", False))
            depth_valid = bool(perception.get("depth_valid", False))

            tf_found = bool(get_nested(perception, ["tf", "transform_found"], False))
            if "base_transform_found" in get_nested(perception, ["tf"], {}):
                tf_found = bool(get_nested(perception, ["tf", "base_transform_found"], False))

            note = perception.get("reason", perception.get("note", ""))

            color_det = perception.get("color_detection", {})
            bbox = color_det.get("bbox_xywh", None)

            bbox_x = bbox_y = bbox_w = bbox_h = ""
            if bbox is not None and len(bbox) == 4:
                bbox_x, bbox_y, bbox_w, bbox_h = bbox

            area_px = color_det.get("area_px", "")
            angle_deg = color_det.get("angle_deg", "")

            depth_median = get_nested(perception, ["depth", "median_m"], "")

            est_base = perception.get("estimate_base_frame", None)

            x_est_base = y_est_base = z_est_base = None
            x_est_world = y_est_world = z_est_world = None

            if isinstance(est_base, dict):
                x_est_base = est_base.get("x")
                y_est_base = est_base.get("y")
                z_est_base = est_base.get("z")

                if is_not_none(x_est_base) and is_not_none(y_est_base) and is_not_none(z_est_base):
                    spawn = cfg["robot_spawn_in_gazebo_world"]
                    x_est_world = float(x_est_base) + float(spawn["x"])
                    y_est_world = float(y_est_base) + float(spawn["y"])
                    z_est_world = float(z_est_base) + float(spawn["z"])

            x_gt = truth.get("x")
            y_gt = truth.get("y")
            z_gt = truth.get("z")
            qx_gt = truth.get("qx")
            qy_gt = truth.get("qy")
            qz_gt = truth.get("qz")
            qw_gt = truth.get("qw")

            yaw_gt = None
            if all(v is not None for v in [qx_gt, qy_gt, qz_gt, qw_gt]):
                yaw_gt = quat_to_yaw(float(qx_gt), float(qy_gt), float(qz_gt), float(qw_gt))

            ex = ey = ez = enorm = None

            if all(v is not None for v in [x_est_world, y_est_world, z_est_world, x_gt, y_gt, z_gt]):
                ex = float(x_est_world) - float(x_gt)
                ey = float(y_est_world) - float(y_gt)
                ez = float(z_est_world) - float(z_gt)
                enorm = float(np.linalg.norm([ex, ey, ez]))

            yaw_err = None
            if yaw_gt is not None:
                yaw_err = angle_diff(float(yaw), yaw_gt)

            row = {
                "sample_id": sample_id,

                "x_req": x,
                "y_req": y,
                "z_req": z,
                "yaw_req": yaw,

                "detected": int(detected),
                "depth_valid": int(depth_valid),
                "tf_found": int(tf_found),
                "note": note,

                "bbox_x": bbox_x,
                "bbox_y": bbox_y,
                "bbox_w": bbox_w,
                "bbox_h": bbox_h,
                "edge_flag": compute_edge_flag(bbox),
                "area_px": area_px,
                "angle_deg": angle_deg,
                "depth_median_m": depth_median,

                "x_est_base": safe_float(x_est_base),
                "y_est_base": safe_float(y_est_base),
                "z_est_base": safe_float(z_est_base),
                "x_est_world": safe_float(x_est_world),
                "y_est_world": safe_float(y_est_world),
                "z_est_world": safe_float(z_est_world),

                "x_gt_world": safe_float(x_gt),
                "y_gt_world": safe_float(y_gt),
                "z_gt_world": safe_float(z_gt),
                "qx_gt": safe_float(qx_gt),
                "qy_gt": safe_float(qy_gt),
                "qz_gt": safe_float(qz_gt),
                "qw_gt": safe_float(qw_gt),
                "yaw_gt": safe_float(yaw_gt),

                "ex": safe_float(ex),
                "ey": safe_float(ey),
                "ez": safe_float(ez),
                "enorm": safe_float(enorm),
                "yaw_err_req_gt": safe_float(yaw_err),

                "perception_json": perception_archive_path,
                "truth_json": truth_json_path
            }

            writer.writerow(row)
            f.flush()

            print("[ROW]", json.dumps(row, indent=2))

    print("\n[DONE]")
    print(f"Dataset saved to: {csv_path}")


if __name__ == "__main__":
    main()
