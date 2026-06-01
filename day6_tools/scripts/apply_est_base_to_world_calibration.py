#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path

import numpy as np


def to_float(v):
    try:
        if v is None or v == "":
            return np.nan
        return float(v)
    except Exception:
        return np.nan


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv-in",
        default="/home/ubuntu/ros2_ws/day6_tools/results/perception_error_eval_raw.csv"
    )
    parser.add_argument(
        "--calib",
        default="/home/ubuntu/ros2_ws/day6_tools/results/est_base_to_world_calibration.json"
    )
    parser.add_argument(
        "--csv-out",
        default="/home/ubuntu/ros2_ws/day6_tools/results/perception_error_dataset.csv"
    )
    args = parser.parse_args()

    calib = json.loads(Path(args.calib).read_text())
    R = np.asarray(calib["R"], dtype=np.float64)
    t = np.asarray(calib["t"], dtype=np.float64)

    rows = []
    with open(args.csv_in, "r") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for r in reader:
            rows.append(r)

    if fieldnames is None:
        raise RuntimeError("Input CSV has no header")

    # 保存旧字段，方便对比
    extra_fields = [
        "x_est_world_old",
        "y_est_world_old",
        "z_est_world_old",
        "ex_old",
        "ey_old",
        "ez_old",
        "enorm_old",
        "calibrated_error_used"
    ]

    out_fields = list(fieldnames)
    for f in extra_fields:
        if f not in out_fields:
            out_fields.append(f)

    for r in rows:
        xb = to_float(r.get("x_est_base"))
        yb = to_float(r.get("y_est_base"))
        zb = to_float(r.get("z_est_base"))

        xg = to_float(r.get("x_gt_world"))
        yg = to_float(r.get("y_gt_world"))
        zg = to_float(r.get("z_gt_world"))

        # 备份旧值
        r["x_est_world_old"] = r.get("x_est_world", "")
        r["y_est_world_old"] = r.get("y_est_world", "")
        r["z_est_world_old"] = r.get("z_est_world", "")
        r["ex_old"] = r.get("ex", "")
        r["ey_old"] = r.get("ey", "")
        r["ez_old"] = r.get("ez", "")
        r["enorm_old"] = r.get("enorm", "")

        if np.isfinite([xb, yb, zb, xg, yg, zg]).all():
            p_base = np.asarray([xb, yb, zb], dtype=np.float64)
            p_world = R @ p_base + t

            err = p_world - np.asarray([xg, yg, zg], dtype=np.float64)
            enorm = np.linalg.norm(err)

            r["x_est_world"] = f"{p_world[0]:.9f}"
            r["y_est_world"] = f"{p_world[1]:.9f}"
            r["z_est_world"] = f"{p_world[2]:.9f}"

            r["ex"] = f"{err[0]:.9f}"
            r["ey"] = f"{err[1]:.9f}"
            r["ez"] = f"{err[2]:.9f}"
            r["enorm"] = f"{enorm:.9f}"

            r["calibrated_error_used"] = "1"
        else:
            r["calibrated_error_used"] = "0"

    out_path = Path(args.csv_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=out_fields)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"Saved calibrated dataset to: {out_path}")


if __name__ == "__main__":
    main()
