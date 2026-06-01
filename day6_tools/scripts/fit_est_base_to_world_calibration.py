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


def to_int(v):
    try:
        if v is None or v == "":
            return 0
        return int(float(v))
    except Exception:
        return 0


def fit_rigid_transform(P, Q):
    """
    Fit R, t such that:
        Q ≈ R @ P + t

    P: Nx3, estimate_base
    Q: Nx3, ground-truth world
    """
    Pc = P.mean(axis=0)
    Qc = Q.mean(axis=0)

    P0 = P - Pc
    Q0 = Q - Qc

    H = P0.T @ Q0
    U, S, Vt = np.linalg.svd(H)

    R = Vt.T @ U.T

    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    t = Qc - R @ Pc

    pred = (R @ P.T).T + t
    residual = pred - Q
    residual_norm = np.linalg.norm(residual, axis=1)

    return R, t, residual, residual_norm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        default="/home/ubuntu/ros2_ws/day6_tools/results/perception_error_calibration_raw.csv"
    )
    parser.add_argument(
        "--out",
        default="/home/ubuntu/ros2_ws/day6_tools/results/est_base_to_world_calibration.json"
    )
    args = parser.parse_args()

    rows = []
    with open(args.csv, "r") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    P = []
    Q = []
    used_samples = []

    for r in rows:
        detected = to_int(r.get("detected"))
        depth_valid = to_int(r.get("depth_valid"))

        xb = to_float(r.get("x_est_base"))
        yb = to_float(r.get("y_est_base"))
        zb = to_float(r.get("z_est_base"))

        xg = to_float(r.get("x_gt_world"))
        yg = to_float(r.get("y_gt_world"))
        zg = to_float(r.get("z_gt_world"))

        if detected != 1 or depth_valid != 1:
            continue

        if not np.isfinite([xb, yb, zb, xg, yg, zg]).all():
            continue

        P.append([xb, yb, zb])
        Q.append([xg, yg, zg])
        used_samples.append(r.get("sample_id", ""))

    P = np.asarray(P, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)

    if P.shape[0] < 6:
        raise RuntimeError(
            f"Only {P.shape[0]} valid calibration samples. "
            "Use at least 6, preferably 10-15."
        )

    R, t, residual, residual_norm = fit_rigid_transform(P, Q)

    result = {
        "description": "Rigid calibration from estimate_base_frame to Gazebo world frame",
        "formula": "p_world = R @ p_est_base + t",
        "n_used": int(P.shape[0]),
        "used_samples": used_samples,
        "R": R.tolist(),
        "t": t.tolist(),
        "residual_mean_xyz": residual.mean(axis=0).tolist(),
        "residual_std_xyz": residual.std(axis=0).tolist(),
        "residual_norm_mean": float(residual_norm.mean()),
        "residual_norm_std": float(residual_norm.std()),
        "residual_norm_min": float(residual_norm.min()),
        "residual_norm_max": float(residual_norm.max())
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))

    print(json.dumps(result, indent=2))
    print(f"\nSaved calibration to: {out_path}")


if __name__ == "__main__":
    main()
