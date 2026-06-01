#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def to_float(v):
    if v is None or v == "":
        return np.nan
    try:
        return float(v)
    except Exception:
        return np.nan


def to_int(v):
    if v is None or v == "":
        return 0
    try:
        return int(float(v))
    except Exception:
        return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="/home/ubuntu/ros2_ws/day6_tools/results/perception_error_dataset.csv")
    parser.add_argument("--model", default="/home/ubuntu/ros2_ws/day6_tools/results/perception_error_model.json")
    parser.add_argument("--summary", default="/home/ubuntu/ros2_ws/day6_tools/results/perception_error_summary.txt")
    parser.add_argument("--plots", default="/home/ubuntu/ros2_ws/day6_tools/results/plots")
    args = parser.parse_args()

    rows = []
    with open(args.csv, "r") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    n_total = len(rows)
    if n_total == 0:
        raise RuntimeError("Dataset is empty")

    detected = np.array([to_int(r["detected"]) for r in rows], dtype=np.int32)
    depth_valid = np.array([to_int(r["depth_valid"]) for r in rows], dtype=np.int32)
    tf_found = np.array([to_int(r["tf_found"]) for r in rows], dtype=np.int32)

    valid_mask = (detected == 1) & (depth_valid == 1)

    ex = np.array([to_float(r["ex"]) for r in rows], dtype=np.float64)
    ey = np.array([to_float(r["ey"]) for r in rows], dtype=np.float64)
    ez = np.array([to_float(r["ez"]) for r in rows], dtype=np.float64)
    enorm = np.array([to_float(r["enorm"]) for r in rows], dtype=np.float64)

    area = np.array([to_float(r["area_px"]) for r in rows], dtype=np.float64)
    edge = np.array([to_int(r["edge_flag"]) for r in rows], dtype=np.int32)

    error_valid = valid_mask & np.isfinite(ex) & np.isfinite(ey) & np.isfinite(ez)

    errors = np.stack([ex[error_valid], ey[error_valid], ez[error_valid]], axis=1) if np.count_nonzero(error_valid) > 0 else np.zeros((0, 3))

    n_detected = int(np.count_nonzero(detected == 1))
    n_depth_valid = int(np.count_nonzero(valid_mask))
    n_error_valid = int(errors.shape[0])

    miss_rate = 1.0 - n_detected / max(n_total, 1)
    depth_fail_rate_given_detected = 1.0 - n_depth_valid / max(n_detected, 1)

    model = {
        "n_total": n_total,
        "n_detected": n_detected,
        "n_depth_valid": n_depth_valid,
        "n_error_valid": n_error_valid,
        "miss_rate": miss_rate,
        "depth_fail_rate_given_detected": depth_fail_rate_given_detected,
    }

    lines = []
    lines.append(f"n_total: {n_total}")
    lines.append(f"n_detected: {n_detected}")
    lines.append(f"n_depth_valid: {n_depth_valid}")
    lines.append(f"n_error_valid: {n_error_valid}")
    lines.append(f"miss_rate: {miss_rate:.6f}")
    lines.append(f"depth_fail_rate_given_detected: {depth_fail_rate_given_detected:.6f}")

    if n_error_valid > 0:
        mean_xyz = errors.mean(axis=0)
        std_xyz = errors.std(axis=0)
        cov_xyz = np.cov(errors.T).tolist() if n_error_valid > 1 else np.diag((std_xyz + 1e-6) ** 2).tolist()

        norm_valid = np.linalg.norm(errors, axis=1)

        model.update({
            "error_mean_xyz": mean_xyz.tolist(),
            "error_std_xyz": std_xyz.tolist(),
            "error_cov_xyz": cov_xyz,
            "error_norm_mean": float(norm_valid.mean()),
            "error_norm_std": float(norm_valid.std()),
            "error_samples_xyz": errors.tolist(),
            "recommended_sampler": "empirical_or_gaussian",
        })

        lines.append(f"error_mean_x: {mean_xyz[0]:.6f}")
        lines.append(f"error_mean_y: {mean_xyz[1]:.6f}")
        lines.append(f"error_mean_z: {mean_xyz[2]:.6f}")
        lines.append(f"error_std_x: {std_xyz[0]:.6f}")
        lines.append(f"error_std_y: {std_xyz[1]:.6f}")
        lines.append(f"error_std_z: {std_xyz[2]:.6f}")
        lines.append(f"error_norm_mean: {norm_valid.mean():.6f}")
        lines.append(f"error_norm_std: {norm_valid.std():.6f}")

    else:
        model.update({
            "error_mean_xyz": [0.0, 0.0, 0.0],
            "error_std_xyz": [0.02, 0.02, 0.02],
            "error_cov_xyz": [[0.0004, 0, 0], [0, 0.0004, 0], [0, 0, 0.0004]],
            "error_samples_xyz": [],
            "recommended_sampler": "default_due_to_no_valid_samples",
        })
        lines.append("No valid 3D error samples. Fix perception before training.")

    out_model = Path(args.model)
    out_model.parent.mkdir(parents=True, exist_ok=True)
    out_model.write_text(json.dumps(model, indent=2))

    out_summary = Path(args.summary)
    out_summary.write_text("\n".join(lines) + "\n")

    print("\n".join(lines))
    print(f"\nSaved model: {out_model}")
    print(f"Saved summary: {out_summary}")

    plots_dir = Path(args.plots)
    plots_dir.mkdir(parents=True, exist_ok=True)

    if n_error_valid > 0:
        # 1. ex hist
        plt.figure()
        plt.hist(errors[:, 0], bins=20)
        plt.xlabel("x error (m)")
        plt.ylabel("count")
        plt.title("Perception x error")
        plt.savefig(plots_dir / "error_x_hist.png", dpi=150)
        plt.close()

        # 2. ey hist
        plt.figure()
        plt.hist(errors[:, 1], bins=20)
        plt.xlabel("y error (m)")
        plt.ylabel("count")
        plt.title("Perception y error")
        plt.savefig(plots_dir / "error_y_hist.png", dpi=150)
        plt.close()

        # 3. ez hist
        plt.figure()
        plt.hist(errors[:, 2], bins=20)
        plt.xlabel("z error (m)")
        plt.ylabel("count")
        plt.title("Perception z error")
        plt.savefig(plots_dir / "error_z_hist.png", dpi=150)
        plt.close()

        # 4. xy scatter
        plt.figure()
        plt.scatter(errors[:, 0], errors[:, 1])
        plt.xlabel("x error (m)")
        plt.ylabel("y error (m)")
        plt.title("XY error scatter")
        plt.savefig(plots_dir / "error_xy_scatter.png", dpi=150)
        plt.close()

        # 5. norm hist
        norm_valid = np.linalg.norm(errors, axis=1)
        plt.figure()
        plt.hist(norm_valid, bins=20)
        plt.xlabel("error norm (m)")
        plt.ylabel("count")
        plt.title("Perception 3D error norm")
        plt.savefig(plots_dir / "error_norm_hist.png", dpi=150)
        plt.close()

        # 6. error norm vs area
        area_valid = area[error_valid]
        plt.figure()
        plt.scatter(area_valid, norm_valid)
        plt.xlabel("red area (px)")
        plt.ylabel("error norm (m)")
        plt.title("Error norm vs detected area")
        plt.savefig(plots_dir / "error_vs_area.png", dpi=150)
        plt.close()

    print(f"Saved plots to: {plots_dir}")


if __name__ == "__main__":
    main()
