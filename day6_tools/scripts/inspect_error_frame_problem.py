#!/usr/bin/env python3
import csv
import numpy as np
from pathlib import Path

csv_path = Path("/home/ubuntu/ros2_ws/day6_tools/results/perception_error_dataset.csv")

def f(v):
    try:
        if v is None or v == "":
            return np.nan
        return float(v)
    except Exception:
        return np.nan

def i(v):
    try:
        if v is None or v == "":
            return 0
        return int(float(v))
    except Exception:
        return 0

rows = []
with open(csv_path, "r") as fp:
    reader = csv.DictReader(fp)
    for r in reader:
        rows.append(r)

valid_rows = []
for r in rows:
    if i(r.get("detected")) == 1 and i(r.get("depth_valid")) == 1:
        valid_rows.append(r)

print("total rows:", len(rows))
print("detected & depth_valid rows:", len(valid_rows))

if len(valid_rows) == 0:
    print("No valid rows. Cannot diagnose coordinate error.")
    raise SystemExit

print("\n=== First few valid rows ===")
for r in valid_rows[:5]:
    print("sample:", r["sample_id"])
    print("  request world:  ", f(r["x_req"]), f(r["y_req"]), f(r["z_req"]))
    print("  gt world:       ", f(r["x_gt_world"]), f(r["y_gt_world"]), f(r["z_gt_world"]))
    print("  est base:       ", f(r["x_est_base"]), f(r["y_est_base"]), f(r["z_est_base"]))
    print("  est world(old): ", f(r["x_est_world"]), f(r["y_est_world"]), f(r["z_est_world"]))
    print("  error old:      ", f(r["ex"]), f(r["ey"]), f(r["ez"]), "norm=", f(r["enorm"]))

req = np.array([[f(r["x_req"]), f(r["y_req"]), f(r["z_req"])] for r in valid_rows], dtype=float)
gt = np.array([[f(r["x_gt_world"]), f(r["y_gt_world"]), f(r["z_gt_world"])] for r in valid_rows], dtype=float)
est_base = np.array([[f(r["x_est_base"]), f(r["y_est_base"]), f(r["z_est_base"])] for r in valid_rows], dtype=float)
est_world_old = np.array([[f(r["x_est_world"]), f(r["y_est_world"]), f(r["z_est_world"])] for r in valid_rows], dtype=float)

mask = np.isfinite(req).all(axis=1) & np.isfinite(gt).all(axis=1) & np.isfinite(est_base).all(axis=1)
req = req[mask]
gt = gt[mask]
est_base = est_base[mask]
est_world_old = est_world_old[mask]

print("\nusable rows:", len(gt))

if len(gt) == 0:
    raise SystemExit

print("\n=== Check set_pose / truth consistency ===")
req_minus_gt = req - gt
print("mean(req - gt):", req_minus_gt.mean(axis=0))
print("std(req - gt): ", req_minus_gt.std(axis=0))

print("\n=== Old error statistics ===")
old_err = est_world_old - gt
print("mean old error:", old_err.mean(axis=0))
print("std old error: ", old_err.std(axis=0))
print("mean old norm: ", np.linalg.norm(old_err, axis=1).mean())

print("\n=== Candidate pure translation: gt_world - est_base ===")
offsets = gt - est_base
print("mean offset:", offsets.mean(axis=0))
print("median offset:", np.median(offsets, axis=0))
print("std offset:", offsets.std(axis=0))

# 判断是不是只差一个固定平移
translation_only_residual = est_base + np.median(offsets, axis=0) - gt
print("\ntranslation-only residual mean:", translation_only_residual.mean(axis=0))
print("translation-only residual std: ", translation_only_residual.std(axis=0))
print("translation-only residual norm mean:", np.linalg.norm(translation_only_residual, axis=1).mean())

# Kabsch rigid transform: est_base -> gt_world
if len(gt) >= 3:
    P = est_base.copy()
    Q = gt.copy()

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

    print("\n=== Rigid transform calibration est_base -> gt_world ===")
    print("R:")
    print(R)
    print("t:", t)
    print("residual mean:", residual.mean(axis=0))
    print("residual std: ", residual.std(axis=0))
    print("residual norm mean:", np.linalg.norm(residual, axis=1).mean())

    out = {
        "R": R.tolist(),
        "t": t.tolist(),
        "residual_mean": residual.mean(axis=0).tolist(),
        "residual_std": residual.std(axis=0).tolist(),
        "residual_norm_mean": float(np.linalg.norm(residual, axis=1).mean()),
        "note": "calibration maps estimate_base_frame to Gazebo world"
    }

    out_path = Path("/home/ubuntu/ros2_ws/day6_tools/results/base_to_world_calibration_from_dataset.json")
    import json
    out_path.write_text(json.dumps(out, indent=2))
    print("\nSaved:", out_path)
else:
    print("\nNot enough points for rigid transform calibration. Need at least 3 valid samples.")
