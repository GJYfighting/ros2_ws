#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

def safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pattern", nargs="?", default="step7_*.json")
    args = parser.parse_args()

    result_dir = Path.home() / "ros2_ws/day7_baseline/results"
    files = sorted(result_dir.glob(args.pattern))

    rows = []
    for p in files:
        try:
            d = json.loads(p.read_text())
        except Exception:
            continue

        if "failure_type" not in d:
            continue

        b = d.get("truth_before") or {}
        a = d.get("truth_after") or {}

        bx = safe_float(b.get("x"))
        by = safe_float(b.get("y"))
        bz = safe_float(b.get("z"))
        ax = safe_float(a.get("x"))
        ay = safe_float(a.get("y"))
        az = safe_float(a.get("z"))

        waypoints = d.get("waypoints_base") or {}
        down = waypoints.get("down") or ["", "", ""]

        rows.append({
            "trial": d.get("trial_id", p.stem),
            "success": d.get("success"),
            "failure": d.get("failure_type"),
            "dz": az - bz,
            "dx": ax - bx,
            "dy": ay - by,
            "down_z": down[2] if len(down) >= 3 else "",
            "close_sequence": d.get("close_sequence", ""),
            "path": str(p),
        })

    if not rows:
        print(f"No logs matched: {result_dir / args.pattern}")
        return

    print(
        f"{'trial':<30} {'ok':<5} {'failure':<16} "
        f"{'dz':>8} {'dx':>8} {'dy':>8} {'down_z':>8} {'close_sequence'}"
    )

    for r in rows:
        print(
            f"{r['trial']:<30} "
            f"{str(r['success']):<5} "
            f"{r['failure']:<16} "
            f"{r['dz']:>+8.4f} "
            f"{r['dx']:>+8.4f} "
            f"{r['dy']:>+8.4f} "
            f"{str(r['down_z']):>8} "
            f"{r['close_sequence']}"
        )

if __name__ == "__main__":
    main()
