#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

def f(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default

def diff(before, after):
    if not before or not after:
        return None
    return (
        f(after.get("x")) - f(before.get("x")),
        f(after.get("y")) - f(before.get("y")),
        f(after.get("z")) - f(before.get("z")),
    )

def fmt(d):
    if d is None:
        return "   ---      ---      --- "
    dx, dy, dz = d
    return f"{dx:+8.4f} {dy:+8.4f} {dz:+8.4f}"

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

        b = d.get("truth_before")
        down = d.get("truth_after_down")
        close = d.get("truth_after_close")
        final = d.get("truth_after")

        params = d.get("params") or {}
        waypoints = d.get("waypoints_base") or {}
        down_wp = waypoints.get("down") or ["", "", ""]

        rows.append({
            "trial": d.get("trial_id", p.stem),
            "success": d.get("success"),
            "failure": d.get("failure_type", ""),
            "down_z": params.get("down_z_base", down_wp[2] if len(down_wp) >= 3 else ""),
            "xoff": params.get("x_offset", ""),
            "yoff": params.get("y_offset", ""),
            "down_diff": diff(b, down),
            "close_diff": diff(b, close),
            "final_diff": diff(b, final),
            "close_seq": d.get("close_sequence", params.get("close_seq", "")),
        })

    if not rows:
        print(f"No logs matched: {result_dir / args.pattern}")
        return

    print(
        f"{'trial':<34} {'ok':<5} {'failure':<24} "
        f"{'z':>7} {'xoff':>7} {'yoff':>7} "
        f"{'down dx dy dz':>27} {'close dx dy dz':>27} {'final dx dy dz':>27} close_seq"
    )

    for r in rows:
        print(
            f"{r['trial']:<34} "
            f"{str(r['success']):<5} "
            f"{r['failure']:<24} "
            f"{str(r['down_z']):>7} "
            f"{str(r['xoff']):>7} "
            f"{str(r['yoff']):>7} "
            f"{fmt(r['down_diff'])} "
            f"{fmt(r['close_diff'])} "
            f"{fmt(r['final_diff'])} "
            f"{r['close_seq']}"
        )

if __name__ == "__main__":
    main()
