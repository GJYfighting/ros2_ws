#!/usr/bin/env python3
import argparse
import json
import subprocess
import time
from collections import Counter
from pathlib import Path

WS = Path.home() / "ros2_ws"
RESULT_DIR = WS / "day7_baseline" / "results"
ONCE = WS / "day7_baseline" / "scripts" / "day7_rule_baseline_once.py"
SET_BLOCK = WS / "day4_tools" / "scripts" / "set_block_pose.py"
RANDOMIZE = WS / "day4_tools" / "scripts" / "randomize_block.py"


def run(cmd, timeout=120):
    print("\n[RUN]", " ".join(map(str, cmd)), flush=True)
    p = subprocess.run(
        cmd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
    )
    print(p.stdout)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(map(str, cmd))}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["fixed", "random"], default="fixed")
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--seed", type=int, default=700)
    parser.add_argument("--use-truth", action="store_true")
    parser.add_argument("--x", type=float, default=0.12)
    parser.add_argument("--y", type=float, default=0.0)
    parser.add_argument("--z", type=float, default=0.775)
    parser.add_argument("--skip-home", action="store_true")
    args = parser.parse_args()

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    records = []

    for i in range(args.n):
        trial_id = f"{args.mode}_{i:03d}_{int(time.time())}"

        if args.mode == "fixed":
            run(
                [
                    "python3",
                    str(SET_BLOCK),
                    "--x",
                    str(args.x),
                    "--y",
                    str(args.y),
                    "--z",
                    str(args.z),
                    "--yaw",
                    "0.0",
                ],
                timeout=30,
            )
        else:
            run(
                [
                    "python3",
                    str(RANDOMIZE),
                    "--mode",
                    "randomize",
                    "--n",
                    "1",
                    "--pose-only",
                    "--seed",
                    str(args.seed + i),
                ],
                timeout=40,
            )

        time.sleep(1.0)

        cmd = ["python3", str(ONCE), "--trial-id", trial_id]
        if args.use_truth:
            cmd.append("--use-truth")
        if args.skip_home:
            cmd.append("--skip-home")

        run(cmd, timeout=180)

        log_path = RESULT_DIR / f"{trial_id}.json"
        if log_path.exists():
            records.append(json.loads(log_path.read_text()))

    n = len(records)
    ok = sum(1 for r in records if r.get("success"))
    failure_counts = Counter(r.get("failure_type", "missing_log") for r in records)

    summary = {
        "mode": args.mode,
        "n_requested": args.n,
        "n_logged": n,
        "success_count": ok,
        "success_rate": ok / n if n else 0.0,
        "failure_counts": dict(failure_counts),
        "logs": [str(RESULT_DIR / f"{r['trial_id']}.json") for r in records],
    }

    out = RESULT_DIR / f"day7_summary_{args.mode}_{int(time.time())}.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    print("\n[DAY7 SUMMARY]")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\n[DAY7] saved summary: {out}")


if __name__ == "__main__":
    main()
