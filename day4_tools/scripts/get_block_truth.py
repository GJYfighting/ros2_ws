#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from ign_world_utils import load_cfg, get_truth_pose

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(Path.home() / "ros2_ws/day4_tools/config/world_config.yaml"))
    parser.add_argument("--out", default=str(Path.home() / "ros2_ws/day4_tools/results/block_truth_latest.json"))
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    world = cfg["world"]["name"]
    model = cfg["world"]["model_name"]

    pose = get_truth_pose(world, model)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(pose, indent=2))

    print(json.dumps(pose, indent=2))
    print(f"\nSaved to: {out_path}")

if __name__ == "__main__":
    main()
