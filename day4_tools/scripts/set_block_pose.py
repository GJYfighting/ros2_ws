#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from ign_world_utils import load_cfg, set_pose

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(Path.home() / "ros2_ws/day4_tools/config/world_config.yaml"))
    parser.add_argument("--x", type=float, required=True)
    parser.add_argument("--y", type=float, required=True)
    parser.add_argument("--z", type=float, default=None)
    parser.add_argument("--yaw", type=float, default=0.0)
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    world = cfg["world"]["name"]
    model = cfg["world"]["model_name"]
    z = args.z if args.z is not None else cfg["block_nominal"]["z"]

    out = set_pose(world, model, args.x, args.y, z, args.yaw)

    result = {
        "world": world,
        "model": model,
        "x": args.x,
        "y": args.y,
        "z": z,
        "yaw": args.yaw,
        "service_result": out
    }
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
