#!/usr/bin/env python3
import argparse
import csv
import json
import random
import time
from pathlib import Path
from ign_world_utils import (
    load_cfg,
    service_exists,
    set_pose,
    remove_model,
    create_model_from_sdf,
    write_temp_block_sdf,
    get_truth_pose,
)

def sample_uniform(rng, lo, hi):
    return rng.uniform(lo, hi)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(Path.home() / "ros2_ws/day4_tools/config/world_config.yaml"))
    parser.add_argument("--mode", choices=["reset", "randomize"], default="randomize")
    parser.add_argument("--n", type=int, default=1, help="number of samples")
    parser.add_argument("--pose-only", action="store_true", help="only use set_pose, do not respawn model")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    world = cfg["world"]["name"]
    model = cfg["world"]["model_name"]
    nominal = cfg["block_nominal"]
    rand_cfg = cfg["randomization"]

    create_srv = cfg["world"]["create_service"]
    remove_srv = cfg["world"]["remove_service"]

    can_full_respawn = service_exists(create_srv) and service_exists(remove_srv)
    if not can_full_respawn and not args.pose_only:
        print("[WARN] create/remove service not both available, fallback to pose-only.")
        args.pose_only = True

    rng = random.Random(args.seed)

    log_path = Path.home() / "ros2_ws/day4_tools/results/randomization_log.csv"
    new_file = not log_path.exists()
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with open(log_path, "a", newline="") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow([
                "mode", "pose_only", "x_req", "y_req", "z_req", "yaw_req",
                "mass_req", "mu_req",
                "x_gt", "y_gt", "z_gt", "qx_gt", "qy_gt", "qz_gt", "qw_gt",
                "source_topic"
            ])

        for idx in range(args.n):
            if args.mode == "reset":
                x = nominal["x"]
                y = nominal["y"]
                z = nominal["z"]
                yaw = 0.0
                mass = nominal["mass"]
                mu = nominal["mu"]
            else:
                x = sample_uniform(rng, rand_cfg["x_min"], rand_cfg["x_max"])
                y = sample_uniform(rng, rand_cfg["y_min"], rand_cfg["y_max"])
                z = nominal["z"]
                yaw = sample_uniform(rng, rand_cfg["yaw_min"], rand_cfg["yaw_max"])
                mass = sample_uniform(rng, rand_cfg["mass_min"], rand_cfg["mass_max"])
                mu = sample_uniform(rng, rand_cfg["mu_min"], rand_cfg["mu_max"])

            if args.pose_only:
                out = set_pose(world, model, x, y, z, yaw)
                print(f"[{idx}] pose-only set_pose result: {out}")
            else:
                tmp_sdf = write_temp_block_sdf(
                    tmp_dir=str(Path.home() / "ros2_ws/day4_tools/tmp_models"),
                    model_name=model,
                    size=nominal["size"],
                    mass=mass,
                    mu=mu,
                    rgba=nominal["color_rgba"],
                )
                try:
                    rm_out = remove_model(world, model)
                    print(f"[{idx}] remove result: {rm_out}")
                except Exception as e:
                    print(f"[{idx}] remove warning: {e}")

                time.sleep(0.4)
                cr_out = create_model_from_sdf(world, model, tmp_sdf, x, y, z, yaw)
                print(f"[{idx}] create result: {cr_out}")

            time.sleep(0.8)
            gt = get_truth_pose(world, model)

            writer.writerow([
                args.mode, args.pose_only, x, y, z, yaw,
                mass, mu,
                gt["x"], gt["y"], gt["z"], gt["qx"], gt["qy"], gt["qz"], gt["qw"],
                gt["source_topic"]
            ])
            f.flush()

            print(json.dumps({
                "req": {"x": x, "y": y, "z": z, "yaw": yaw, "mass": mass, "mu": mu},
                "gt": gt
            }, indent=2))

    print(f"\nSaved log to: {log_path}")

if __name__ == "__main__":
    main()
