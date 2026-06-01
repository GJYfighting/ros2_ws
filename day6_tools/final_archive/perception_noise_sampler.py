#!/usr/bin/env python3
import argparse
import json
import numpy as np


class PerceptionNoiseSampler:
    def __init__(self, model_path, seed=0, mode="empirical", scale=1.0):
        with open(model_path, "r") as f:
            self.model = json.load(f)

        self.rng = np.random.default_rng(seed)
        self.mode = mode
        self.scale = float(scale)

        self.miss_rate = float(self.model.get("miss_rate", 0.0))
        self.depth_fail_rate = float(self.model.get("depth_fail_rate_given_detected", 0.0))

        self.mean = np.asarray(
            self.model.get("error_mean_xyz", [0.0, 0.0, 0.0]),
            dtype=np.float64
        )

        self.cov = np.asarray(
            self.model.get("error_cov_xyz", np.eye(3) * 1e-4),
            dtype=np.float64
        )

        samples = self.model.get("error_samples_xyz", [])
        self.samples = (
            np.asarray(samples, dtype=np.float64)
            if len(samples) > 0
            else np.zeros((0, 3), dtype=np.float64)
        )

    def sample_error(self):
        if self.mode == "empirical" and self.samples.shape[0] > 0:
            idx = self.rng.integers(0, self.samples.shape[0])
            err = self.samples[idx].copy()
        else:
            err = self.rng.multivariate_normal(self.mean, self.cov)

        # 保留均值偏差，并放大随机部分
        centered = err - self.mean
        return self.mean + self.scale * centered

    def sample_measurement(self, true_xyz):
        true_xyz = np.asarray(true_xyz, dtype=np.float64)

        if self.rng.random() < self.miss_rate:
            return {
                "visible": False,
                "depth_valid": False,
                "estimate_xyz": None,
                "error_xyz": None,
                "reason": "missed_detection"
            }

        if self.rng.random() < self.depth_fail_rate:
            return {
                "visible": True,
                "depth_valid": False,
                "estimate_xyz": None,
                "error_xyz": None,
                "reason": "depth_failure"
            }

        err = self.sample_error()
        est = true_xyz + err

        return {
            "visible": True,
            "depth_valid": True,
            "estimate_xyz": est.tolist(),
            "error_xyz": err.tolist(),
            "reason": "ok"
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="/home/ubuntu/ros2_ws/day6_tools/results/perception_error_model.json"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mode", choices=["empirical", "gaussian"], default="empirical")
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--x", type=float, default=0.12)
    parser.add_argument("--y", type=float, default=0.0)
    parser.add_argument("--z", type=float, default=0.775)
    parser.add_argument("--n", type=int, default=5)

    args = parser.parse_args()

    sampler = PerceptionNoiseSampler(
        model_path=args.model,
        seed=args.seed,
        mode=args.mode,
        scale=args.scale
    )

    true_xyz = [args.x, args.y, args.z]

    for i in range(args.n):
        sample = sampler.sample_measurement(true_xyz)
        print(json.dumps(sample, indent=2))


if __name__ == "__main__":
    main()
