#!/usr/bin/env python3
import argparse
import csv
import json
import subprocess
import time
from pathlib import Path

import numpy as np

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node

from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from geometry_msgs.msg import PoseStamped
from moveit_msgs.msg import MoveItErrorCodes
from moveit_msgs.srv import GetPositionIK
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


ARM_JOINTS = ["joint1", "joint2", "joint3", "joint4", "joint5"]
GRIPPER_JOINTS = [
    "r_joint",
    "l_joint",
    "l_in_joint",
    "l_out_joint",
    "r_in_joint",
    "r_out_joint",
]

HOME = [0.0, 0.0, 0.0, 0.0, 0.0]
OBSERVE = [0.0, 0.7309, -1.6127, -1.6591, 0.0]
GRIPPER_OPEN = 0.8
GRIPPER_CLOSE = 0.0

ROBOT_BASE_WORLD = np.array([-0.20, 0.00, 0.75], dtype=float)

def gripper_positions_from_r(q):
    q = float(q)
    return [q, -q, -q, q, -q, q]

WS = Path.home() / "ros2_ws"
RESULT_DIR = WS / "day7_baseline" / "results"
PERCEPTION_SCRIPT = WS / "day5_method_a" / "scripts" / "method_a_color_rgbd_once.py"
PERCEPTION_JSON = WS / "day5_method_a" / "results" / "method_a_color_once.json"
CALIB_JSON = WS / "day6_tools" / "results" / "est_base_to_world_calibration.json"
TRUTH_SCRIPT = WS / "day4_tools" / "scripts" / "get_block_truth.py"


def duration_msg(seconds: float) -> Duration:
    sec = int(seconds)
    nanosec = int(round((float(seconds) - sec) * 1e9))
    return Duration(sec=sec, nanosec=nanosec)


class MotionAndIK(Node):
    def __init__(self):
        super().__init__("day7_rule_baseline_once")
        self.arm_client = ActionClient(
            self,
            FollowJointTrajectory,
            "/arm_controller/follow_joint_trajectory",
        )
        self.gripper_client = ActionClient(
            self,
            FollowJointTrajectory,
            "/gripper_controller/follow_joint_trajectory",
        )
        self.ik_client = self.create_client(GetPositionIK, "/compute_ik")
        self._latest_joint_state = None
        self._joint_state_sub = self.create_subscription(
            JointState,
            "/joint_states",
            self._on_joint_state,
            10,
        )

    def _on_joint_state(self, msg):
        self._latest_joint_state = msg

    def joint_snapshot(self, timeout_sec=2.0):
        deadline = time.time() + float(timeout_sec)
        while rclpy.ok() and self._latest_joint_state is None and time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)

        msg = self._latest_joint_state
        if msg is None:
            return {}

        return {
            name: float(pos)
            for name, pos in zip(msg.name, msg.position)
            if name in set(ARM_JOINTS + GRIPPER_JOINTS)
        }

    def send_trajectory(self, client, joint_names, positions, duration_sec: float):
        if not client.wait_for_server(timeout_sec=10.0):
            raise RuntimeError(f"Action server not available: {client._action_name}")

        goal = FollowJointTrajectory.Goal()
        traj = JointTrajectory()
        traj.joint_names = list(joint_names)

        point = JointTrajectoryPoint()
        point.positions = [float(x) for x in positions]
        point.time_from_start = duration_msg(duration_sec)

        traj.points = [point]
        goal.trajectory = traj
        goal.goal_time_tolerance = duration_msg(2.0)

        self.get_logger().info(f"send {client._action_name}: {point.positions}")

        send_future = client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_future)
        goal_handle = send_future.result()

        if goal_handle is None or not goal_handle.accepted:
            raise RuntimeError(f"Goal rejected by {client._action_name}")

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        return result_future.result()

    def arm(self, q, duration=2.5):
        return self.send_trajectory(self.arm_client, ARM_JOINTS, q, duration)

    def gripper(self, q, duration=0.8):
        return self.send_trajectory(
            self.gripper_client,
            GRIPPER_JOINTS,
            gripper_positions_from_r(q),
            duration,
        )

    def compute_ik(self, xyz_base, seed, ik_link_name="end_effector_link"):
        if not self.ik_client.wait_for_service(timeout_sec=10.0):
            raise RuntimeError(
                "MoveIt /compute_ik service not available. "
                "Start: ros2 launch robot_moveit_config move_group.launch.py use_sim_time:=true"
            )

        req = GetPositionIK.Request()
        req.ik_request.group_name = "arm"
        req.ik_request.ik_link_name = ik_link_name
        req.ik_request.avoid_collisions = False
        req.ik_request.timeout = duration_msg(1.0)

        js = JointState()
        js.name = list(ARM_JOINTS)
        js.position = [float(x) for x in seed]
        req.ik_request.robot_state.joint_state = js

        ps = PoseStamped()
        ps.header.frame_id = "base_link"
        ps.header.stamp.sec = 0
        ps.header.stamp.nanosec = 0
        ps.pose.position.x = float(xyz_base[0])
        ps.pose.position.y = float(xyz_base[1])
        ps.pose.position.z = float(xyz_base[2])
        ps.pose.orientation.w = 1.0
        req.ik_request.pose_stamped = ps

        future = self.ik_client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
        resp = future.result()

        if resp is None:
            return None, "ik_timeout"

        if resp.error_code.val != MoveItErrorCodes.SUCCESS:
            return None, f"ik_error_{resp.error_code.val}"

        sol = dict(zip(resp.solution.joint_state.name, resp.solution.joint_state.position))
        missing = [j for j in ARM_JOINTS if j not in sol]
        if missing:
            return None, f"ik_missing_joints_{missing}"

        return [float(sol[j]) for j in ARM_JOINTS], "ok"


def run_checked(cmd, timeout=30):
    p = subprocess.run(
        cmd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
    )
    if p.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(map(str, cmd))}\n{p.stdout}")
    return p.stdout


def get_truth(tag: str):
    out_path = RESULT_DIR / f"truth_{tag}.json"
    run_checked(["python3", str(TRUTH_SCRIPT), "--out", str(out_path)], timeout=20)
    return json.loads(out_path.read_text())


def run_perception():
    run_checked(["python3", str(PERCEPTION_SCRIPT)], timeout=30)

    if not PERCEPTION_JSON.exists():
        raise RuntimeError(f"Perception JSON not found: {PERCEPTION_JSON}")

    data = json.loads(PERCEPTION_JSON.read_text())

    if not data.get("detected", False):
        raise RuntimeError("perception_miss")
    if not data.get("depth_valid", False):
        raise RuntimeError("depth_fail")
    if not data.get("tf", {}).get("transform_found", False):
        raise RuntimeError("tf_fail")
    if data.get("estimate_base_frame") is None:
        raise RuntimeError("estimate_base_frame_missing")

    return data


def calibrated_world_from_perception(perception):
    cal = json.loads(CALIB_JSON.read_text())
    R = np.array(cal["R"], dtype=float)
    t = np.array(cal["t"], dtype=float)

    p = perception["estimate_base_frame"]
    p_est = np.array([p["x"], p["y"], p["z"]], dtype=float)
    p_world = R @ p_est + t
    return p_world.tolist()


def world_to_base(p_world):
    return (np.array(p_world, dtype=float) - ROBOT_BASE_WORLD).tolist()


def append_csv(row):
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULT_DIR / "day7_rule_baseline_trials.csv"
    new_file = not csv_path.exists()

    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trial-id", default=f"trial_{int(time.time())}")
    parser.add_argument("--use-truth", action="store_true")
    parser.add_argument("--skip-home", action="store_true")

    parser.add_argument("--pre-z-base", type=float, default=0.22)
    parser.add_argument("--down-z-base", type=float, default=0.095)
    parser.add_argument("--lift-z-base", type=float, default=0.26)

    parser.add_argument("--x-offset", type=float, default=0.0)
    parser.add_argument("--y-offset", type=float, default=0.0)
    parser.add_argument("--close", type=float, default=GRIPPER_CLOSE)
    parser.add_argument(
        "--close-seq",
        default="",
        help="Comma-separated gripper closing sequence, e.g. 0.6,0.4,0.2,0.0",
    )
    parser.add_argument("--close-wait", type=float, default=0.8)
    parser.add_argument("--gripper-close-duration", type=float, default=1.2)
    parser.add_argument("--lift-duration", type=float, default=3.5)
    parser.add_argument("--down-duration", type=float, default=2.5)
    parser.add_argument("--hold-after-down", type=float, default=1.5)
    parser.add_argument("--hold-after-close", type=float, default=2.0)
    parser.add_argument(
        "--stop-after-close",
        action="store_true",
        help="Move to down pose, close gripper, record truth_after_close, then stop before lift.",
    )
    parser.add_argument(
        "--stop-after-down",
        action="store_true",
        help="Only move to down pose, record truth_after_down, then stop before close/lift.",
    )

    args = parser.parse_args()

    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    log = {
        "trial_id": args.trial_id,
        "use_truth": bool(args.use_truth),
        "success": False,
        "failure_type": "unknown",
        "t_start": time.time(),
        "params": {
            "pre_z_base": args.pre_z_base,
            "down_z_base": args.down_z_base,
            "lift_z_base": args.lift_z_base,
            "x_offset": args.x_offset,
            "y_offset": args.y_offset,
            "close": args.close,
            "close_seq": args.close_seq,
            "close_wait": args.close_wait,
            "gripper_close_duration": args.gripper_close_duration,
            "down_duration": args.down_duration,
            "lift_duration": args.lift_duration,
        },
    }

    rclpy.init()
    node = MotionAndIK()

    try:
        try:
            if not args.skip_home:
                log["stage"] = "home"
                node.arm(HOME, duration=3.0)

            log["stage"] = "observe"
            node.gripper(GRIPPER_OPEN, duration=0.8)
            time.sleep(0.2)
            log["joint_after_open_1"] = node.joint_snapshot()
            node.arm(OBSERVE, duration=3.0)
            node.gripper(GRIPPER_OPEN, duration=0.8)
            time.sleep(0.2)
            log["joint_after_open_2"] = node.joint_snapshot()

            truth_before = get_truth(args.trial_id + "_before")
            log["truth_before"] = truth_before

            if args.use_truth:
                p_world = [truth_before["x"], truth_before["y"], truth_before["z"]]
                log["target_source"] = "gazebo_truth"
                log["perception"] = None
            else:
                log["stage"] = "perception"
                perception = run_perception()
                p_world = calibrated_world_from_perception(perception)
                log["target_source"] = "color_rgbd_calibrated"
                log["perception"] = {
                    "detected": perception.get("detected"),
                    "depth_valid": perception.get("depth_valid"),
                    "tf": perception.get("tf"),
                    "estimate_base_frame": perception.get("estimate_base_frame"),
                    "color_detection": perception.get("color_detection"),
                }

            p_base = world_to_base(p_world)
            p_base[0] += args.x_offset
            p_base[1] += args.y_offset

            log["target_world_xyz"] = [float(x) for x in p_world]
            log["target_base_xy"] = [float(p_base[0]), float(p_base[1])]

            waypoints = {
                "pregrasp": [p_base[0], p_base[1], args.pre_z_base],
                "down": [p_base[0], p_base[1], args.down_z_base],
                "lift": [p_base[0], p_base[1], args.lift_z_base],
            }
            log["waypoints_base"] = waypoints

            seed = OBSERVE
            q_targets = {}

            for name in ["pregrasp", "down", "lift"]:
                q, status = node.compute_ik(
                    waypoints[name],
                    seed,
                    ik_link_name="end_effector_link",
                )

                if q is None:
                    q, status = node.compute_ik(
                        waypoints[name],
                        seed,
                        ik_link_name="link5",
                    )

                if q is None:
                    raise RuntimeError(f"ik_fail_{name}_{status}")

                q_targets[name] = q
                seed = q

            log["q_targets"] = q_targets

            log["stage"] = "pregrasp"
            node.arm(q_targets["pregrasp"], duration=2.5)

            log["stage"] = "down"
            node.arm(q_targets["down"], duration=args.down_duration)
            time.sleep(args.hold_after_down)
            log["joint_after_down"] = node.joint_snapshot()

            truth_after_down = get_truth(args.trial_id + "_after_down")
            log["truth_after_down"] = truth_after_down

            if args.stop_after_down:
                log["truth_after"] = truth_after_down

                bx = float(truth_before["x"])
                by = float(truth_before["y"])
                bz = float(truth_before["z"])
                ax = float(truth_after_down["x"])
                ay = float(truth_after_down["y"])
                az = float(truth_after_down["z"])

                probe_dx = ax - bx
                probe_dy = ay - by
                probe_dz = az - bz

                log["probe_motion"] = {
                    "dx": probe_dx,
                    "dy": probe_dy,
                    "dz": probe_dz,
                    "abs_xy": float((probe_dx ** 2 + probe_dy ** 2) ** 0.5),
                }

                moved = (
                    abs(probe_dx) > 0.003
                    or abs(probe_dy) > 0.003
                    or abs(probe_dz) > 0.003
                )

                log["success"] = False
                log["failure_type"] = "probe_contact_or_motion" if moved else "probe_no_contact"
                log["stage"] = "probe_done"
                raise SystemExit(0)

            log["stage"] = "close"
            if args.close_seq.strip():
                close_values = [
                    float(x.strip())
                    for x in args.close_seq.split(",")
                    if x.strip()
                ]
            else:
                close_values = [float(args.close)]

            log["close_sequence"] = close_values

            for i, cv in enumerate(close_values):
                log["stage"] = f"close_{i}_{cv}"
                node.gripper(cv, duration=args.gripper_close_duration)
                time.sleep(args.close_wait)
                log[f"joint_after_close_{i}"] = node.joint_snapshot()
                log[f"truth_after_close_{i}"] = get_truth(args.trial_id + f"_after_close_{i}")

            if args.stop_after_close:
                log["truth_after"] = log[f"truth_after_close_{len(close_values) - 1}"]
                log["success"] = False
                log["failure_type"] = "probe_stop_after_close"
                log["stage"] = "probe_close_done"
                raise SystemExit(0)

            log["stage"] = "lift"
            node.arm(q_targets["lift"], duration=args.lift_duration)

            time.sleep(0.8)
            log["joint_after_lift"] = node.joint_snapshot()

            truth_after = get_truth(args.trial_id + "_after")
            log["truth_after"] = truth_after

            z_before = float(truth_before["z"])
            z_after = float(truth_after["z"])

            lifted = (z_after > 0.825) or (z_after - z_before > 0.035)

            log["success"] = bool(lifted)
            log["failure_type"] = "success" if lifted else "grasp_no_lift"
            log["stage"] = "done"

        except RuntimeError as e:
            msg = str(e)

            if "perception_miss" in msg:
                log["failure_type"] = "perception_miss"
            elif "depth_fail" in msg:
                log["failure_type"] = "depth_fail"
            elif "tf_fail" in msg:
                log["failure_type"] = "tf_fail"
            elif "ik_fail" in msg:
                log["failure_type"] = "ik_fail"
            elif "Action server" in msg or "Goal rejected" in msg:
                log["failure_type"] = "controller_fail"
            else:
                log["failure_type"] = "runtime_error"

            log["error"] = msg

        finally:
            log["t_end"] = time.time()
            log["duration_s"] = log["t_end"] - log["t_start"]

            out = RESULT_DIR / f"{args.trial_id}.json"
            out.write_text(json.dumps(log, indent=2, ensure_ascii=False))

            target = log.get("target_world_xyz") or ["", "", ""]
            row = {
                "trial_id": log["trial_id"],
                "use_truth": log["use_truth"],
                "target_source": log.get("target_source", ""),
                "success": log["success"],
                "failure_type": log["failure_type"],
                "duration_s": f"{log['duration_s']:.3f}",
                "target_world_x": target[0],
                "target_world_y": target[1],
                "target_world_z": target[2],
                "truth_before_z": (log.get("truth_before") or {}).get("z", ""),
                "truth_after_z": (log.get("truth_after") or {}).get("z", ""),
                "json_path": str(out),
            }
            append_csv(row)

            print(json.dumps(log, indent=2, ensure_ascii=False))
            print(f"\n[DAY7] saved: {out}")

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
