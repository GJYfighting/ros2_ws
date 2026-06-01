#!/usr/bin/env python3
import math
import re
import subprocess
import tempfile
import time
from pathlib import Path
import yaml

def load_cfg(cfg_path):
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)

def run_cmd(cmd, check=True):
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if check and proc.returncode != 0:
        raise RuntimeError(
            f"Command failed: {' '.join(cmd)}\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )
    return proc

def list_services():
    proc = run_cmd(["ign", "service", "--list"], check=True)
    return [x.strip() for x in proc.stdout.splitlines() if x.strip()]

def list_topics():
    proc = run_cmd(["ign", "topic", "-l"], check=True)
    return [x.strip() for x in proc.stdout.splitlines() if x.strip()]

def service_exists(name):
    return name in list_services()

def topic_exists(name):
    return name in list_topics()

def yaw_to_quat(yaw):
    qz = math.sin(yaw / 2.0)
    qw = math.cos(yaw / 2.0)
    return 0.0, 0.0, qz, qw

def set_pose(world_name, model_name, x, y, z, yaw):
    _, _, qz, qw = yaw_to_quat(yaw)
    req = (
        f'name: "{model_name}" '
        f'position: {{ x: {x} y: {y} z: {z} }} '
        f'orientation: {{ x: 0.0 y: 0.0 z: {qz} w: {qw} }}'
    )
    proc = run_cmd([
        "ign", "service",
        "-s", f"/world/{world_name}/set_pose",
        "--reqtype", "ignition.msgs.Pose",
        "--reptype", "ignition.msgs.Boolean",
        "--timeout", "2000",
        "--req", req
    ], check=True)
    return proc.stdout.strip()

def remove_model(world_name, model_name):
    req = f'type: MODEL name: "{model_name}"'
    proc = run_cmd([
        "ign", "service",
        "-s", f"/world/{world_name}/remove",
        "--reqtype", "ignition.msgs.Entity",
        "--reptype", "ignition.msgs.Boolean",
        "--timeout", "2000",
        "--req", req
    ], check=True)
    return proc.stdout.strip()

def create_model_from_sdf(world_name, model_name, sdf_filename, x, y, z, yaw):
    _, _, qz, qw = yaw_to_quat(yaw)
    req = (
        f'name: "{model_name}"; '
        f'sdf_filename: "{sdf_filename}"; '
        f'pose: {{ '
        f'position: {{ x: {x} y: {y} z: {z} }} '
        f'orientation: {{ x: 0.0 y: 0.0 z: {qz} w: {qw} }} '
        f'}}'
    )
    proc = run_cmd([
        "ign", "service",
        "-s", f"/world/{world_name}/create",
        "--reqtype", "ignition.msgs.EntityFactory",
        "--reptype", "ignition.msgs.Boolean",
        "--timeout", "3000",
        "--req", req
    ], check=True)
    return proc.stdout.strip()

def choose_pose_topic(world_name):
    preferred = f"/world/{world_name}/dynamic_pose/info"
    fallback = f"/world/{world_name}/pose/info"
    topics = list_topics()
    if preferred in topics:
        return preferred
    if fallback in topics:
        return fallback
    raise RuntimeError(
        f"Neither {preferred} nor {fallback} exists. "
        "Run: ign topic -l | grep /world/<world>/"
    )

def capture_pose_text(topic, seconds=1.5):
    # 用 timeout 截一小段持续发布的 topic 文本
    proc = subprocess.run(
        ["timeout", str(seconds), "ign", "topic", "-e", "-t", topic],
        capture_output=True,
        text=True
    )
    text = proc.stdout.strip()
    if not text:
        raise RuntimeError(
            f"No text captured from {topic}. "
            "Please check that Gazebo is running and the topic is active."
        )
    return text

def split_pose_blocks(text):
    blocks = []
    current = []
    level = 0
    capturing = False

    for raw in text.splitlines():
        line = raw.rstrip()
        stripped = line.strip()

        if stripped.startswith("pose {") and not capturing:
            capturing = True
            current = [stripped]
            level = 1
            continue

        if capturing:
            current.append(stripped)
            level += stripped.count("{")
            level -= stripped.count("}")
            if level == 0:
                blocks.append("\n".join(current))
                current = []
                capturing = False

    return blocks

def parse_pose_block(block, model_name):
    if f'name: "{model_name}"' not in block:
        return None

    pos = {}
    ori = {}
    section = None

    for raw in block.splitlines():
        line = raw.strip()

        if line.startswith("position {"):
            section = "pos"
            continue
        if line.startswith("orientation {"):
            section = "ori"
            continue
        if line == "}":
            section = None
            continue

        m = re.match(r'([xyzw]):\s*([-+0-9.eE]+)', line)
        if m and section == "pos":
            pos[m.group(1)] = float(m.group(2))
        elif m and section == "ori":
            ori[m.group(1)] = float(m.group(2))

    if all(k in pos for k in ["x", "y", "z"]) and all(k in ori for k in ["x", "y", "z", "w"]):
        return {
            "name": model_name,
            "x": pos["x"],
            "y": pos["y"],
            "z": pos["z"],
            "qx": ori["x"],
            "qy": ori["y"],
            "qz": ori["z"],
            "qw": ori["w"],
        }
    return None

def get_truth_pose(world_name, model_name):
    topic = choose_pose_topic(world_name)
    text = capture_pose_text(topic)
    blocks = split_pose_blocks(text)

    for block in blocks:
        parsed = parse_pose_block(block, model_name)
        if parsed is not None:
            parsed["source_topic"] = topic
            return parsed

    raise RuntimeError(
        f"Model '{model_name}' not found in pose stream from {topic}."
    )

def cube_inertia(mass, size):
    i = mass * (size ** 2) / 6.0
    return i, i, i

def build_block_sdf(model_name, size, mass, mu, rgba):
    ixx, iyy, izz = cube_inertia(mass, size)
    r, g, b, a = rgba
    return f"""<?xml version='1.0'?>
<sdf version='1.7'>
  <model name='{model_name}'>
    <link name='block_link'>
      <inertial>
        <mass>{mass}</mass>
        <inertia>
          <ixx>{ixx}</ixx>
          <iyy>{iyy}</iyy>
          <izz>{izz}</izz>
          <ixy>0</ixy>
          <ixz>0</ixz>
          <iyz>0</iyz>
        </inertia>
      </inertial>

      <collision name='block_collision'>
        <geometry>
          <box>
            <size>{size} {size} {size}</size>
          </box>
        </geometry>
        <surface>
          <friction>
            <ode>
              <mu>{mu}</mu>
              <mu2>{mu}</mu2>
            </ode>
          </friction>
        </surface>
      </collision>

      <visual name='block_visual'>
        <geometry>
          <box>
            <size>{size} {size} {size}</size>
          </box>
        </geometry>
        <material>
          <ambient>{r} {g} {b} {a}</ambient>
          <diffuse>{r} {g} {b} {a}</diffuse>
          <specular>0.1 0.1 0.1 1.0</specular>
        </material>
      </visual>
    </link>
  </model>
</sdf>
"""

def write_temp_block_sdf(tmp_dir, model_name, size, mass, mu, rgba):
    Path(tmp_dir).mkdir(parents=True, exist_ok=True)
    tmp_path = Path(tmp_dir) / f"{model_name}_runtime.sdf"
    tmp_path.write_text(build_block_sdf(model_name, size, mass, mu, rgba))
    return str(tmp_path)
