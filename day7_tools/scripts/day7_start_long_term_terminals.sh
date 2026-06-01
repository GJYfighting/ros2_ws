#!/usr/bin/env bash
set -u

WS="$HOME/ros2_ws"
LOG_DIR="$WS/day7_tools/results/logs"
PID_FILE="$WS/day7_tools/results/day7_long_term_processes.txt"
TS="$(date +%Y%m%d_%H%M%S)"

mkdir -p "$LOG_DIR"

GAZEBO_LOG="$LOG_DIR/day7_gazebo_${TS}.log"
CAMERA_LOG="$LOG_DIR/day7_camera_info_${TS}.log"
MOVEIT_LOG="$LOG_DIR/day7_moveit_${TS}.log"

GAZEBO_CMD="cd '$WS'; source /opt/ros/humble/setup.bash; source '$WS/install/setup.bash'; CAMERA_TYPE=GEMINI ros2 launch robot_gazebo worlds.launch.py world_name:=grasp_table 2>&1 | tee '$GAZEBO_LOG'; exec bash"
CAMERA_CMD="cd '$WS'; source /opt/ros/humble/setup.bash; source '$WS/install/setup.bash'; python3 '$WS/camera_info_republisher.py' 2>&1 | tee '$CAMERA_LOG'; exec bash"
MOVEIT_CMD="cd '$WS'; source /opt/ros/humble/setup.bash; source '$WS/install/setup.bash'; ros2 launch robot_moveit_config move_group.launch.py use_sim_time:=true 2>&1 | tee '$MOVEIT_LOG'; exec bash"

open_term() {
  local title="$1"
  local cmd="$2"

  if command -v gnome-terminal >/dev/null 2>&1; then
    gnome-terminal --title="$title" -- bash -lc "$cmd" >/dev/null 2>&1 && return 0
  fi

  if command -v x-terminal-emulator >/dev/null 2>&1; then
    x-terminal-emulator -T "$title" -e bash -lc "$cmd" >/dev/null 2>&1 && return 0
  fi

  return 1
}

start_nohup() {
  : > "$PID_FILE"

  bash -lc "cd '$WS'; source /opt/ros/humble/setup.bash; source '$WS/install/setup.bash'; CAMERA_TYPE=GEMINI ros2 launch robot_gazebo worlds.launch.py world_name:=grasp_table" > "$GAZEBO_LOG" 2>&1 &
  echo "gazebo_pid=$! log=$GAZEBO_LOG" >> "$PID_FILE"
  sleep 15

  bash -lc "cd '$WS'; source /opt/ros/humble/setup.bash; source '$WS/install/setup.bash'; python3 '$WS/camera_info_republisher.py'" > "$CAMERA_LOG" 2>&1 &
  echo "camera_info_pid=$! log=$CAMERA_LOG" >> "$PID_FILE"
  sleep 5

  bash -lc "cd '$WS'; source /opt/ros/humble/setup.bash; source '$WS/install/setup.bash'; ros2 launch robot_moveit_config move_group.launch.py use_sim_time:=true" > "$MOVEIT_LOG" 2>&1 &
  echo "moveit_pid=$! log=$MOVEIT_LOG" >> "$PID_FILE"
}

echo "[DAY7] log_dir=$LOG_DIR"
echo "[DAY7] timestamp=$TS"

if open_term "DAY7-1 Gazebo grasp_table" "$GAZEBO_CMD"; then
  echo "[DAY7] opened terminal 1: Gazebo"
  sleep 15
  if open_term "DAY7-2 camera_info" "$CAMERA_CMD"; then
    echo "[DAY7] opened terminal 2: camera_info"
    sleep 5
    if open_term "DAY7-3 MoveIt move_group" "$MOVEIT_CMD"; then
      echo "[DAY7] opened terminal 3: MoveIt"
      echo "[DAY7] gazebo_log=$GAZEBO_LOG"
      echo "[DAY7] camera_info_log=$CAMERA_LOG"
      echo "[DAY7] moveit_log=$MOVEIT_LOG"
      exit 0
    fi
  fi
fi

echo "[DAY7] graphical terminal unavailable, falling back to nohup background processes"
start_nohup
echo "[DAY7] background process file=$PID_FILE"
cat "$PID_FILE"
