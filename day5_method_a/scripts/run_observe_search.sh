#!/usr/bin/env bash
set -e

source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash
source ~/ros2_ws/venv_day5/bin/activate

YAML=~/ros2_ws/day5_method_a/config/observe_search_poses.yaml

POSES=(
observe_s01
observe_s02
observe_s03
observe_s04
observe_s05
observe_s06
observe_s07
observe_s08
observe_s09
observe_s10
observe_s11
observe_s12
observe_s13
observe_s14
observe_s15
observe_s16
observe_s17
observe_s18
observe_s19
observe_s20
)

for P in "${POSES[@]}"; do
  echo "=============================="
  echo "Testing $P"
  echo "=============================="

  python3 ~/ros2_ws/day3_tools/scripts/send_named_pose.py \
    --yaml "$YAML" \
    --pose "$P"

  sleep 2

  python3 ~/ros2_ws/day5_method_a/scripts/save_and_score_rgbd_observe.py \
    --pose "$P"

  sleep 1
done

echo "Done. Results:"
echo "~/ros2_ws/day5_method_a/results/observe_search/observe_search_results.csv"
