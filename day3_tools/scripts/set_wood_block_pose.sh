#!/usr/bin/env bash
set -e

X="${1:?need x}"
Y="${2:?need y}"
Z="${3:-0.775}"

ign service -s /world/robot_world/set_pose \
  --reqtype ignition.msgs.Pose \
  --reptype ignition.msgs.Boolean \
  --timeout 2000 \
  --req 'name: "wood_block", position: { x: '"${X}"', y: '"${Y}"', z: '"${Z}"' }'
