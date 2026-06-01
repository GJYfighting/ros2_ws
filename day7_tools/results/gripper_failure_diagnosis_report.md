# Day7 夹爪抓取失败诊断报告

生成时间：2026-05-28  
工作空间：`/home/ubuntu/ros2_ws`  
约束：本报告只基于只读检查、新建 `day7_tools/results` 报告文件；未修改机器人核心模型、控制器配置、MoveIt 配置，未删除已有文件。

## 1. 仓库和文件地图

### Day3-Day7 工具与结果

- `day3_tools/config/observe_pose.yaml`
  - Day3 观察位姿配置：`arm: [0.0, 0.7309, -1.6127, -1.6591, 0.0]`，`gripper: 1.4`。
- `day3_tools/scripts/send_named_pose.py`
  - 发送命名关节位姿。
- `day4_tools/scripts/set_block_pose.py`
  - 用 Ignition/Gazebo 服务设置 `wood_block` 位姿。
- `day4_tools/scripts/get_block_truth.py`
  - 从 Ignition/Gazebo 动态位姿读取 `wood_block` 真值。
- `day4_tools/scripts/ign_world_utils.py`
  - Day4 Ignition 工具函数；包含 `wood_block` SDF 生成逻辑。
- `day4_tools/tmp_models/wood_block_runtime.sdf`
  - 运行期生成的 `wood_block` SDF。
- `day5_method_a/scripts/method_a_color_rgbd_once.py`
  - OpenCV 红色块检测 + RGB-D 反投影。
- `day5_method_a/results/method_a_color_once.json`
  - 最近一次 Method A 感知输出。
- `day6_tools/results/est_base_to_world_calibration.json`
  - Day6 刚性标定：`p_world = R @ p_est_base + t`。
- `day6_tools/results/perception_error_model.json`
  - Day6 感知误差模型。
- `day7_baseline/scripts/day7_rule_baseline_once.py`
  - 已有 Day7 单次规则式抓取脚本；执行 Home/Observe/Pregrasp/Down/Close/Lift。
- `day7_baseline/scripts/day7_batch_baseline.py`
  - 已有批量统计脚本。
- `day7_baseline/scripts/day7_hold_close_lift_probe.py`
  - 保持夹爪 goal active 后 lift 的诊断脚本。
- `day7_baseline/scripts/day7_hold_close_multilift_probe.py`
  - 多阶段 lift 诊断脚本。
- `day7_baseline/scripts/day7_positive_close_root_probe.py`
  - 正向 close/root 接触诊断脚本。
- `day7_baseline/results/day7_rule_baseline_trials.csv`
  - 已有 Day7 trial 汇总。
- `day7_baseline/results/step7_*.json`
  - 已有大量 down_z、close、x/y offset、proxy collision、hold/lift 探针结果。

### 机器人、夹爪控制相关文件

- `src/simulations/robot_moveit_config/config/ros2_controllers.yaml`
  - MoveIt/ros2_control 控制器配置。
  - `arm_controller` 为 `joint_trajectory_controller/JointTrajectoryController`，控制 `joint1`-`joint5`。
  - `gripper_controller` 为 `joint_trajectory_controller/JointTrajectoryController`，只配置 `r_joint`。
- `src/simulations/robot_moveit_config/config/jetarm_6dof.ros2_control.xacro`
  - MoveIt fake/system ros2_control 描述。
  - 只暴露 `joint1`-`joint5` 和 `r_joint` 的 position command interface。
- `src/simulations/jetarm_6dof_description/gazebo/jetarm.transmission.xacro`
  - Gazebo transmission 宏。
  - 只为 `joint1`-`joint5` 和 `r_joint` 生成 transmission。
- `src/simulations/robot_gazebo/config/robot_config.yaml`
  - Gazebo/ign_ros2_control 使用的控制器配置。存在 `robot_config.yaml.bak_day7_multijoint`，说明此前曾尝试过多夹爪关节配置备份。
- `src/simulations/robot_gazebo/urdf/robot.gazebo.xacro`
  - 加载 `ign_ros2_control::IgnitionROS2ControlPlugin`，参数文件为 `robot_gazebo/config/robot_config.yaml`。

### 夹爪结构与物理碰撞相关文件

- `src/simulations/jetarm_6dof_description/urdf/gripper.urdf.xacro`
  - 夹爪 URDF/xacro 主文件。
  - 主动关节：`r_joint`。
  - mimic 关节：`l_joint`、`l_in_joint`、`l_out_joint`、`r_in_joint`、`r_out_joint` 均 mimic `r_joint`。
  - 夹爪多数 link 的 collision 使用 STL mesh；当前 `l_out_link` 和 `r_out_link` 已被改为 box collision：`0.040 0.022 0.050`。
  - 存在备份：`gripper.urdf.xacro.bak_day7_proxy_collision`、`gripper.urdf.xacro.bak_day7_remove_mimic`。
- `src/simulations/jetarm_6dof_description/gazebo/jetarm.gazebo.xacro`
  - Gazebo 物理参数。
  - `gazebo_gripper` 对 `r_link`、`l_link`、`l_out_link`、`r_out_link` 设置 `kp=1000000`、`kd=1000`、`mu1=8.0`、`mu2=8.0`、`minDepth=0.001`。
- `src/simulations/jetarm_6dof_description/meshes/gripper/*.STL`
  - 夹爪视觉/碰撞 mesh 资源。

### wood_block 物理属性相关文件

- `src/simulations/robot_gazebo/worlds/grasp_table.sdf`
  - world 内置 `wood_block`：
    - pose：`0.12 0.00 0.775 0 0 0`
    - collision：box，`0.05 0.05 0.05`
    - mass：`0.03`
    - inertia：`1.25e-5`
    - friction：`mu=10.0`，`mu2=10.0`
    - contact：`kp=10000000`，`kd=10000`，`max_vel=0.01`，`min_depth=0.001`
- `day4_tools/scripts/ign_world_utils.py`
  - 运行期 `wood_block` SDF 生成：
    - collision/visual 均为 box。
    - 惯性按立方体公式 `mass * size^2 / 6`。
    - friction `mu`/`mu2` 由参数传入。
- `day4_tools/tmp_models/wood_block_runtime.sdf`
  - 当前 runtime block：
    - size：`0.05 0.05 0.05`
    - mass：`0.02793287699887069`
    - inertia：`1.1638698749529455e-05`
    - friction：`mu=0.8914426144413563`，`mu2=0.8914426144413563`

## 2. 在线系统状态检查

执行环境：`/home/ubuntu/ros2_ws`，已 source `/opt/ros/humble/setup.bash` 和 `/home/ubuntu/ros2_ws/install/setup.bash`。

### 结果

- `ROS_DOMAIN_ID=0`
- `ros2 topic list --no-daemon`
  - 只看到：
    - `/parameter_events`
    - `/rosout`
- `ros2 node list --no-daemon`
  - 无节点输出。
- `ros2 service list --no-daemon`
  - 无服务输出。
- `ros2 topic echo /joint_states --once`
  - 失败：`topic [/joint_states] does not appear to be published yet`
- `ros2 control list_controllers`
  - 超时等待 `/controller_manager/list_controllers`。
- `ros2 control list_hardware_interfaces`
  - 超时等待 `/controller_manager/list_hardware_interfaces`。
- `ign topic -l`
  - 无 topic 输出。
- `ign service -l`
  - 无 service 输出。

### 验收项状态

- `joint_state_broadcaster active`：未满足，当前无 controller_manager。
- `arm_controller active`：未满足。
- `gripper_controller active`：未满足。
- `/compute_ik`：未满足，当前无 service。
- `/joint_states` 可读：未满足。
- RGB-D 和 camera_info 话题：未满足，当前无 `/depth_cam/rgbd/image`、`/depth_cam/rgbd/depth_image`、`/depth_cam/rgbd/camera_info`。
- `wood_block` 在线：未满足，当前 Ignition 无 world/topic/service 输出。

当前不能继续固定 5 次和随机 20 次统计。需要先启动：

终端 1：

```bash
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash
CAMERA_TYPE=GEMINI ros2 launch robot_gazebo worlds.launch.py world_name:=grasp_table
```

终端 2：

```bash
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash
python3 ~/ros2_ws/camera_info_republisher.py
```

终端 3：

```bash
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash
ros2 launch robot_moveit_config move_group.launch.py use_sim_time:=true
```

终端 4 当前 Codex 终端后续需要：

```bash
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash
source ~/ros2_ws/venv_day5/bin/activate
```

## 3. 夹爪失败原因证据

### A. 控制器证据

文件：`src/simulations/robot_moveit_config/config/ros2_controllers.yaml`

- `gripper_controller` 类型：`joint_trajectory_controller/JointTrajectoryController`。
- joints 列表只有：
  - `r_joint`
- command interface：
  - `position`
- state interfaces：
  - `position`
  - `velocity`

文件：`src/simulations/robot_moveit_config/config/jetarm_6dof.ros2_control.xacro`

- ros2_control 只暴露一个夹爪 command joint：`r_joint`。
- `l_joint`、`l_in_joint`、`l_out_joint`、`r_in_joint`、`r_out_joint` 没有 command interface。

文件：`src/simulations/jetarm_6dof_description/gazebo/jetarm.transmission.xacro`

- transmission 只包含：
  - `joint1`-`joint5`
  - `r_joint`
- 其他夹爪关节没有独立 transmission。

结论：控制器证据强支持“gripper_controller 只主动控制 r_joint”。

### B. URDF/xacro/SDF 结构证据

文件：`src/simulations/jetarm_6dof_description/urdf/gripper.urdf.xacro`

夹爪关节结构：

- `r_joint`
  - revolute，父 link `gripper_servo_link`，子 link `r_link`
  - 无 mimic，是唯一主动控制夹爪关节。
- `l_joint`
  - mimic `r_joint`
  - multiplier `-1`
- `l_in_joint`
  - mimic `r_joint`
  - multiplier `-1`
- `l_out_joint`
  - mimic `r_joint`
  - multiplier `1`
- `r_in_joint`
  - mimic `r_joint`
  - multiplier `-1`
- `r_out_joint`
  - mimic `r_joint`
  - multiplier `1`

这是一套单主动关节驱动多连杆夹爪的结构。对 MoveIt/robot_state_publisher 来说 mimic 可以同步状态；但在 Gazebo 物理接触中，如果 mimic 关节没有独立 actuator/constraint 行为或 ros2_control 不对它们施加闭环命令，实际双侧夹持力容易依赖仿真器对 mimic 关节的处理质量。

结论：结构证据强支持“其他夹爪关节依赖 mimic”。

### C. collision 与物理参数证据

夹爪 collision：

- `r_link`、`l_link`、`l_in_link`、`r_in_link` 仍使用 STL mesh collision。
- `l_out_link`、`r_out_link` 当前使用 box collision：`0.040 0.022 0.050`。
- 左右外指 box collision 尺寸对称，origin 分别约为：
  - 左：`0.0060 0.0015 0.0096`
  - 右：`0.0060 -0.0015 0.0096`
- 夹爪 link 质量很小，典型为毫克到克级：
  - `r_link` mass 约 `0.00206`
  - `l_link` mass 约 `0.00219`
  - `l_in_link`/`r_in_link` mass 约 `0.000777`
  - `l_out_link`/`r_out_link` mass 约 `0.003847`

Gazebo 夹爪摩擦：

- `r_link`、`l_link`、`l_out_link`、`r_out_link` 设置 `mu1=8.0`、`mu2=8.0`。
- `l_in_link`、`r_in_link` 仅设置 black material，没有 `gazebo_gripper` 的高摩擦参数。

wood_block：

- world 内置版本是规则 box collision，质量 `0.03 kg`，摩擦 `mu=10.0`，接触刚度较高。
- Day4 runtime 版本也是规则 box collision，质量约 `0.02793 kg`，但摩擦约 `0.891`。

风险判断：

- `wood_block` collision 本身简单且合理，不像主要问题。
- 夹爪 collision 曾经或仍然部分依赖复杂 STL mesh；虽然外指已经被替换为 box proxy，但根部/中间连杆仍为 mesh，且多 mimic 关节没有独立控制。
- 外指 box proxy 不能自动保证物理夹持力足够，因为夹持力来源仍来自单 `r_joint` 及 mimic 结构，而不是左右指尖独立受控闭合。

结论：collision 证据支持存在明显物理风险；wood_block 本身不是主要嫌疑。

### D. `/joint_states` 运行证据

当前系统未启动，不能实时采集打开/闭合时 `/joint_states` 中各夹爪关节变化。

已有 Day7 日志间接证据：

- Day7 脚本 `day7_rule_baseline_once.py` 中 `GRIPPER_JOINTS = ["r_joint"]`，发送给 `/gripper_controller/follow_joint_trajectory` 的 positions 只有 `[q]`。
- 现有探针日志记录的是 `actual_r_joint_before_lift` 和 `actual_r_joint_after_lift`，没有其他夹爪关节独立命令记录。
- 代表性日志：
  - `step7_hold_c09.json`
    - `actual_r_joint_before_lift = 1.2316297969`
    - `actual_r_joint_after_lift = 0.9000859291`
    - 诊断：夹爪保持接触但物块不跟随，剩余问题是 contact geometry/friction。
  - `step7_hold_proxy_c09.json`
    - `actual_r_joint_before_lift = 1.2352444648`
    - `actual_r_joint_after_lift = 0.9000185316`
  - `step7_bigproxy_y0_c09.json`
    - `actual_r_joint_before_lift = 1.2427778012`
    - lift 阶段 `actual_r_joint` 到约 `0.9`
  - `step7_xoff_0000.json`
    - `actual_r_joint_before_lift = 1.2352444649`
    - lift 阶段 `actual_r_joint` 到约 `0.75`

需要在系统启动后补采的新证据：

```bash
ros2 topic echo /joint_states --once
```

并在发送 open/close 后比较 `r_joint`、`l_joint`、`l_in_joint`、`l_out_joint`、`r_in_joint`、`r_out_joint` 是否都出现在 `/joint_states` 且同步变化。

### E. 抓取结果证据

已有 Day7 汇总 `day7_baseline/results/day7_rule_baseline_trials.csv` 显示：

- 多个 down_z、close、x/y offset 探针均失败。
- 早期多项为 `grasp_no_lift` 或 `probe_no_contact`。
- 在后续接触/代理碰撞/hold close 测试中，失败形态转为：
  - `partial_contact_not_clamped`
  - `not_clamped`
  - `grasp_no_lift`

代表性结果：

- `step7_hold_c09.json`
  - `failure_type = not_clamped`
  - lift 后总 `dz = 0.0000915 m`，约 0.09 mm。
  - 日志诊断明确写到：夹爪保持接触但物块不跟随。
- `step7_hold_proxy_c09.json`
  - `failure_type = not_clamped`
  - lift 后总 `dz = 0.00643 m`，约 6.4 mm，仍远低于成功阈值。
- `step7_bigproxy_y0_c09.json`
  - `failure_type = partial_contact_not_clamped`
  - 多阶段 lift 中最高 `dz` 约 `0.00212 m`，约 2.1 mm。
- `step7_xoff_0000.json`
  - `failure_type = partial_contact_not_clamped`
  - 多阶段 lift 中最高 `dz` 约 `0.00127 m`，约 1.3 mm。
- `fixed_truth_001.json`
  - `failure_type = grasp_no_lift`

这说明失败主要不是“完全没到目标”或“IK 完全失败”，而是接触/轻微带动后不能稳定夹持，lift 时物块不跟随或很快滑脱。

## 4. 明确结论

结论分类：**强支持该假设**。

理由：

1. 控制器确实只主动控制 `r_joint`。
   - `ros2_controllers.yaml` 的 `gripper_controller.joints` 只有 `r_joint`。
   - ros2_control xacro 只给 `r_joint` 提供夹爪 command interface。
   - transmission 只包含 `r_joint`，没有其他夹爪关节 actuator/transmission。

2. 其他夹爪关节确实依赖 mimic。
   - `l_joint`、`l_in_joint`、`l_out_joint`、`r_in_joint`、`r_out_joint` 均 mimic `r_joint`。
   - 左右指和内外连杆不是独立控制的多关节闭环夹持结构。

3. collision/物理结构有明显风险。
   - 夹爪多处 collision 使用 STL mesh；外指虽然已改为 box proxy，但夹持力仍来自单主动关节 + mimic。
   - 夹爪 link 质量很小，mimic 多连杆和接触约束容易在 Gazebo Fortress 中表现为接触但夹持力不足。
   - `wood_block` 是规则 box collision，质量和惯性合理，不是首要嫌疑。

4. 失败形态符合“接触后滑脱/夹不住”，而不是单纯定位错误。
   - 已有日志中 `r_before` 约 `1.23`，说明闭合过程中接触到物块。
   - `r_after` 可达到 `0.9` 或更小，但 lift_dz 多为毫米级或更低。
   - 多次 x/y/down_z/close/proxy collision 尝试后仍以 `partial_contact_not_clamped`、`not_clamped`、`grasp_no_lift` 为主。

限制：

- 当前 ROS2/Gazebo/MoveIt 未启动，无法在本次检查中实时采集 `/joint_states` 的 open/close 动态证据。
- 因此“强支持”主要基于文件结构、控制器配置、已有 Day7 日志，而不是本次新采集的运行时 joint trace。

## 5. 后续 Day7 建议

在不修改机器人模型、控制器配置和 MoveIt 配置的前提下，后续 Day7 应继续定义为：

**规则式基线流程统计与失败分类**，不是继续盲目调参追求真实 Gazebo 夹持成功率。

继续固定 5 次和随机 20 次前，必须先满足：

- `joint_state_broadcaster` active
- `arm_controller` active
- `gripper_controller` active
- `/compute_ik` 存在
- `/joint_states` 可读
- `/depth_cam/rgbd/image`、`/depth_cam/rgbd/depth_image`、`/depth_cam/rgbd/camera_info` 存在
- Ignition world 和 `wood_block` 存在，或能通过 Day4 脚本设置

当前不满足这些条件，所以应暂停，等待仿真、camera_info 和 move_group 启动后再继续统计实验。
