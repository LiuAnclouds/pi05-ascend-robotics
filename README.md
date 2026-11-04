# Pi0.5 Ascend Robotics

基于 Ascend 310P1 的 pi0.5 量化与机械臂部署工程，提供模型导出、昇腾离线模型（OM）执行，以及 ROS2 双摄观测、任务输入和 Piper 控制。

## 一、部署方案与量化结果

视觉与语言采用 W8A8，动作专家采用 W16A16，记为 **ACT-W16A16**。W 表示权重精度，A 表示激活精度，数字表示位数。八位计算用于线性层；注意力矩阵、归一化等运算保留对应的浮点精度。

| 模块 | 权重 | 激活 | 推理耗时 |
|---|---|---|---:|
| 视觉与语言 | INT8 | 动态 INT8 | 106.112 ms |
| 动作专家，十步去噪 | FP16 | FP16 | 80.545 ms |
| 完整推理 | — | — | **189.967 ms** |

完整推理包含模型输入输出传输和去噪更新，不含加载、图像预处理和机械臂执行。以上是双摄 224 × 224、64 个文本词元、50 点动作块的已验证测试数据。

| 指标 | ACT-W16A16 |
|---|---:|
| 全模型参数中 W8 占比 | **63.44%** |
| 线性层权重中 W8 占比 | **84.22%** |
| 矩阵运算输入激活中 A8 占比 | **62.34%** |
| 相对浮点参考的平均余弦相似度 | **99.7938%** |
| 夹爪平均绝对偏差 | **1.214 mm** |
| 夹爪 P95 偏差 | **2.051 mm** |

同平台全模块八位方案（FULL-W8A8）为 197.444 ms。保留动作专家 FP16 后，完整推理耗时减少 **3.79%**。与 Orin 最快的实测配置相比，310P 耗时减少 **66.43%**。测试条件、完整误差与方案分析见[量化结果](docs/quantization.md)。

## 二、运行环境

| 项目 | 板端配置 |
|---|---|
| 硬件 | 香橙派，Ascend 310P1，ARM64 |
| 系统 | openEuler 22.03 LTS-SP3，Linux 5.10.0+ |
| 昇腾软件 | CANN 8.5.0；驱动包 26.0.t1 |
| ROS2 | Humble，RoboStack ARM64 发行版 |
| 推理环境 | `pi05_humble`，Python 3.11.17，NumPy 1.26.4 |
| 导出环境 | `Pi05_ascend`，Python 3.10.21，PyTorch 2.5.1 |
| 机械臂 | Piper SDK 0.6.2，python-can 4.6.1 |

完整环境创建与首次构建步骤见[环境配置](docs/environment.md)。推理与导出使用独立环境，CANN 和设备驱动由目标板系统提供。

## 三、安装与启动

### 1. 获取源码与模型

```bash
git clone https://github.com/LiuAnclouds/pi05-ascend-robotics.git
cd pi05-ascend-robotics
```

将配套模型放到以下位置。模型权重与 OM 文件单独分发。

| 资源 | 路径 |
|---|---|
| 视觉语言 OM | `config/models/om/part1.om` |
| 串行动作 OM，含 FP16 去噪更新输出 | `config/models/om/part2.om` |
| 分词器 | `config/models/tokenizer.model` |
| 归一化统计 | `config/norm_stats.json` |
| RTC 配套资源，使用 RTC 时提供 | `config/models/rtc/` |

当前串行执行使用设备上的 FP16 去噪更新。`part2.om` 需要同时输出动作速度和下一步动作状态；[模型导出](docs/quantization.md#四模型导出)给出生成方法。

### 2. 首次构建

先按[环境配置](docs/environment.md)创建 `pi05_humble`。在项目根目录执行：

```bash
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
conda activate pi05_humble
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
python -m pip install --no-deps -e .
colcon --log-base outputs/ros2/log build \
  --build-base outputs/ros2/build \
  --install-base outputs/ros2/install \
  --base-paths inference/ros2 inference/ros2_interfaces \
  --symlink-install
```

`outputs/ros2/` 保存 ROS2 构建与安装产物。`--symlink-install` 让安装结果引用源码，日常修改 Python 文件后可直接使用；更改消息定义和入口配置后需要重新构建。

### 3. 启动推理

每个新终端先加载环境，并进入项目根目录：

```bash
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
conda activate pi05_humble
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
source outputs/ros2/install/setup.bash
ros2 launch pi05_inference inference.launch.py
```

公开配置默认 `robot.enabled: false`，启动双摄与推理节点。连接实机时，将 `config/config.yaml` 中的 `robot.enabled` 改为 `true`，配置 CAN：

```bash
sudo bash activate_can.sh can0
```

再使用同一条推理启动命令，启动程序会加入机械臂节点并发布实测状态。关闭实机控制时，状态由外部 `/joint_states` 发布器提供。

### 4. 输入与切换任务

在另一个已加载同一环境的终端中运行：

```bash
ros2 run pi05_inference prompt_node
```

输入任务并按回车，例如：

```text
Pick up the blue cylindrical box and place it in the white square basket.
```

出现 `Prompt changed:` 表示发布成功。继续输入下一行即可切换任务。串行模式会完成当前动作块，下一块使用最新任务；旧任务尚未返回的预测会被丢弃。

## 四、执行与参数

当前默认使用 `serial`（串行）：读取最新观测，预测 50 点，对六关节拟合并限制相邻目标角度，逐点发送，完成后再预测下一块。夹爪保留模型输出，执行范围由 Piper SDK 限制。

| 参数 | 功能 |
|---|---|
| `execution.serial.command_hz` | Piper 目标发送频率；当前 60 Hz，50 点约 0.83 秒 |
| `robot.state_publish_hz` | 实测关节与夹爪状态的发布频率；当前 50 Hz |
| `robot.motion_speed` | SDK 运动速度比例，范围 1–100；当前 80 |
| `trajectory_fit.jerk_penalty_weight` | 抑制关节轨迹的急剧变化；当前 0.20 |
| `trajectory_fit.terminal_preserve_points` | 末段根据运动方向减少拟合改动的点数；当前 10 |
| `trajectory_fit.jitter_range_deg` | J1–J6 五点窗口内的小幅抖动范围 |
| `trajectory_fit.jitter_net_ratio` | 判断反复变化时，首尾净变化与总变化量的比例上限 |
| `trajectory_fit.max_joint_step_deg` | 每个关节相邻发送目标的最大角度差；当前 1.75° |

拟合与抖动处理只作用于 J1–J6。每步角度限制的第一个参考是当前实测位置，后续参考上一条发送目标。完整配置和 ROS2 接口见[运行说明](docs/runtime.md)。

RTC（执行期间重新预测）实现保留在源码中，可通过 `execution.mode: rtc` 选择，需要配套的动作 OM 和梯度分片。当前串行 OM 含专用去噪输出，切换模式时也应切换对应模型文件。

## 五、工程结构

| 目录 | 内容 |
|---|---|
| `config/` | 运行配置、归一化统计与模型资源路径 |
| `inference/core/` | 昇腾执行、观测处理、轨迹处理、RTC 和运行记录 |
| `inference/ros2/` | Prompt、双摄、推理与机械臂四个节点 |
| `inference/ros2_interfaces/` | RTC 消息定义 |
| `quantization/` | PyTorch 转换、ONNX 适配、OM 编译和算子实现 |
| `include/` | 项目公共路径 |
| `third_party/` | 配套 OpenPI 和 ACLLite 源码及许可证 |
| `docs/` | 环境、运行和量化说明 |
| `outputs/` | 本地构建与运行产物，不提交 Git |

## 六、许可证

本工程采用 [Apache-2.0](LICENSE)。OpenPI、Gemma 和 ACLLite 的许可及修改说明见[第三方组件](THIRD_PARTY.md)。模型和分词器按对应资源的许可证使用。
