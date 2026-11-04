# 环境配置

## 一、环境组成

| 环境 | 用途 | Python | 主要依赖 |
|---|---|---|---|
| `pi05_humble` | 双摄、ROS2、OM 推理和 Piper 控制 | 3.11 | ROS2 Humble、NumPy 1.26.4、pyACL、Piper SDK |
| `Pi05_ascend` | 模型精度转换、ONNX 导出和 OM 编译 | 3.10 | PyTorch 2.5.1、Transformers 4.53.3、ONNX、CANN |

板端系统为 openEuler 22.03 LTS-SP3，架构为 ARM64，CANN 安装位于 `/usr/local/Ascend/cann-8.5.0`。通用环境入口为 `/usr/local/Ascend/ascend-toolkit/latest/set_env.sh`。

## 二、创建推理环境

安装适用于 Linux ARM64 的 Conda，先按设备厂商说明安装 Ascend 驱动、固件和匹配的 CANN。创建项目环境：

```bash
conda env create -f environment.yml
conda activate pi05_humble
python -m pip install --no-deps -e .
```

`environment.yml` 使用 RoboStack Humble 与 conda-forge 软件源，包含 ROS2 基础包、接口生成器、构建工具和当前推理依赖。

每次使用时加载：

```bash
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
conda activate pi05_humble
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
```

将 Conda 的 `lib` 放在前面，以便 ROS2 使用该环境配套的 C++ 运行库。板端已通过环境激活脚本配置这一步。

### 已验证版本

| 组件 | 版本 |
|---|---|
| Python | 3.11.17 |
| ROS2 `rclpy` | 3.3.16 |
| NumPy | 1.26.4 |
| OpenCV Headless | 4.11.0.86 |
| Pillow | 12.3.0 |
| SentencePiece | 0.2.2 |
| PyYAML | 6.0.3 |
| Piper SDK | 0.6.2 |
| python-can | 4.6.1 |
| setuptools | 69.5.1 |
| EmPy | 3.3.4 |
| colcon-core | 0.19.0 |

ROS2 消息生成使用 EmPy 3.3.4，Python 构建使用 setuptools 69.5.1；环境文件固定这两项版本。

## 三、构建 ROS2 包

在工程根目录执行：

```bash
colcon --log-base outputs/ros2/log build \
  --build-base outputs/ros2/build \
  --install-base outputs/ros2/install \
  --base-paths inference/ros2 inference/ros2_interfaces \
  --symlink-install
source outputs/ros2/install/setup.bash
```

两个包分别为 `pi05_interfaces` 与 `pi05_inference`。前者生成 RTC 消息，后者安装四个节点和主启动文件。

每个新终端都要执行 `source outputs/ros2/install/setup.bash`。主启动文件会导入 `pi05_inference.config`；出现 `No module named 'pi05_inference'` 时，先检查当前环境和这条加载命令。

## 四、创建量化导出环境

安装 Conda 后，在项目根目录执行：

```bash
PI05_CONDA_ENV=Pi05_ascend bash setup_env.sh
conda activate Pi05_ascend
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
```

安装脚本读取 `quantization/requirements.txt`，安装本工程，并应用配套的 Transformers 模型实现。源 OpenPI 代码保存在 `third_party/openpi/`。

| 组件 | 版本 |
|---|---|
| Python | 3.10.21 |
| PyTorch | 2.5.1 |
| Transformers | 4.53.3 |
| NumPy | 1.26.4 |
| ONNX | 1.23.1 |
| SciPy | 1.15.3 |
| safetensors | 0.8.0 |

ONNX 导出与 OM 编译入口见[量化说明](quantization.md#四模型导出)。

## 五、模型与设备

| 配置 | 内容 |
|---|---|
| `model.part1_om` | 双摄视觉语言图 |
| `model.part2_om` | 与执行模式匹配的动作图 |
| `model.tokenizer` | 配套分词器 |
| `model.stats` | 配套状态和动作归一化统计 |
| `camera.front_device` | 前视摄像头设备 |
| `camera.wrist_device` | 腕部摄像头设备 |
| `robot.can_interface` | Piper CAN 接口 |

配置中的相对路径以项目根目录为基准。OM、分词器与归一化统计需要来自同一部署模型。当前双摄图在预处理后接收两路 224 × 224 图像，文本最多使用 64 个有效词元。

确认 NPU 与 ROS2 环境：

```bash
npu-smi info
python -c "import acl, rclpy, numpy, cv2; print(numpy.__version__, cv2.__version__)"
ros2 pkg prefix pi05_inference
```
