<div align="center">

# π0.5 · Ascend Robotics

OpenPI π0.5 模型导出与 Piper 机械臂部署

![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-ARM64-2496ED?logo=docker&logoColor=white)
![CANN](https://img.shields.io/badge/CANN-8.5.0-D71920)
![SoC](https://img.shields.io/badge/Ascend-310P1-DB2626)
![License](https://img.shields.io/badge/License-Apache--2.0-blue)

[快速部署](#快速部署) · [模型导出](#模型导出) · [文件与日志](#文件与日志) · [驱动与 CANN 安装](#驱动与-cann-安装)

</div>

基于官方 OpenPI，支持 ONNX 导出、ATC 编译和 OM 实机推理。使用一个 Docker 镜像管理运行环境，集成双相机采集与 Piper 机械臂控制。

**适用平台：** Orange Pi AI Station · ARM64 · Ascend310P1 · openEuler 22.03 LTS-SP3 · CANN 8.5.0。

## 性能

| 指标 | 实测结果 |
| --- | ---: |
| Part1 推理 | **259.96 ms**（预热后 20 次平均） |
| Part2 单步推理 | **21.86 ms**（10 次单步平均） |
| 完整推理（10 步去噪） | **467.80 ms**（单次测量） |
| ONNX / OM 动作余弦相似度 | **0.999999894** |
| ONNX / OM 动作 RMSE | **0.00028993** |

使用真实 Piper 样本测量。各项延迟独立统计，不含相机采集和机械臂运动；精度比较同输入下的最终归一化动作，不代表抓取成功率。[查看基准数据](config/benchmark.json)

## 快速部署

以下命令在香橙派宿主机以 root 执行。宿主机需已安装匹配的驱动、CANN 8.5.0 Toolkit 和 310P 算子包，首次装机请先完成 [驱动与 CANN 安装](#驱动与-cann-安装)。

### 1. 构建 Docker 镜像

安装 Docker 和 Git（已安装可跳过）：

```bash
dnf install -y git docker-engine
systemctl enable --now docker
```

克隆项目并构建：

```bash
git clone https://github.com/LiuAnclouds/pi05-ascend-robotics.git
cd pi05-ascend-robotics
bash setup_docker.sh
```

镜像名为 `pi05-ascend-robotics:cann8.5`，约 **1.88 GB**。首次构建需联网；更新代码后重新执行构建脚本。当前提供 Dockerfile，由用户本地构建。

Docker Hub 无法访问时，可改用基础镜像代理：

```bash
PI05_BASE_IMAGE=docker.m.daocloud.io/library/python:3.10-slim-bookworm bash setup_docker.sh
```

### 2. 准备模型

**模型下载入口待发布。** 当前请将配套的模型文件放到以下位置：

```text
models/paligemma-3b-pt-224/tokenizer.model
config/norm_stats.json
outputs/om/part1.om
outputs/om/part2.om
```

这四个文件须来自同一模型版本。仅运行 OM 不需要训练 checkpoint，也无需配置 Conda。

### 3. 启动推理

连接机械臂、CAN 和两路相机，确认默认相机对应关系：

| 参数 | 默认设备 | 视角 |
| --- | --- | --- |
| `--camera-a` | `/dev/video0` | 第三视角 |
| `--camera-b` | `/dev/video2` | 腕部视角 |

USB 重插后编号可能变化。确认机械臂上电、物理急停解除且工作区域安全，在项目根目录执行：

```bash
bash run_docker.sh python runtime/activate_can.py

bash run_docker.sh \
  --task "Pick up the blue cylindrical box and place it in the white square basket." \
  --send-motion
```

`--task` 必填，填写任务指令；`--send-motion` 开启机械臂控制，省略则只推理、不下发动作。

默认每轮 **10 步去噪、50 个动作点、30 Hz 下发、速度 15**。程序持续运行，不自动判断任务完成；**按 Ctrl+C 快速停止并保存报告。**

<details>
<summary>修改运行参数</summary>

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--can` | `can0` | CAN 接口 |
| `--steps` | `10` | 每轮去噪次数 |
| `--action-fps` | `30` | 轨迹点下发频率，Hz |
| `--motion-speed` | `15` | Piper 运动速度参数 |
| `--iterations` | `0` | 0 表示持续运行，正整数指定推理轮数 |
| `--output` | 自动按时间命名 | JSON 报告路径 |

参数直接追加到启动命令后。完整说明：`bash run_docker.sh --help`。

启动前自动完成设备准备；软件急停恢复会短暂失能，再使能进入控制模式。运行中触发急停或出现故障会停止下发，不自动恢复。

</details>

## 模型导出

已有 OM 可跳过本节。自行导出时，准备 OpenPI PyTorch 权重：

```text
models/weights/instrction_9.14_float32/model.safetensors
```

并准备真实样本 `data/sample.npz`：

| 字段 | 格式 |
| --- | --- |
| `image` / `wrist_image` | 第三视角 / 腕部图像，HWC、uint8、RGB |
| `state` | 7 维：六关节角度（度）＋夹爪开度（毫米） |
| `prompt` | 样本对应的任务文本 |

在项目根目录依次执行：

```bash
# 准备输入
bash run_docker.sh python export/prepare_input.py --sample data/sample.npz
bash run_docker.sh python export/prepare_part2_input.py

# 导出 ONNX
bash run_docker.sh python export/export_part1.py
bash run_docker.sh python export/export_part2.py

# 编译 OM
bash run_docker.sh python export/compile_om.py --part 1
bash run_docker.sh python export/compile_om.py --part 2
```

采用 **FP16 / FP32 混合精度**：FP16 为主，Softmax、归一化等敏感计算保留 FP32；ATC 使用 `origin` 保持图中精度。Part1 / Part2 分别使用 ONNX opset 17 / 14。

导出在 CPU 上进行，需预留完整模型的内存与磁盘空间。可用 `--weights` 指定其他权重目录；已有 OM 不会被覆盖，用 `--output` 指定新文件。各入口均支持 `--help`。

## 文件与日志

```text
export/       输入准备、ONNX 导出、OM 编译
runtime/      实机推理、相机与机械臂控制
include/      公共定义与环境加载
config/       归一化统计与基准数据
openpi/       官方源码
models/       权重与 tokenizer
data/         原始样本与输入张量
outputs/
  onnx/       part1/part1.onnx、part2/part2.onnx 及外部权重
  om/         part1.om、part2.om
  runs/       Infer_report_<时间>/
```

每次运行单独保存 `result.json`（动作结果、关节状态、相机时间戳、耗时）和 `result.log`（运行诊断）。中断未完成的报告保留 `result.jsonl` 记录；不录制相机视频。

模型、数据和输出保存在宿主机，不入 Git，容器退出不会丢失。导出与编译日志跟随对应模型保存；**ONNX 文件与其外部权重须一起保留。**

## 驱动与 CANN 安装

宿主机负责驱动、固件和 CANN，Docker 启动脚本自动挂载使用。已有正常环境无需重复安装。

<details>
<summary>首次装机：下载、安装与验证</summary>

**① 板卡系统与驱动**

从 [Orange Pi AI Station 官方支持页](http://www.orangepi.org/html/hardWare/computerAndMicrocontrollers/service-and-support/Orange-Pi-AI-Station.html) 获取对应板型的系统、驱动和固件，按厂商手册配套安装。

本项目使用驱动 `7.6.T7.0.B056`（`npu-smi` 显示包版本 `26.0.t1`）。以下命令仅适用于对应的板卡专用安装包：

```bash
bash ./Ascend-hdk-310p-npu-driver_7.6.t7.0.b056_linux-aarch64_chip-enable-opiaistation-260825.run --full
```

按安装器提示和板卡手册完成固件配套、重启；`npu-smi info` 应显示 310P1 且 Health 为 OK。不要用通用服务器驱动替换板卡专用驱动。

**② CANN Toolkit 与算子包**

在 [昇腾官方下载页](https://www.hiascend.com/developer/download/community/result?module=cann) 选择 **8.5.0 / AArch64 / run**，下载 Toolkit 与 **310P** 算子包。系统依赖和版本配套见 [CANN 8.5.0 安装指南](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/softwareinst/instg/instg_0000.html)。在安装包目录执行：

```bash
bash ./Ascend-cann-toolkit_8.5.0_linux-aarch64.run \
  --install --install-path=/usr/local/Ascend
source /usr/local/Ascend/cann-8.5.0/set_env.sh
bash ./Ascend-cann-310p-ops_8.5.0_linux-aarch64.run --install
```

安装参数指定父目录，版本目录自动生成为 `/usr/local/Ascend/cann-8.5.0`。不要混装其他版本或 A3、310B、910B 算子包；已有环境升级请遵循官方指南。

**③ 验证安装**

```bash
source /usr/local/Ascend/cann-8.5.0/set_env.sh
cat /usr/local/Ascend/cann-8.5.0/aarch64-linux/ascend_toolkit_install.info
cat /usr/local/Ascend/cann-8.5.0/aarch64-linux/ascend_ops_install.info
atc --help
npu-smi info
```

确认 Toolkit 与 310P 算子包版本均为 `8.5.0`。若使用自定义安装目录，用 `PI05_CANN_ROOT` 指定包含 `set_env.sh` 的目录。其他芯片或 CANN 版本需重新编译验证 OM。

完成后返回 [快速部署](#快速部署)，构建项目镜像。

</details>

---

项目代码采用 **Apache-2.0**。第三方代码和模型遵循各自许可，见 [THIRD_PARTY.md](THIRD_PARTY.md)。
