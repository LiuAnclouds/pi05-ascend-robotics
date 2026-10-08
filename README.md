# pi05-ascend-robotics

![Orange Pi](https://img.shields.io/badge/Orange_Pi-AI_Station-F58220)
![Ascend](https://img.shields.io/badge/Ascend-310P1-D71920)
![CANN](https://img.shields.io/badge/CANN-8.5.0-D71920)
![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-ARM64-2496ED?logo=docker&logoColor=white)
![Precision](https://img.shields.io/badge/Precision-FP16%20%2B%20FP32-0E8A16)

`pi05-ascend-robotics` 是 OpenPI π0.5 在 Orange Pi AI Station 上的模型部署项目。推理程序读取双路相机、机械臂状态和任务指令，生成 Piper 动作轨迹；导出程序将微调后的 PyTorch 权重转换为 ONNX 和昇腾 OM。两种用途共用一个 Docker 环境。

## 算法简介

[π0.5](https://github.com/Physical-Intelligence/openpi) 是视觉语言动作模型（VLA），根据图像、语言指令和机器人状态预测动作序列。本项目适配使用 Piper 数据微调的 OpenPI π0.5 模型。

| 项目 | 内容 |
| --- | --- |
| 图像输入 | 第三视角、腕部视角，预处理为 `224 × 224` |
| 状态输入 | 六个关节角度和夹爪开度 |
| 任务输入 | 通过 `--task` 传入自然语言指令 |
| 动作输出 | 每轮 `50 × 7`：50 个轨迹点，每点六关节目标和夹爪开度 |

任务指令应与模型训练的任务对应，例如将蓝色罐子放入白色篮子：

```text
Pick up the blue cylindrical box and place it in the white square basket.
```

## 推理性能

以下为 Ascend310P1 / CANN 8.5.0 上真实 Piper 样本的测试结果。

| 模块 | 耗时 | 测量方式 |
| --- | ---: | --- |
| Part1 | **229.7 ms** | 设备内存 KV cache，10 次平均 |
| Part2 | **12.21 ms / 步** | 设备内存 KV cache，10 步平均 |
| 完整推理 | **354.2 ms** | Part1 + 10 步 Part2，10 次平均 |

| 比较对象 | 余弦相似度 | RMSE |
| --- | ---: | ---: |
| ONNX / OM 最终动作 | **0.999999894** | **0.00028993** |

耗时不含相机采集和轨迹执行。KV cache 仅在 NPU 设备内存中复用，不改变计算图或动作数；与旧主机拷贝路径相比，最终动作逐元素一致。精度比较同一输入下最终 `50 × 7` 归一化动作，不代表抓取成功率。[完整指标](config/benchmark.json)

## 模型与精度

推理流程为 `双路图像 + 状态 + Prompt → 预处理 → Part1 → Part2 去噪 → 反归一化 → Piper 轨迹`。

| 模块 | 作用 | 部署配置 |
| --- | --- | --- |
| Part1 | 视觉与语言编码，生成 KV cache | 每轮执行 1 次；ONNX opset 17 |
| Part2 | Action Expert 预测 velocity，迭代生成动作 | 默认去噪 10 次；ONNX opset 14 |
| 浮点精度 | 保留敏感计算精度 | FP16 为主；Softmax、归一化等使用 FP32 |
| ATC 编译 | 将 ONNX 编译为 OM | `Ascend310P1`、静态 ND 输入、`precision_mode_v2=origin` |
| 动作处理 | 将模型输出转换为控制目标 | 六关节增量加观测状态，夹爪按绝对开度处理 |

本项目采用 FP16 / FP32 混合精度部署。每轮执行完整的 50 点轨迹，然后采集新观测进行下一轮推理。

## 开发环境

| 项目 | 配置 |
| --- | --- |
| 硬件 | Orange Pi AI Station，AArch64，Ascend310P1 |
| 系统 | openEuler 22.03 LTS-SP3 |
| 昇腾环境 | 板卡配套驱动、CANN Toolkit 8.5.0、310P 算子包 8.5.0 |
| 运行环境 | Docker，Python 3.10 |
| 模型工具 | PyTorch、Transformers、ONNX、ONNX Runtime、ATC |
| 外设 | Piper 机械臂、USB-CAN、两路 V4L2 相机 |

驱动与 CANN 安装在宿主机；项目 Python 依赖由 Dockerfile 安装，无需另外配置 Conda。

## 准备工作

以下操作在香橙派上以 root 执行。先准备宿主机环境，再构建镜像、放置模型。

### 安装驱动与 CANN

已有配套驱动、CANN Toolkit 8.5.0 和 310P 算子包时可跳过安装。

<details>
<summary>首次安装：驱动、CANN 与验证命令</summary>

从 [Orange Pi AI Station 官方支持页](http://www.orangepi.org/html/hardWare/computerAndMicrocontrollers/service-and-support/Orange-Pi-AI-Station.html) 获取板卡系统、驱动和固件，按厂商手册配套安装。本项目使用驱动 `7.6.T7.0.B056`（`npu-smi` 软件包版本 `26.0.t1`）。

以下命令仅适用于对应的 `opiaistation` 板卡专用包：

```bash
bash ./Ascend-hdk-310p-npu-driver_7.6.t7.0.b056_linux-aarch64_chip-enable-opiaistation-260825.run --full
```

按安装器和板卡手册完成固件配套、重启。`npu-smi info` 应显示 310P1 且 Health 为 OK。

在 [昇腾官方下载页](https://www.hiascend.com/developer/download/community/result?module=cann) 选择 **8.5.0 / AArch64 / run**，下载 Toolkit 和 **310P** 算子包。系统依赖、驱动配套和升级说明见 [CANN 8.5.0 安装指南](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/softwareinst/instg/instg_0000.html)。首次安装时，在安装包目录依次执行：

```bash
bash ./Ascend-cann-toolkit_8.5.0_linux-aarch64.run \
  --install --install-path=/usr/local/Ascend
source /usr/local/Ascend/cann-8.5.0/set_env.sh
bash ./Ascend-cann-310p-ops_8.5.0_linux-aarch64.run --install
```

`--install-path` 指定父目录，安装器生成 `/usr/local/Ascend/cann-8.5.0`。Toolkit 和算子包须同为 8.5.0，算子包须选择 310P。

验证安装：

```bash
source /usr/local/Ascend/cann-8.5.0/set_env.sh
cat /usr/local/Ascend/cann-8.5.0/aarch64-linux/ascend_toolkit_install.info
cat /usr/local/Ascend/cann-8.5.0/aarch64-linux/ascend_ops_install.info
atc --help
npu-smi info
```

两份安装信息应分别为 `Ascend-cann-toolkit`、`Ascend-cann-310p-ops`，版本均为 `8.5.0`。自定义安装目录通过 `PI05_CANN_ROOT` 指定，目录内须有 `set_env.sh`。

</details>

### 构建 Docker 镜像

安装 Git 和 Docker，并启动服务。已完成的步骤可跳过：

```bash
dnf install -y git docker-engine
systemctl enable --now docker
docker version
```

克隆项目并构建镜像：

```bash
git clone https://github.com/LiuAnclouds/pi05-ascend-robotics.git
cd pi05-ascend-robotics
bash setup_docker.sh
```

构建完成后显示 `Image ready`，镜像名为 `pi05-ascend-robotics:cann8.5`，大小约 1.88 GB。首次构建需联网，并预留模型和构建缓存空间。代码更新后重新执行 `bash setup_docker.sh`。

<details>
<summary>Docker Hub 无法访问时</summary>

通过基础镜像代理构建：

```bash
PI05_BASE_IMAGE=docker.m.daocloud.io/library/python:3.10-slim-bookworm bash setup_docker.sh
```

</details>

### 准备模型

**Hugging Face 模型仓库待发布。** 当前需自行准备配套文件，并按下面位置放入项目：

| 文件 | 项目内路径 |
| --- | --- |
| Part1 OM | `outputs/om/part1.om` |
| Part2 OM | `outputs/om/part2.om` |
| Tokenizer | `models/paligemma-3b-pt-224/tokenizer.model` |
| 归一化参数 | `config/norm_stats.json` |

四个文件须与所部署的模型匹配。仅推理不需要 PyTorch 权重；自行导出见 [模型导出](#模型导出)。

## 实机推理

### 连接设备

机械臂上电，连接 USB-CAN 和两路相机。默认设备对应关系如下：

| 设备 | 默认接口 | 参数 |
| --- | --- | --- |
| 第三视角相机 | `/dev/video0` | `--camera-a` |
| 腕部相机 | `/dev/video2` | `--camera-b` |
| Piper CAN | `can0` | `--can` |

USB 重插后相机编号可能变化，请按实际视角设置。下列命令均在项目根目录执行。

初始化 CAN，默认 `can0`、1 Mbps：

```bash
bash run_docker.sh python runtime/activate_can.py
```

该命令配置通信接口，不会使能或移动机械臂。输出 `can0 is UP` 后即可启动推理。

### 启动机械臂控制

确认物理急停已解除、工作区域安全后执行：

```bash
bash run_docker.sh \
  --task "Pick up the blue cylindrical box and place it in the white square basket." \
  --send-motion
```

`--task` 为必填任务指令；`--send-motion` 表示将预测轨迹发送给机械臂。仅需查看推理结果时，省略 `--send-motion`。

终端依次显示以下启动阶段：

```text
1/4 | Device setup
2/4 | Model loading
3/4 | Warm-up
4/4 | Inference running
```

设备准备阶段完成相机连接和机械臂使能。软件急停恢复会先短暂失能，再使能进入 CAN/MOVE_J；预热不下发动作。进入 `Inference running` 后，每轮显示推理耗时、已发送点数及执行状态。

默认每轮预测并下发 **50 个动作点**，程序持续运行。**按 Ctrl+C 快速停止并保存报告**；程序不会自动判断抓取任务完成。运行中急停或故障会停止下发，不自动恢复。

### 运行参数

参数可追加到启动命令后；完整说明使用 `bash run_docker.sh --help`。

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--steps` | `10` | 每轮 Part2 去噪次数 |
| `--action-fps` | `30` | 轨迹点下发频率，Hz |
| `--motion-speed` | `15` | Piper 运动速度参数，与下发频率独立 |
| `--iterations` | `0` | 0 表示持续运行；正整数指定推理轮数 |
| `--output` | 按时间自动命名 | JSON 报告路径 |

## 模型导出

本节用于将自己的 OpenPI π0.5 PyTorch 权重导出为 ONNX 和 OM。已有 OM 的用户可直接使用上面的实机推理命令。

### 准备权重与样本

默认权重文件放在：

```text
models/weights/instrction_9.14_float32/model.safetensors
```

准备该模型配套的 tokenizer、`config/norm_stats.json`，以及真实样本 `data/sample.npz`：

| 字段 | 格式 |
| --- | --- |
| `image` | 第三视角图像，HWC、uint8、RGB |
| `wrist_image` | 腕部图像，HWC、uint8、RGB |
| `state` | 7 维：六关节角度（度）与夹爪开度（毫米） |
| `prompt` | 该样本的任务文本 |

转换为导出输入：

```bash
bash run_docker.sh python export/prepare_input.py --sample data/sample.npz
bash run_docker.sh python export/prepare_part2_input.py
```

生成 `data/part1.pt` 和 `data/part2.pt`。

### 导出 ONNX

```bash
bash run_docker.sh python export/export_part1.py
bash run_docker.sh python export/export_part2.py
```

输出为 `outputs/onnx/part1/part1.onnx` 和 `outputs/onnx/part2/part2.onnx`。ONNX 与同目录外部权重须一起保留。

### 编译 OM

```bash
bash run_docker.sh python export/compile_om.py --part 1
bash run_docker.sh python export/compile_om.py --part 2
```

输出为 `outputs/om/part1.om` 和 `outputs/om/part2.om`，可用于实机推理。已有 OM 不会被覆盖，可用 `--output` 指定新文件。

输入准备和 ONNX 导出在 CPU 上执行，需留足完整模型的内存与磁盘空间。使用其他权重目录时，在 `prepare_part2_input.py` 和两个 ONNX 导出命令中传入 `--weights`。各入口均支持 `--help`。

## 输出结果

推理报告保存在宿主机项目目录，容器退出后保留：

```text
outputs/runs/Infer_report_<时间>/
├── result.json
└── result.log
```

| 文件 | 内容 |
| --- | --- |
| `result.json` | 完整动作、关节反馈、相机时间戳和各模块耗时 |
| `result.log` | SDK 输出和异常诊断 |

运行期间由后台写入 `result.jsonl`，正常停止时写完并汇总为 `result.json`；意外退出时保留日志，但突然断电可能丢失尚未写盘的少量记录。程序不录制相机视频。

模型输出统一保存在：

```text
outputs/
├── onnx/
│   ├── part1/    # part1.onnx、外部权重与导出记录
│   └── part2/    # part2.onnx、外部权重与导出记录
├── om/          # part1.om、part2.om 与编译记录
└── runs/        # 每次推理的独立报告
```

## 许可证

项目代码采用 Apache-2.0。OpenPI、Ascend ACLLite、Piper SDK 和模型权重遵循各自许可证，来源见 [THIRD_PARTY.md](THIRD_PARTY.md)。
