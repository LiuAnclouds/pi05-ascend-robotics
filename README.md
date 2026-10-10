# π0.5 · Ascend Robotics

![Orange Pi](https://img.shields.io/badge/Orange_Pi-AI_Station-F58220)
![Ascend](https://img.shields.io/badge/Ascend-310P1-D71920)
![CANN](https://img.shields.io/badge/CANN-8.5.0-D71920)
![Docker](https://img.shields.io/badge/Docker-ARM64-2496ED?logo=docker&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.10-3776AB)
![License](https://img.shields.io/badge/License-Apache_2.0-green)

将 [OpenPI π0.5](https://github.com/Physical-Intelligence/openpi) 部署到 Orange Pi AI Station，通过双路相机和语言指令控制 Piper 机械臂。支持 PyTorch → ONNX → OM 导出，编译与推理共用一个 Docker 环境，无需 Conda。

**适用设备：** Ascend310P1 / ARM64、openEuler 22.03 LTS-SP3、Piper + USB-CAN、两路 USB 相机。模型须使用对应 Piper 数据微调。

## 1. 安装环境

以下命令均在板端以 root 执行。已有环境可跳过对应安装步骤。

### 驱动与 CANN（宿主机）

从 [Orange Pi 官方支持页](http://www.orangepi.org/html/hardWare/computerAndMicrocontrollers/service-and-support/Orange-Pi-AI-Station.html) 安装板卡配套系统、驱动及固件，按手册重启。已验证驱动版本为 `7.6.T7.0.B056`。执行 `npu-smi info`，确认设备为 310P1、Health 为 OK。

从 [昇腾下载页](https://www.hiascend.com/developer/download/community/result?module=cann) 下载 **8.5.0 / AArch64 的 Toolkit 和 310P 算子包**，在安装包目录执行：

```bash
bash ./Ascend-cann-toolkit_8.5.0_linux-aarch64.run --install --install-path=/usr/local/Ascend
source /usr/local/Ascend/cann-8.5.0/set_env.sh
bash ./Ascend-cann-310p-ops_8.5.0_linux-aarch64.run --install
atc --help
npu-smi info
```

系统依赖和驱动配套要求见 [CANN 8.5 安装指南](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/softwareinst/instg/instg_0000.html)。Docker 复用宿主机驱动与 CANN，不替代这一步。

### 构建项目镜像

```bash
dnf install -y git docker-engine python3
systemctl enable --now docker
git clone https://github.com/LiuAnclouds/pi05-ascend-robotics.git
cd pi05-ascend-robotics
bash setup_docker.sh
```

看到 `Image ready` 即构建完成。镜像安装 Python 3.10、PyTorch、ONNX/ORT、Piper SDK 和编译依赖，版本见 [requirements.txt](requirements.txt)。首次构建需联网；更新代码后重新执行构建命令。

若 Docker Hub 无法访问，可使用可达的基础镜像源：

```bash
PI05_BASE_IMAGE=docker.m.daocloud.io/library/python:3.10-slim-bookworm bash setup_docker.sh
```

## 2. 准备模型

**模型下载仓库尚未发布，目前需自行准备下面四个配套文件。仅克隆代码不能开始推理。**

| 文件 | 放置位置（相对项目根目录） |
| --- | --- |
| Part1 OM | `outputs/om/part1.om` |
| Part2 OM | `outputs/om/part2.om` |
| Tokenizer | `models/paligemma-3b-pt-224/tokenizer.model` |
| 训练归一化参数 | `config/norm_stats.json` |

已有 `norm_stats.json` 只适用于配套模型，不要与其他权重混用。模型发布后可用 `bash run_docker.sh python export/download_models.py --repo OWNER/REPO` 下载完整包，`OWNER/REPO` 替换为发布地址。

## 3. 启动 Piper

连接相机和 USB-CAN，给机械臂上电、解除物理急停，保持工作区域无遮挡。以下命令均在项目根目录执行。

**终端一：初始化 CAN，启动推理。**

```bash
bash run_docker.sh python runtime/activate_can.py
bash run_docker.sh \
  --task "Pick up the blue cylindrical box and place it in the white square basket." \
  --send-motion --action-fps 20 --motion-speed 10
```

CAN 默认 `can0`、1 Mbps。启动依次完成设备准备、模型加载、一次丢弃结果的预热，然后显示 `Waiting for W`。

**终端二：进入同一个项目目录，确认开始。**

```bash
python3 runtime/set_prompt.py
```

输入 `W` 并按 Enter 开始执行。换任务时，在终端二输入新 prompt 并回车；当前 50 点执行完后暂停，等待再次输入 `W`。无需重新加载模型或 ROS。

**停止：在终端一按 Ctrl+C，快速停止并保存报告。** 终端二的 Ctrl+C 只关闭输入程序。程序持续运行，不自动判断任务完成；运行中急停或故障会停止下发。

| 参数 | 默认值 | 用途 |
| --- | --- | --- |
| `--task` | 必填 | 任务指令，无默认 prompt |
| `--send-motion` | 关闭 | 开启动作下发；省略则只推理 |
| `--camera-a` / `--camera-b` | `/dev/video0` / `/dev/video2` | 第三视角 / 腕部视角，重插后须核对编号 |
| `--can` | `can0` | CAN 接口 |
| `--steps` | `10` | Part2 去噪次数 |
| `--action-fps` / `--motion-speed` | `50` / `30` | 下发频率 / Piper 速度；上方示例显式使用 `20` / `10` |
| `--iterations` | `0` | 持续运行；正整数限制推理轮数 |

完整参数：`bash run_docker.sh --help`。每轮执行完整 **50 点**；六关节按增量还原，夹爪按绝对开度处理。关节默认保留 EMA、反向死区和相邻点限幅。

## 4. 自行导出模型

已有 OM 可跳过本节。将配套 OpenPI π0.5 PyTorch 权重放到 `models/weights/instrction_9.14_float32/model.safetensors`，准备真实样本 `data/sample.npz`：

| 字段 | 格式 |
| --- | --- |
| `image` / `wrist_image` | 第三视角 / 腕部图像，HWC、uint8、RGB |
| `state` | 六关节角度（度）+ 夹爪开度（毫米），共 7 维 |
| `prompt` | 该样本的任务文本 |

```bash
bash run_docker.sh python export/prepare_input.py --sample data/sample.npz
bash run_docker.sh python export/prepare_part2_input.py
bash run_docker.sh python export/export_part1.py
bash run_docker.sh python export/export_part2.py
bash run_docker.sh python export/compile_om.py --part 1
bash run_docker.sh python export/compile_om.py --part 2
```

生成 `outputs/onnx/part1/part1.onnx`、`outputs/onnx/part2/part2.onnx` 及 `outputs/om/part1.om`、`outputs/om/part2.om`。ONNX 的同目录外部权重必须一起保留。已有 OM 不覆盖，另用 `--output` 指定新文件。CPU 导出需预留完整模型的内存和磁盘空间。

| 导出策略 | 配置 |
| --- | --- |
| Part1 / Part2 | 视觉语言编码及 KV cache / Action Expert 去噪；opset 17 / 14 |
| 精度 | FP16 主计算，Softmax、归一化等敏感路径保留 FP32 |
| 算子适配 | causal mask 累积使用 INT32、position IDs 累积保留 INT64；显式 attention 和旋转位置编码适配 |
| ATC | `Ascend310P1`、静态 ND、`precision_mode_v2=origin`；Part1 优先高性能实现 |

## 性能与精度

Ascend310P1 / CANN 8.5.0，10 次去噪，每次输出 50 个动作点。

| 指标 | 香橙派 | NVIDIA Orin |
| --- | ---: | --- |
| 部署精度 | FP16 + 部分 FP32 | 待测试 |
| 模型平均耗时（固定真实输入，30 次） | 358.08 ms | 待测试 |
| 实机模型平均耗时（571 个完整块） | 389.91 ms | 待测试 |
| 每 50 点整轮耗时（25 Hz / speed 10） | 2427.31 ms | 待测试 |
| ONNX → OM 最终动作 Cosine / RMSE | 0.999999894 / 0.00028993 | 待测试 |
| 加载及预热后的设备 DDR 增量 | 6.508 GiB | 待测试 |

精度来自单个真实样本；模型耗时不含运动，整轮包含 50 点下发。DDR 为设备全局增量，非进程独占显存；以上指标不代表抓取成功率。

## 文件与报告

```text
export/     模型下载、输入准备、ONNX 导出、OM 编译
runtime/    实时推理、相机、Piper 控制、prompt 输入、报告
include/    路径、常量、环境入口
openpi/     上游模型代码
config/     归一化参数和性能记录
models/     权重与 tokenizer（不提交 Git）
data/       原始样本与输入张量（不提交 Git）
outputs/    onnx/、om/、runs/（不提交 Git）
tests/      回归测试
```

每次推理生成 `outputs/runs/Infer_report_<时间>/result.json`（完整动作、关节反馈、相机时间戳、耗时）和 `result.log`（SDK 诊断）。运行中先写 `result.jsonl`，退出汇总为 JSON；不录制视频。

## 常见问题

| 提示 | 处理 |
| --- | --- |
| `CAN port ... is not UP` | 重新执行 `runtime/activate_can.py` 的 Docker 命令 |
| `Waiting for W` | 终端二输入 `W` 并回车；切换任务后同样需要确认 |
| 找不到相机 / 视角反了 | 检查 `/dev/video*`，通过相机参数指定实际设备 |
| `JOINT_COMMUNICATION_ERR` | 检查供电与 CAN 接线；通信故障不能靠切换模式修复 |
| 缺少模型文件 | 按第 2 节放入配套四个文件后重启 |

## 许可证

项目采用 Apache-2.0；OpenPI、ACLLite、Piper SDK 及模型权重遵循各自许可证，见 [THIRD_PARTY.md](THIRD_PARTY.md)。
