<div align="center">

# π0.5 · Ascend Robotics

**OpenPI π0.5 + Ascend310P1 + Piper deployment**

![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-ARM64-2496ED?logo=docker&logoColor=white)
![CANN](https://img.shields.io/badge/CANN-8.5.0-D71920)
![SoC](https://img.shields.io/badge/Ascend-310P1-DB2626)
![Precision](https://img.shields.io/badge/Precision-FP16%20%2B%20FP32-0E8A16)
![License](https://img.shields.io/badge/License-Apache--2.0-blue)

</div>

这是官方 OpenPI π0.5 模型在香橙派昇腾平台上的部署项目，包含 ONNX 导出、ATC 编译、OM 推理、双相机采集和 Piper 控制。推荐按下面的 **Docker 教程**部署：在板端本地构建镜像，宿主机无需安装 Python 环境或 Conda。暂不提供公共预构建镜像。

## 1. 准备香橙派和 Docker

已验证平台：**ARM64 / Ascend310P1 / openEuler 22.03 LTS-SP3 / CANN 8.5.0**，Piper 机械臂、USB-CAN、两路 V4L2 相机。先按板卡厂商说明准备匹配的系统、昇腾驱动、CANN Toolkit 和 310P 算子包；本项目不会安装内核驱动。其他芯片或 CANN 版本需要重新编译、验证 OM。

下面命令均在香橙派上以 root 执行；普通账户先使用 `sudo -i`。Docker 已安装时可跳过安装命令：

```bash
dnf install -y git docker-engine
systemctl enable --now docker
docker version
npu-smi info
test -f /usr/local/Ascend/cann-8.5.0/set_env.sh
```

`docker version` 应显示 Client 和 Server，`npu-smi info` 应能看到昇腾设备。已有驱动/CANN 的系统可直接继续。以下教程不要求重新安装正常工作的驱动。

## 2. 克隆项目并构建镜像

```bash
git clone https://github.com/LiuAnclouds/pi05-ascend-robotics.git
cd pi05-ascend-robotics
bash setup_docker.sh
```

脚本生成 `pi05-ascend-robotics:cann8.5`，首次构建需要联网下载基础镜像和 Python 依赖，完成后显示 `Image ready`。已验证镜像约 **1.88 GB**；请另外预留模型、构建缓存和导出文件空间。镜像统一支持导出、编译和推理。

无法连接 Docker Hub 时，可指定基础镜像代理重试，不需修改 Docker 系统配置：

```bash
PI05_BASE_IMAGE=docker.m.daocloud.io/library/python:3.10-slim-bookworm bash setup_docker.sh
```

也可直接使用 `docker build -t pi05-ascend-robotics:cann8.5 .`。代码更新后重新执行构建脚本；代码随镜像发布，权重和结果保存在镜像外。

## 3. 准备模型

**Hugging Face 模型仓库尚未发布。** 当前需手动准备以下文件；仓库发布后，可使用下方下载命令。只有代码和 Docker 镜像不足以启动推理。

```text
models/paligemma-3b-pt-224/tokenizer.model
config/norm_stats.json
outputs/om/part1.om
outputs/om/part2.om
```

tokenizer、归一化统计和两个 OM 必须来自同一个模型发布版本。当前六关节输出为相对观测状态的增量，夹爪为绝对开度。仅推理不需要训练 checkpoint；自行导出时另准备 `models/weights/instrction_9.14_float32/model.safetensors`。

Hugging Face 发布时沿用上面的相对路径。下载命令中的 `HF_OWNER/HF_MODEL` 是占位符，需换成届时公布的仓库名；下载期间停止推理：

```bash
docker run --rm -it --entrypoint python \
  --mount "type=bind,src=$PWD,dst=/app" \
  pi05-ascend-robotics:cann8.5 \
  export/download_models.py --repo HF_OWNER/HF_MODEL
```

加 `--weights` 同时下载导出用 checkpoint，`--revision` 可指定模型版本。下载器固定同一次下载的仓库提交，全部下载成功后替换对应文件。此下载命令无需挂载 NPU，也无需宿主机 Python。

## 4. 连接设备并启动

机械臂上电，连接 CAN 和两路相机，确认 `/dev/video0` 为第三视角、`/dev/video2` 为腕部视角。USB 重插后编号可能变化，可用 `--camera-a`、`--camera-b` 指定。

```bash
bash run_docker.sh python runtime/activate_can.py

bash run_docker.sh \
  --task "Pick up the blue cylindrical box and place it in the white square basket." \
  --send-motion
```

`--task` 必填。去掉 `--send-motion` 仅推理、不下发动作；加上它会启动机械臂控制，需确认工作区域和急停状态。默认 CAN 为 `can0`，去噪 **10 步**，每轮 **50 个动作点**，下发 **30 Hz**，机械臂速度 **15**。参数可分别用 `--steps`、`--action-fps`、`--motion-speed` 修改；`--iterations 0` 持续运行，程序不会自动判断抓取任务是否完成。

按 **Ctrl+C** 快速停止并保存报告。程序先准备设备，再加载模型、预热、推理；运行中急停、失能或故障会停止下发，物理急停需手动解除。相机持续采集最新帧；当前推理与完整轨迹下发依次执行，尚未实现 RTC/异步融合。

启动时若处于软件急停，程序先恢复并确认正常失能，再使能六关节、确认 CAN/MOVE_J 控制模式；各阶段按反馈确认，不用固定延时猜测恢复完成。若无故障但持续待机，只在启动阶段做一次失能/使能重试，过程会短暂卸载电机力矩。持续故障会报错退出。

所有输出保存在宿主机项目 `outputs/`，容器退出不会丢失。启动脚本自动只读挂载 CANN/驱动和模型、映射昇腾设备及相机、使用主机网络访问 CAN，不使用 `--privileged`。Docker 不替代宿主机的驱动和 CANN 安装。`PI05_IMAGE` 可覆盖镜像标签，`PI05_CANN_ROOT` 可覆盖宿主机 CANN 目录。

板端验证：真实 Piper 样本完整 10 步推理 **468.25 ms**，动作与宿主机逐值一致；小模型 PyTorch→ONNX→ATC→OM、两路相机/CAN 读取及 Ctrl+C 保存报告通过。此容器验证未下发机械臂动作，也未重新编译完整 π0.5 模型。

## 导出和编译

权重应为训练完成的 OpenPI PyTorch checkpoint，其他路径可用 `--weights` 指定。准备真实样本 `data/sample.npz`：`image`（第三视角）和 `wrist_image`（腕部）为 HWC uint8 RGB 图像，`state[7]` 为六关节角度（度）和夹爪开度（毫米），`prompt` 为字符串：

```bash
bash run_docker.sh python export/prepare_input.py --sample data/sample.npz
bash run_docker.sh python export/prepare_part2_input.py
bash run_docker.sh python export/export_part1.py
bash run_docker.sh python export/export_part2.py
bash run_docker.sh python export/compile_om.py --part 1
bash run_docker.sh python export/compile_om.py --part 2
```

Part1 使用显式 Gemma 层图、opset 17；Part2 使用官方 action-expert 图、opset 14。导出保持模型的浮点策略：FP16 为主，Softmax/归一化/位置编码等敏感计算保留 FP32；mask、索引按算子要求使用 BOOL/INT32/INT64。ATC 使用 `Ascend310P1`、静态 ND 输入、`precision_mode_v2=origin`，Part1 选择 `high_performance_for_all`。这不是 INT8 量化，也不是全图 FP32。

导出在 CPU 上运行，需有足够内存和磁盘加载完整权重。Part1 ONNX 与同目录外部权重必须一起保留。已有 OM 不会被 `compile_om.py` 覆盖，请使用新的 `--output` 生成候选。历史 Part1 编译仍有 CumSum 未命中高优先级库的提示，不宣称所有算子已最优。

<details>
<summary>可选：已有 Conda 环境的本机部署</summary>

不使用 Docker 时，先安装 Miniconda，再执行：

```bash
bash setup_env.sh
source include/environment.sh
python runtime/activate_can.py
bash run_inference.sh --task "Pick up the blue cylindrical box and place it in the white square basket." --send-motion
```

统一环境名为 `Pi05_ascend`；导出时直接执行相同 Python 入口。两种方式使用同一套模型和精度策略。

</details>

## 实测结果

以下数据来自真实 Piper 记录样本和 Ascend310P1/CANN 8.5：

| 指标 | 结果 | 口径 |
| --- | ---: | --- |
| Part1 OM | **259.96 ms** | 预热后 20 次平均，P95 260.60 ms |
| Part2 OM | **21.86 ms/步** | 10 次单步测量 |
| 完整 10 步推理 | **467.80 ms** | Part1 + 10 次 Part2，不含相机和运动 |
| 完整动作 ONNX↔OM cosine | **0.999999894** | 相同输入、tokens、seed，最终 `[50,7]` 归一化动作 |
| 完整动作 RMSE | **0.00028993** | 同一比较 |
| Part1 reference↔ONNX cosine | **0.999968244** | KV cache，不是最终动作 |
| Part1 reference↔OM cosine | **0.999992992** | KV cache，不是最终动作 |
| Part2 FP16 reference↔OM cosine | **0.999999838** | 单步 velocity 输出 |

这些是单样本数值一致性和耗时指标，不代表真实抓取成功率。完整动作基准是适配后的 ONNX，不能据此证明原始 JAX 训练处理链完全一致；KV 指标包含全张量。数据来源见 [`config/benchmark.json`](config/benchmark.json)。

每次推理单独保存在 `outputs/runs/Infer_report_<时间>/`：`result.json` 保存完整动作、关节反馈、相机时间戳及分模块耗时，`result.log` 保存异常和 SDK 输出。运行中的 `result.jsonl` 实时落盘，最终报告保存成功后自动移除；突然断电时保留它用于排查。不录制相机视频。指定 `--output` 可自选报告位置，已有运行记录不会覆盖。

## 目录

```text
export/       数据准备、ONNX 导出、OM 编译
runtime/      相机、Piper、OM 推理和报告
include/      公共路径、定义和环境加载
config/       归一化统计与基准指标
openpi/       已验证 OpenPI 源码快照
models/       本地权重/tokenizer（不入 Git）
data/         本地样本和处理张量（不入 Git）
outputs/      ONNX/OM/运行报告（不入 Git）
```

所有生成结果只分三类，相关记录跟随对应文件保存：

```text
outputs/
  onnx/
    part1/    part1.onnx、外部权重、part1.export.json
    part2/    part2.onnx、part2.export.json
  om/         part1.om、part2.om，编译日志与验证结果
  runs/
    Infer_report_<时间>/    result.json、result.log
```

导出记录为 `part<N>.export.json`，编译记录为 `part<N>.compile.json/.log`，精度验证结果为 `part<N>.validation.json`。这些记录在执行对应步骤时生成。ONNX 和 OM 统一使用 `part1`、`part2` 命名；只保留一个正式 Part1 导出入口。

`openpi/` 和 `runtime/acllite/` 的第三方来源及许可证见 [`THIRD_PARTY.md`](THIRD_PARTY.md)。CANN 默认 `/usr/local/Ascend/cann-8.5.0`，Conda 默认 `$HOME/miniconda3`，分别可用 `PI05_CANN_ROOT`、`PI05_CONDA_ROOT` 覆盖。入口均支持 `--help`。

## 许可证

项目代码采用 Apache-2.0；OpenPI、Ascend ACLLite、Piper SDK 和模型权重遵循各自许可证。
