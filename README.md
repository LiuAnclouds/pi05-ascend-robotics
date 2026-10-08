<div align="center">

# π0.5 · Ascend Robotics

**OpenPI π0.5 + Ascend310P1 + Piper deployment**

![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)
![CANN](https://img.shields.io/badge/CANN-8.5.0-D71920)
![SoC](https://img.shields.io/badge/Ascend-310P1-DB2626)
![Precision](https://img.shields.io/badge/Precision-FP16%20%2B%20FP32-0E8A16)
![License](https://img.shields.io/badge/License-Apache--2.0-blue)

</div>

这是官方 OpenPI π0.5 模型在香橙派昇腾平台上的工程化部署项目，包含 ONNX 导出、CANN ATC 编译、OM 推理、双相机采集和 Piper 控制。统一 Conda 环境为 **`Pi05_ascend`**。

## 快速启动

平台要求：Linux aarch64、Ascend310P1、CANN 8.5.0、Python 3.10。先安装匹配的昇腾驱动、CANN Toolkit/算子包与 Miniconda；安装脚本只管理 Python 环境。仓库不包含训练权重、tokenizer、OM 和原始数据；这些文件必须按项目结构放置。

```bash
git clone https://github.com/LiuAnclouds/pi05-ascend-robotics.git
cd pi05-ascend-robotics
bash setup_env.sh
source include/environment.sh
python runtime/activate_can.py
```

准备以下部署资产（`model.safetensors` 仅导出时需要）：

```text
models/weights/instrction_9.14_float32/model.safetensors
models/paligemma-3b-pt-224/tokenizer.model
config/norm_stats.json
outputs/om/1.om
outputs/om/2.om
```

`config/norm_stats.json` 是本项目训练数据的统计，换权重需同步核对统计和动作约定。当前六关节为相对观测状态的增量，夹爪为绝对开度。

不发送动作的影子推理：

```bash
bash run_inference.sh \
  --task "Pick up the blue cylindrical box and place it in the white square basket."
```

确认设备、工作区和急停状态后，开启真实动作：

```bash
bash run_inference.sh \
  --task "Pick up the blue cylindrical box and place it in the white square basket." \
  --send-motion
```

`--task` 必填；默认相机为 `/dev/video0`（第三视角）和 `/dev/video2`（腕部），CAN 为 `can0`，持续运行，按 `Ctrl+C` 快速停止并保存报告。程序先准备相机/CAN/机械臂，再加载模型、预热，最后推理；运行中发生急停、失能或故障会停止下发，不会自动恢复。

常用参数：`--steps 10`（去噪次数）、`--action-fps 30`（轨迹频率）、`--motion-speed 15`（Piper 速度百分比）、`--iterations 0`（持续运行）。每轮保持完整 **50 点**轨迹。USB 重插后需确认视角映射；物理急停需手动解除。程序不会自动判断任务完成。

当前为“推理 → 下发完整轨迹”的顺序流程，30 Hz 的 50 点约需 1.67 秒，尚未实现 RTC/异步轨迹融合。相机独立持续采集，只保留最新帧。

## 导出和编译

权重应为训练完成的 OpenPI PyTorch checkpoint，其他路径可用 `--weights` 指定。准备真实样本 `data/sample.npz`：`image`（第三视角）和 `wrist_image`（腕部）为 HWC uint8 RGB 图像，`state[7]` 为六关节角度（度）和夹爪开度（毫米），`prompt` 为字符串：

```bash
source include/environment.sh
python export/prepare_input.py --sample data/sample.npz
python export/prepare_part2_input.py
python export/export_part1.py
python export/export_part2.py
python export/compile_om.py --part 1
python export/compile_om.py --part 2
```

Part1 使用显式 Gemma 层图、opset 17；Part2 使用官方 action-expert 图、opset 14。导出保持模型的浮点策略：FP16 为主，Softmax/归一化/位置编码等敏感计算保留 FP32；mask、索引按算子要求使用 BOOL/INT32/INT64。ATC 使用 `Ascend310P1`、静态 ND 输入、`precision_mode_v2=origin`，Part1 选择 `high_performance_for_all`。这不是 INT8 量化，也不是全图 FP32。

导出在 CPU 上运行，需有足够内存和磁盘加载完整权重。Part1 ONNX 与同目录外部权重必须一起保留。已有 OM 不会被 `compile_om.py` 覆盖，请使用新的 `--output` 生成候选。历史 Part1 编译仍有 CumSum 未命中高优先级库的提示，不宣称所有算子已最优。

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
    1/        1.onnx、外部权重、1.export.json
    2/        2.onnx、2.export.json
  om/         1.om、2.om，编译日志与验证结果
  runs/
    Infer_report_<时间>/    result.json、result.log
```

导出记录为 `<编号>.export.json`，编译记录为 `<编号>.compile.json/.log`，精度验证结果为 `<编号>.validation.json`。这些记录在执行对应步骤时生成。Part1 原 `manual/` 目录已统一为 `onnx/1/`；只保留一个正式 Part1 导出入口。

`openpi/` 和 `runtime/acllite/` 的第三方来源及许可证见 [`THIRD_PARTY.md`](THIRD_PARTY.md)。CANN 默认 `/usr/local/Ascend/cann-8.5.0`，Conda 默认 `$HOME/miniconda3`，分别可用 `PI05_CANN_ROOT`、`PI05_CONDA_ROOT` 覆盖。入口均支持 `--help`。

## 许可证

项目代码采用 Apache-2.0；OpenPI、Ascend ACLLite、Piper SDK 和模型权重遵循各自许可证。
