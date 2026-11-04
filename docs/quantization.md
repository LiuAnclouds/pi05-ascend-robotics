# 量化方案与性能分析

## 一、精度方案

| 方案 | 视觉线性层 | 语言线性层 | 动作专家及投影线性层 |
|---|---|---|---|
| FULL-W8A8，全模块八位 | W8A8 | W8A8 | W8A8 |
| ACT-W16A16，动作专家十六位 | W8A8 | W8A8 | W16A16 |

权重按输出通道进行八位量化；激活在运行时计算缩放系数，由 `DynamicQuant`（动态量化算子）生成 INT8 输入，再通过 `QuantBatchMatMul`（量化矩阵乘法算子）完成线性运算。动作专家的 W16A16 路径使用原始浮点权重转换得到的 FP16 权重。

注意力投影属于线性层。注意力内部矩阵计算及归一化等运算沿用对应的浮点精度。

### 1. 310P 推理耗时

双摄 224 × 224，64 个文本词元，50 个动作点，十步去噪；预热 3 次、计时 30 次。表中为模型推理耗时，包含输入输出传输和去噪更新。

| 方案 | 视觉与语言 | 动作专家十步 | 完整推理 |
|---|---:|---:|---:|
| FULL-W8A8 | 106.116 ms | 88.010 ms | 197.444 ms |
| **ACT-W16A16** | **106.112 ms** | **80.545 ms** | **189.967 ms** |

动作专家采用 FP16 后，动作阶段耗时减少 **8.48%**，完整推理耗时减少 **3.79%**。动态量化需要计算缩放系数、转换输入并恢复输出；动作序列较短，这些开销又在十步去噪中重复出现，因此本次 310P 动作图采用 FP16 更快。

当前串行实现进一步将 FP16 去噪更新放在 NPU 上：十步复用设备动作状态，在同一 ACL 执行流中提交，最后统一等待完成。表中基准采用 CPU 浮点更新路径，两者的精度分配均为 ACT-W16A16。

### 2. Orin 与 310P

| 配置 | Ascend 310P1 | Jetson AGX Orin 64 GB | 310P 耗时减少 |
|---|---:|---:|---:|
| FULL-W8A8 | 197.444 ms | 565.917 ms | 65.11% |
| ACT-W16A16 | 189.967 ms | 604.595 ms | 68.58% |
| 各平台最快配置 | **189.967 ms** | **565.917 ms** | **66.43%** |

Orin 使用 MAXN 模式与 JAX 整数矩阵运算，310P 使用 CANN 原生量化算子和离线编译图。按两端各自最快的实测配置，310P 推理速度约为 Orin 的 **2.98 倍**。

## 二、量化覆盖率与输出精度

### 1. 权重与激活占比

| 统计范围 | FULL-W8A8 | ACT-W16A16 |
|---|---:|---:|
| W8 权重占全部模型参数 | 75.33% | **63.44%** |
| W8 权重占全部线性层权重 | 100.00% | **84.22%** |
| 视觉线性层的 W8 占比 | 100.00% | 100.00% |
| 语言线性层的 W8 占比 | 100.00% | 100.00% |
| 动作专家线性层的 W8 占比 | 100.00% | 0.00% |

ACT-W16A16 的矩阵输入激活中，**62.34%** 为 A8，37.66% 为 FP16。激活比例按一次视觉语言图和十次动作图的输入元素数量统计。

### 2. 相对浮点参考的误差

ACT-W16A16 在 144 组观测上验证，每个输出轴统计 7,200 个动作点，参考为相同输入与噪声下的 FP16 输出。整段输出的平均余弦相似度为 **99.7938%**。

| 输出 | 平均绝对偏差 | P95 | 最大偏差 |
|---|---:|---:|---:|
| J1 | 0.144° | 0.530° | 2.625° |
| J2 | 0.456° | 1.930° | 12.224° |
| J3 | 0.332° | 1.401° | 9.417° |
| J4 | 0.179° | 0.559° | 3.303° |
| J5 | 0.287° | 1.051° | 11.729° |
| J6 | 0.257° | 0.610° | 3.797° |
| 夹爪 | **1.214 mm** | **2.051 mm** | 70.936 mm |

量化输出整体接近浮点参考，少数夹爪目标存在较大局部偏差。P95 表示 95% 的预测点偏差不超过该值。

保持视觉语言图不变、只切换动作专家精度时，两种量化方案的平均余弦相似度为 **99.9967%**，夹爪平均绝对差为 **0.230 mm**。结合耗时结果，310P 采用 ACT-W16A16。

## 三、量化方法先验实验

RTX 3090 实验以原始 BF16 模型为参考，比较视觉、语言和动作专家的单模块、两两组合和三模块量化，并进一步比较各模块中的注意力投影与前馈网络。

视觉与语言是主要误差来源，动作专家对八位量化较不敏感。由此保留 FULL-W8A8 与 ACT-W16A16 两种端侧候选配置。

| 误差优化方法 | 平均相对动作误差 | P95 相对动作误差 |
|---|---:|---:|
| 原始 W8A8 | 3.50% | 9.50% |
| QuantVLA 校准 | 3.34% | 9.88% |
| 正交变换 | **2.31%** | **7.10%** |
| QuantVLA 与正交变换组合 | 2.36% | 7.27% |

QuantVLA 使用注意力温度匹配与输出幅度校准；正交处理使用输入和权重配对的 Hadamard（哈达玛）变换。组合处理改善了 94.44% 验证观测的相对误差。二者选为后续误差优化方法，端侧性能表对应本仓库已验证的逐通道权重与动态激活量化实现。

## 四、模型导出

在量化导出环境中准备源权重 `config/models/weights/model.safetensors`。观测 NPZ 包含：

| 字段 | 内容 |
|---|---|
| `image` | 前视图像，HWC、uint8、RGB |
| `wrist_image` | 腕部图像，HWC、uint8、RGB |
| `state` | 六关节角度与夹爪毫米值，共 7 维 |
| `prompt` | 标量任务字符串 |

公开导出入口生成三路图像槽位、200 个文本词元的通用静态图，第三路图像由掩码关闭。性能表采用双摄、64 个文本词元的专用部署图；输入形状随模型资源固定。

### 1. 准备导出输入

```bash
python -m quantization.pytorch.prepare_input \
  --sample data/sample.npz --output data/part1.pt
python -m quantization.pytorch.prepare_part2_input \
  --weights config/models/weights \
  --part1-input data/part1.pt --output data/part2.pt
```

### 2. 导出 ACT-W16A16

```bash
python -m quantization.export_onnx \
  --part 1 --weights config/models/weights \
  --input data/part1.pt --output outputs/onnx/part1.onnx \
  --precision w8a8 --action-precision fp16
python -m quantization.export_onnx \
  --part 2 --weights config/models/weights \
  --input data/part2.pt --output outputs/onnx/part2.onnx \
  --precision w8a8 --action-precision fp16 --serial-euler-state
```

第二条命令同时生成带 FP16 更新输出的 `part2_serial_euler.onnx`，供当前串行执行使用。导出 FULL-W8A8 时，将动作精度改为 `--action-precision w8a8`。

### 3. 编译 OM

```bash
python -m quantization.export_om --part 1 \
  --input outputs/onnx/part1.onnx \
  --output outputs/om/part1.om --soc Ascend310P1
python -m quantization.export_om --part 2 \
  --input outputs/onnx/part2_serial_euler.onnx \
  --output outputs/om/part2.om --soc Ascend310P1
```

编译完成后，将配套的 `part1.om` 和 `part2.om` 放到 `config/models/om/`，或在配置中填写其绝对路径。ONNX 的外部权重文件需要与图文件保存在一起。
