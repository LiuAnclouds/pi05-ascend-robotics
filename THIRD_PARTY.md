# 第三方组件

| 组件 | 用途 | 许可证 |
|---|---|---|
| [OpenPI](https://github.com/Physical-Intelligence/openpi) | pi0.5 模型定义与导出 | [Apache-2.0](third_party/openpi/LICENSE)；[Gemma 许可](third_party/openpi/LICENSE_GEMMA.txt) |
| ACLLite | 昇腾执行工具与参考代码 | [Apache-2.0](third_party/acllite/LICENSE) |
| Piper SDK | CAN 通信与机械臂接口 | 使用安装包附带许可证 |

OpenPI 基于提交 `215abfb217dbac7d5f1273282331b9b1866c0479`，保留昇腾注意力掩码、CumSum、位置编码和 Python 3.10 时区适配。`setup_env.sh` 将配套 Gemma、PaliGemma、SigLIP 实现安装到导出环境的 Transformers 4.53.3 中。

模型权重、分词器及数据集作为独立资源提供，按各自许可证使用。
