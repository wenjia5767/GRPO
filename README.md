# LLM Alignment

大语言模型（LLM）对齐项目，集成了监督微调（Supervised Fine-Tuning, SFT）、群体相对策略优化（Group Relative Policy Optimization, GRPO）和直接偏好优化（Direct Preference Optimization, DPO）三种方法。

## 简介 (Introduction)

本项目实现了包括SFT、GRPO和DPO在内的多种主流对齐算法，并提供了代码、训练脚本和评估流程。

### 主要内容

* **SFT**：高效的监督微调实现。
* **GRPO**：对Group Relative Policy Optimization算法的完整实现。
* **DPO**：对Direct Preference Optimization算法的完整实现。
* **评估脚本**：提供脚本用于评估模型性能。

### 硬件与实验环境

本项目的所有实验均在以下配置的服务器上运行：

* **CPU**: Intel(R) Xeon(R) w9-3595X (60 核 / 120 线程)
* **GPU**: 2 x NVIDIA RTX PRO 6000 (每张显存 96 GB)
* **CUDA 版本**: 12.9
* **NVIDIA 驱动版本**: 575.57.08


### 环境要求 (Prerequisites)

* **Python** 3.12
* **PyTorch** 2.7.1

### 安装 (Installation)

1.  克隆项目仓库：
    ```bash
    git clone https://github.com/wenjia5767/GRPO.git
    cd GRPO
    ```

2.  安装依赖库：
    ```bash
    pip install -r requirements.txt
    ```

-----

### 1. Qwen-2.5-Math-1.5B模型在GSM8K数据集上的Zero-Shot评测

##### 评测 `Qwen-2.5-Math-1.5B` 模型在 **GSM8K** 数据集上的零样本数学推理能力。
-----

### 🎯 方法

  * **模型**: `Qwen-2.5-Math-1.5B`
  * **数据集**: GSM8K (共 1319 个样本用于评估)
  * **任务**: 零样本数学推理 (Zero-shot Mathematical Reasoning)
  * **提示工程 (Prompting)**: 数据集中的每个问题都通过 **`r1_zero` 提示模板**进行格式化。该模板要求模型在 `<think>` 标签内生成其推理过程，并在 `<answer>` 标签内生成最终的数值答案。
  * **推理**: 使用 `vllm` 库进行高效的模型推理生成。
  * **评估**: 使用 `r1_zero_reward_fn` 函数来解析模型生成的文本，并将提取出的答案与标准答案进行比较打分。

-----

### 🚀 运行

1.  **环境配置**: 确保已安装所需的 Python 库，主要包括 `vllm`, `datasets`, 和 `transformers`。
2.  **路径配置**: 在运行脚本前，请根据实际情况，修改模型和本地数据集缓存的硬编码路径。
3.  **执行脚本**: 在终端中运行脚本：
    ```bash
    python alignment/gsm8k_baseline.py
    ```

结果在以下文件中呈现：`gsm8k_eval_results.jsonl` (包含每个样本的详细结果) 和 `gsm8k_eval_summary.json` (包含整体的性能指标)。

-----

### 📊 评测结果

在 GSM8K 测试集的 1319 个样本上，模型的基线性能 (baseline performance) 评测结果如下：

```json
{
  "num_examples": 1319,
  "format_rate": 0.025018953752843062,
  "accuracy": 0.0037907505686125853,
  "avg_reward": 0.0037907505686125853,
  "results_path": "gsm8k_eval_results.jsonl"
}
```

  * 模型遵循 `<think>`/`<answer>` 格式的能力较差，导致**格式正确率 (format rate) 极低，仅约 2.5%**。
  * 最终答案的**准确率 (accuracy) 也非常低，仅约 0.38%**，表明该模型在 GSM8K 数据集的零样本设置下面临巨大挑战。

1.  **指令遵循能力缺失**

      * 评测结果的首要问题是模型无法稳定遵循 `<think>`/`<answer>` 这种严格的 XML 风格的输出格式，导致**格式正确率 (format rate) 仅有约 2.5%**。
      * 这表明，`Qwen-2.5-Math-1.5B` 模型虽然可能具备一定的数学知识，但在零样本（Zero-shot）场景下，其指令遵循 (Instruction Following) 能力不足以应对这种复杂的、结构化的输出要求。模型没有被充分地微调来理解并执行这种特定的格式指令。

2.  **准确率是格式错误的直接后果**

      * **准确率 (accuracy) 极低（约 0.38%）**，但这并非完全代表模型的数学推理能力不行。更准确地说，这是格式错误的**下游效应**。
      * `r1_zero_reward_fn` 评估函数必须先成功解析出 `<answer>` 标签里的内容，然后才能判断答案是否正确。既然只有 2.5% 的输出格式正确，那么理论上的最高准确率也已经被限制在了 2.5%。这揭示了一个关键问题：**模型表达答案的能力，成为了评估其推理能力的严重障碍。**

3.  **对零样本评测方法评价**

      * **结论**: 这次基线评测证明，对于 `1.5B` 参数规模的模型，在这种需要复杂推理和严格格式控制的任务上，简单的零样本提示（Prompting）策略是远远不够的。
      * **下一步**:
          * **监督式微调 (Supervised Fine-Tuning)**: 最根本的提升方法是在一批遵循此格式的问答数据上对模型进行微调，从而直接向模型“教授”这种结构化输出的技能。


## 使用指南 (Usage)

### 1. 准备数据 (Data Preparation)

请将你的数据集组织成特定格式。例如：

* **SFT数据**：包含`prompt`和`response`的JSONL文件。
* **GRPO/DPO数据**：包含`prompt`、`chosen_response`和`rejected_response`的JSONL文件。

* **待完善**: 详细说明你的数据格式，可以提供一个简单的示例。

### 2. 监督微调 (Supervised Fine-Tuning)

**待完善**: 详细说明如何运行SFT。
* **训练脚本示例**: 提供一个完整的命令行示例，包括参数（如模型路径、数据路径、输出目录等）。
* **重要参数说明**: 简要解释关键参数的作用，如`--model_name_or_path`、`--data_path`、`--per_device_train_batch_size`。

### 3. 群体相对策略优化 (GRPO)

**待完善**: 详细说明如何运行GRPO。
* **训练脚本示例**: 提供GRPO的命令行示例。
* **重要参数说明**: 简要解释关键参数，例如如何配置奖励模型（reward model）或偏好数据。

### 4. 直接偏好优化 (DPO)

**待完善**: 详细说明如何运行DPO。
* **训练脚本示例**: 提供DPO的命令行示例。
* **重要参数说明**: 解释DPO特有的参数。

## 项目结构 (Project Structure)