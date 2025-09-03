# LLM Alignment

大语言模型（LLM）对齐项目，集成了监督微调（Supervised Fine-Tuning, SFT）、群体相对策略优化（Group Relative Policy Optimization, GRPO）和直接偏好优化（Direct Preference Optimization, DPO）三种方法。

## 简介 (Introduction)

本项目实现了包括SFT、GRPO和DPO在内的多种主流对齐算法，并提供了可复现的代码、训练脚本和评估流程。

## 主要特性 (Features)

* **SFT**：高效的监督微调实现。
* **GRPO**：对Group Relative Policy Optimization算法的完整实现。
* **DPO**：对Direct Preference Optimization算法的完整实现。
* **统一接口**：所有对齐方法都遵循相似的训练和推理流程，易于使用。
* **评估脚本**：提供脚本用于评估模型性能。

## 开始使用 (Getting Started)

### 环境要求 (Prerequisites)

* Python 3.x
* PyTorch [版本号]
* CUDA [版本号] (如果需要GPU支持)

### 安装 (Installation)

1.  克隆项目仓库：
    ```bash
    git clone [https://github.com/](https://github.com/)[你的GitHub用户名]/[你的项目名称].git
    cd [你的项目名称]
    ```

2.  安装依赖库：
    ```bash
    pip install -r requirements.txt
    ```

    * **待完善**: 请在这里列出`requirements.txt`中的主要库，例如`transformers`, `datasets`, `accelerate`等。

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