# Fate: Multi-Task Learning Framework for Pathological Analysis

This repository contains a PyTorch-based framework for multi-modal, multi-task learning, specifically designed for pathological analysis. It leverages features from images (e.g., Whole-Slide Images) and text (e.g., pathology reports) to simultaneously predict multiple clinical endpoints. The framework is built with a modular architecture, incorporating advanced techniques like Mixture-of-Experts (MoE) and Pareto-optimal training strategies.

---

## English Version

### Key Features

*   **Multi-Task Learning**: Simultaneously trains a single model on multiple pathology-related tasks (e.g., predicting cancer presence, lymph node status, vascular thrombus, and perineural invasion).
*   **Multi-Modal Fusion**: Fuses features from different modalities (image and text) using a sophisticated architecture involving cross-attention and a dedicated fusion module.
*   **Advanced Model Architecture**:
    *   **Fate**: A cross-attention mechanism to learn representations between different modalities.
    *   **Mixture-of-Experts (MoE)**: Utilizes task-specific experts (`DeepseekMoE`) to allow for specialized processing for each task, improving model capacity and performance.
    *   **Per-Task Fusion**: Employs a `ProposedFusionModule` for each task to intelligently combine multi-modal information.
*   **Pareto-Optimal Training**: Integrates the `libmtl` library to support advanced multi-task optimization strategies, including Multiple Gradient Descent Algorithm (MGDA), which finds a Pareto-optimal solution by balancing the gradients of different tasks.
*   **Flexible Configuration**: Easily configure training hyperparameters, model architecture details, and loss weighting strategies (`sum`, `dwa`, `mgda`) through command-line arguments.
*   **Cross-Validation Support**: Includes a built-in training loop for k-fold cross-validation to ensure robust model evaluation.

### Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd Fate
    ```

2.  **Install dependencies:**
    It is recommended to use a virtual environment.
    ```bash
    pip install -r requirements.txt
    ```

### Usage

The main entry point for training is `main.py`.

1.  **Prepare your data:**
    *   Place your feature files (e.g., in H5 format) in a data directory.
    *   Prepare a CSV file containing the labels for your tasks.
    *   Organize your data splits (train/validation) into fold directories as expected by the script.

2.  **Start training:**
    You can run the training with custom parameters using the command line. Here is an example:
    ```bash
    python main.py \
        --csv_label_path /path/to/your/labels.csv \
        --h5_data_dir /path/to/your/features \
        --folds_dir /path/to/your/folds \
        --loss_weighting_strategy mgda \
        --learning_rate 1e-4 \
        --batch_size 1 \
        --num_epochs 50 \
        --wandb_project your_project_name \
        --wandb_run_name your_run_name
    ```
    The script will run the cross-validation loop, train the model, log results using `wandb`, and save the best model for each fold.

---

## 中文版本

### Fate: 用于病理分析的多任务学习框架

本仓库是一个基于 PyTorch 的多模态、多任务学习框架，专为病理分析设计。它利用来自图像（如全切片图像）和文本（如病理报告）的特征来同时预测多个临床终点。该框架采用模块化架构，集成了混合专家（MoE）和帕累托最优训练策略等先进技术。

### 主要特性

*   **多任务学习**: 在多个病理相关任务上（例如，预测癌症、淋巴结状态、血管血栓和神经侵犯）同时训练单个模型。
*   **多模态融合**: 使用复杂的架构（包括交叉注意力和专门的融合模块）来融合来自不同模态（图像和文本）的特征。
*   **先进模型架构**:
    *   **Fate**: 一种交叉注意力机制，用于学习不同模态之间的交互表示。
    *   **混合专家模型 (MoE)**: 利用任务特异性专家 (`DeepseekMoE`) 对每个任务进行专门处理，从而提高模型容量和性能。
    *   **任务级融合**: 每个任务都使用一个 `ProposedFusionModule` 模块来智能地结合多模态信息。
*   **帕累托最优训练**: 集成了 `libmtl` 库以支持先进的多任务优化策略，特别是多梯度下降算法（MGDA），该算法通过平衡不同任务的梯度来寻找帕累托最优解。
*   **灵活配置**: 通过命令行参数轻松配置训练超参数、模型架构细节和损失加权策略（支持 `sum`, `dwa`, `mgda`）。
*   **交叉验证支持**: 内置 k 折交叉验证训练循环，以确保稳健的模型评估。

### 安装与设置

1.  **克隆仓库:**
    ```bash
    git clone <repository-url>
    cd Fate
    ```

2.  **安装依赖:**
    建议使用虚拟环境。
    ```bash
    pip install -r requirements.txt
    ```

### 使用方法

训练的主要入口点是 `main.py`。

1.  **准备数据:**
    *   将您的特征文件（例如 H5 格式）放置在数据目录中。
    *   准备一个包含任务标签的 CSV 文件。
    *   按照脚本的预期，将您的数据划分（训练/验证集）组织到不同的折叠目录中。

2.  **开始训练:**
    您可以使用命令行运行训练并指定自定义参数。这是一个示例：
    ```bash
    python main.py \
        --csv_label_path /path/to/your/labels.csv \
        --h5_data_dir /path/to/your/features \
        --folds_dir /path/to/your/folds \
        --loss_weighting_strategy mgda \
        --learning_rate 1e-4 \
        --batch_size 1 \
        --num_epochs 50 \
        --wandb_project your_project_name \
        --wandb_run_name your_run_name
    ```
    脚本将运行交叉验证循环，训练模型，使用 `wandb` 记录结果，并为每一折保存最佳模型。