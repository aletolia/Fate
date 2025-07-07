import dataclasses
import os
import re
import random
import math
from typing import List, Dict, Optional, Any, Tuple

import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Sampler, Subset, Dataset
from sklearn.metrics import roc_auc_score, f1_score
from tqdm.autonotebook import tqdm
import wandb
from Pytorch_PCGrad.pcgrad import PCGrad

# Configurations and constants
@dataclasses.dataclass
class TrainingFlags:
    # Multi-Task Options
    loss_weighting_strategy: str = "dwa"
    use_pcgrad: bool = True
    use_diversity_reg: bool = False
    diversity_beta: float = 1e-4

    # if use task norm, it will normalize the task-specific features
    use_task_norm: bool = False
    use_sampler: bool = False

    # Parameters
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    batch_size: int = 1
    num_epochs: int = 30
    seed: int = 42

    # Early Stopping & DWA
    early_stopping_patience: int = 10
    early_stopping_delta: float = 1e-4
    dwa_temperature: float = 2.0

    folds_dir: str = "folds"
    output_dir: str = "./training_output"
    wandb_project: str = "multitask_moe_refactored"
    wandb_run_name: Optional[str] = None

TASK_CONFIG = {
    "names": ["cancer", "lymph", "vascular", "perineural"],
    "label_keys": ["label", "lymph_node_label", "vascular_thrombus", "perineural_invasion"],
    "criteria": {
        "cancer": nn.BCEWithLogitsLoss(),
        "lymph": nn.BCEWithLogitsLoss(),
        "vascular": nn.BCEWithLogitsLoss(),
        "perineural": nn.BCEWithLogitsLoss(),
    }
}
NUM_TASKS = len(TASK_CONFIG["names"])

# Data Sampler
class ImbalancedDatasetSampler(Sampler):
    def __init__(self, dataset: Dataset, indices: List[int], label_key: str = "label"):
        
        self.indices = indices
        self.num_samples = len(indices)
        first_item = dataset[indices[0]]
        label_freq = {}
        
        for i in indices:
            label = int(dataset[i][label_key])
            label_freq[label] = label_freq.get(label, 0) + 1 

        weights = []
        for i in indices:
            label = int(dataset[i][label_key])
            freq = label_freq.get(label)
            weights.append(1.0 / freq)
                
        self.weights = torch.DoubleTensor(weights)

    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples

# DWA Keeper
class DWAKeeper:
    def __init__(self, num_tasks: int, temperature: float = 2.0):
        self.num_tasks = num_tasks
        self.temperature = temperature
        self.prev_losses: Optional[torch.Tensor] = None

    def compute_weights(self, current_avg_losses: torch.Tensor) -> torch.Tensor:
        if self.prev_losses is None:
            weights = torch.ones(self.num_tasks, device=current_avg_losses.device)
        else:
            prev = self.prev_losses.to(current_avg_losses.device)
            loss_ratio = current_avg_losses / (prev + 1e-8)
            exp_term = torch.exp(loss_ratio / self.temperature)
            weights = self.num_tasks * exp_term / (torch.sum(exp_term) + 1e-8)

        self.prev_losses = current_avg_losses.detach().clone()
        return weights

    def reset(self):
        self.prev_losses = None

# Optimizer Builder
def build_optimizer(model: nn.Module, flags: TrainingFlags) -> torch.optim.Optimizer:
    lr, wd = 1e-4, 1e-4
    optimizer_params = filter(lambda p: p.requires_grad, model.parameters())

    if flags.use_pcgrad:
        from Pytorch_PCGrad.pcgrad import PCGrad
        opt = PCGrad(
            model.parameters(),
            optim_cls=torch.optim.AdamW,
            lr=lr, weight_decay=wd,
            reduction='mean'
        )
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    return opt

def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    processed = {}
    batch_size = len(batch)

    if "image" in batch[0] and isinstance(batch[0]["image"], torch.Tensor):
        processed["image"] = torch.stack([b["image"] for b in batch])

    if "text_feat" in batch[0] and isinstance(batch[0]["text_feat"], torch.Tensor):
        processed["text_feat"] = torch.stack([b["text_feat"] for b in batch])
    else:
        img_dim = processed["image"].shape[-1] if "image" in processed else 768
        processed["text_feat"] = torch.zeros(batch_size, 1, img_dim, dtype=processed["image"].dtype if "image" in processed else torch.float32)

    if "missing_modality" in batch[0] and isinstance(batch[0]["missing_modality"], torch.Tensor):
        processed["missing_modality"] = torch.stack([b["missing_modality"] for b in batch])
    else:
        processed["missing_modality"] = torch.zeros(batch_size, dtype=torch.long)

    if "file_path" in batch[0]:
        processed["file_path"] = [b.get("file_path", None) for b in batch]

    label_keys_to_process = TASK_CONFIG.get("label_keys", [])
    if not label_keys_to_process:
         print("警告：TASK_CONFIG 中未定义 label_keys，无法处理标签。")

    for key in label_keys_to_process:
        arr = torch.full((len(batch),), -1, dtype=torch.long)
        for i, b in enumerate(batch):
            if key in b:
                arr[i] = int(b[key]) if not isinstance(b[key], torch.Tensor) else b[key].long().item()

        processed[key] = arr

    return processed

# Loss Function
def calculate_multitask_loss(
    model_output: Dict[str, Any],
    batch: Dict[str, Any],
    task_criteria: Dict[str, nn.Module],
    flags: TrainingFlags,
    dwa_weights: Optional[torch.Tensor] = None,
    model: Optional[nn.Module] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:

    logits_list = model_output.get("logits", [])

    task_losses = []
    task_losses_dict = {}
    device = logits_list[0].device

    for i, task_name in enumerate(TASK_CONFIG["names"]):
        label_key = TASK_CONFIG["label_keys"][i]
        criterion = task_criteria[task_name]
        logits = logits_list[i]

        if logits.dim() > 2:
            if logits.shape[-1] == 1:
                logits = logits.squeeze(-1)
            else:
                pass

        labels = batch[label_key].float() 
        mask = (labels >= 0)
        
        if mask.any():
            loss_i = criterion(logits[mask], labels[mask])
            task_losses.append(loss_i)
            task_losses_dict[f"{task_name}_loss"] = loss_i.item()
        else:
            task_losses.append(torch.tensor(0.0, device=device, requires_grad=False))
            task_losses_dict[f"{task_name}_loss"] = 0.0 

    task_losses_tensor = torch.stack(task_losses)
    total_loss = torch.tensor(0.0, device=device)

    # if use diversity regularization
    if flags.loss_weighting_strategy == "uw":
        log_sigma = getattr(model, 'log_sigma')
        sigma_sq = torch.exp(log_sigma * 2)
        weighted_losses = task_losses_tensor / (2 * sigma_sq + 1e-8)
        regularization = torch.log(sigma_sq.sqrt() + 1e-8).sum()
        total_loss = weighted_losses.sum() + regularization

        for i, task_name in enumerate(TASK_CONFIG["names"]):
             task_losses_dict[f"{task_name}_sigma"] = sigma_sq[i].sqrt().item()

    # if use dynamic weighting average
    elif flags.loss_weighting_strategy == "dwa":
        total_loss = (task_losses_tensor * dwa_weights.to(device)).sum() # 将权重移到同一设备并计算加权和

    # Simple sum of losses
    elif flags.loss_weighting_strategy == "sum":
        total_loss = task_losses_tensor.sum()
    else:
        raise ValueError(f"未知的损失加权策略: {flags.loss_weighting_strategy}")

    if "aux_loss" in model_output and model_output["aux_loss"] is not None:
        aux_loss = model_output["aux_loss"]
        if isinstance(aux_loss, torch.Tensor) and aux_loss.requires_grad:
             total_loss = total_loss + aux_loss
             task_losses_dict["aux_loss"] = aux_loss.item()
        elif isinstance(aux_loss, float):
             task_losses_dict["aux_loss"] = aux_loss
             if total_loss.item() == 0.0 and not total_loss.requires_grad and aux_loss != 0.0:
                 pass

    return total_loss, task_losses_dict

def run_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    flags: TrainingFlags,
    optimizer: Optional[torch.optim.Optimizer] = None,
    task_criteria: Optional[Dict[str, nn.Module]] = None,
    dwa_keeper: Optional[DWAKeeper] = None,
    epoch: int = 0,
    fold_idx: int = 0
    ) -> Dict[str, float]:
    """运行单个训练或验证 epoch。"""
    is_training = optimizer is not None
    model.train(is_training)

    epoch_task_losses_sum = {name: 0.0 for name in TASK_CONFIG["names"]}
    epoch_correct_preds = {name: 0 for name in TASK_CONFIG["names"]}
    epoch_valid_samples = {name: 0 for name in TASK_CONFIG["names"]}
    epoch_batches = 0
    epoch_total_loss_sum = 0.0
    epoch_other_losses_sum = {}

    dwa_weights_this_epoch = None
    if is_training and flags.loss_weighting_strategy == "dwa":
        if dwa_keeper is None:
            raise ValueError("训练时使用 DWA 策略需要提供 DWAKeeper 实例。")
        if dwa_keeper.prev_losses is not None:
              dwa_weights_this_epoch = dwa_keeper.compute_weights(dwa_keeper.prev_losses)
        else:
              dwa_weights_this_epoch = torch.ones(NUM_TASKS, device=device)

    pbar_desc = f"Epoch {epoch+1}/{flags.num_epochs} {'训练' if is_training else '验证'} (Fold {fold_idx+1})"
    pbar = tqdm(dataloader, desc=pbar_desc, leave=False)
    for batch_idx, batch in enumerate(pbar):
        img_features = batch["image"].to(device, non_blocking=True)
        text_features = batch["text_feat"].to(device, non_blocking=True)
        missing_modality = batch["missing_modality"].to(device, non_blocking=True)

        batch_on_device = {}
        label_keys = TASK_CONFIG.get("label_keys", [])
        for key in label_keys:
            if key in batch:
                batch_on_device[key] = batch[key].to(device, non_blocking=True)
            else:
                print(f"警告: 在 run_epoch 中未在批次中找到标签键 '{key}'。")
                batch_on_device[key] = None # 或根据需要进行处理

        with torch.set_grad_enabled(is_training):
            model_output = model(
                img_features=img_features,
                text_feats=text_features,
                missing_modality=missing_modality,
                return_all=(flags.use_diversity_reg)
            )

            total_loss, batch_task_losses_dict = calculate_multitask_loss(
                model_output=model_output,
                batch=batch_on_device, # 使用包含 device 上张量的字典
                task_criteria=task_criteria if task_criteria else TASK_CONFIG["criteria"],
                flags=flags,
                dwa_weights=dwa_weights_this_epoch if flags.loss_weighting_strategy == "dwa" else None,
                model=model
            )

        if is_training:
            optimizer.zero_grad(set_to_none=True)

            if flags.use_pcgrad and isinstance(optimizer, PCGrad):
                 individual_losses_for_pcgrad = []
                 logits_list = model_output.get("logits", [])
                 for i, task_name in enumerate(TASK_CONFIG["names"]):
                     label_key = TASK_CONFIG["label_keys"][i]
                     criterion = (task_criteria or TASK_CONFIG["criteria"]).get(task_name, nn.BCEWithLogitsLoss().to(device))
                     logits = logits_list[i]
                     if logits.dim() > 2 and logits.shape[-1] == 1: logits = logits.squeeze(-1)
                     labels = batch[label_key].float()
                     mask = (labels >= 0)
                     if mask.any():
                         loss_i = criterion(logits[mask], labels[mask])
                         individual_losses_for_pcgrad.append(loss_i)
                     else:
                         individual_losses_for_pcgrad.append(torch.tensor(0.0, device=device))

                 if not individual_losses_for_pcgrad:
                      individual_losses_for_pcgrad.append(torch.tensor(0.0, device=device, requires_grad=True))

                 optimizer.pc_backward(individual_losses_for_pcgrad)
                 optimizer.step()

            else:
                 if total_loss.requires_grad:
                    total_loss.backward()
                    optimizer.step()

    epoch_batches += 1
    epoch_total_loss_sum += total_loss.item() # total_loss 已经是标量，可直接 .item()

    for task_name in TASK_CONFIG["names"]:
        loss_key = f"{task_name}_loss"
        if loss_key in batch_task_losses_dict:
            epoch_task_losses_sum[task_name] += batch_task_losses_dict[loss_key]

        task_idx = TASK_CONFIG["names"].index(task_name)
        label_key = TASK_CONFIG["label_keys"][task_idx]
        logits = model_output["logits"][task_idx] # 模型输出的 logits 已经在 device 上

        # 从已经移动到 device 的字典中获取标签
        labels = batch_on_device.get(label_key)

        if labels is not None: # 确保标签存在且在 device 上
            # 调整 logits 形状 (如果需要)
            if logits.dim() > 2 and logits.shape[-1] == 1: logits = logits.squeeze(-1)
            # 在 device 上计算 mask
            mask = (labels >= 0)

            if mask.any(): # 如果这个任务有有效标签
                # 在 device 上计算预测值
                preds = (torch.sigmoid(logits[mask]) >= 0.5).long()
                # 在 device 上比较预测值和真实标签 (labels[mask] 也在 device 上)
                correct = (preds == labels[mask].long()).sum().item()
                valid_count = mask.sum().item() # 有效样本数量
                epoch_correct_preds[task_name] += correct # 累加正确预测数
                epoch_valid_samples[task_name] += valid_count # 累加有效样本数

        for loss_key, loss_val in batch_task_losses_dict.items():
            # 如果损失键不是任务损失 (如 'cancer_loss')
            if loss_key not in [f"{n}_loss" for n in TASK_CONFIG["names"]]:
                # 累加该损失值
                epoch_other_losses_sum[loss_key] = epoch_other_losses_sum.get(loss_key, 0.0) + loss_val

        avg_total_loss = epoch_total_loss_sum / epoch_batches # 计算当前平均总损失
        postfix_dict = {"总损失": f"{avg_total_loss:.4f}"} # 显示总损失
        for name in TASK_CONFIG["names"]:
             avg_task_loss = epoch_task_losses_sum[name] / epoch_batches # 计算当前任务平均损失
             acc = (epoch_correct_preds[name] / epoch_valid_samples[name]) if epoch_valid_samples[name] > 0 else 0
             postfix_dict[f"{name[:3]}_loss"] = f"{avg_task_loss:.3f}" # 显示任务损失 (缩写)
             postfix_dict[f"{name[:3]}_acc"] = f"{acc:.3f}" # 显示任务准确率 (缩写)
        pbar.set_postfix(postfix_dict) # 更新进度条

    epoch_metrics = {}
    avg_epoch_task_losses = torch.zeros(NUM_TASKS, device=device)
    for i, name in enumerate(TASK_CONFIG["names"]):
        avg_loss = epoch_task_losses_sum[name] / len(dataloader) if len(dataloader) > 0 else 0.0
        accuracy = (epoch_correct_preds[name] / epoch_valid_samples[name]) if epoch_valid_samples[name] > 0 else 0.0
        epoch_metrics[f"{name}_loss"] = avg_loss
        epoch_metrics[f"{name}_acc"] = accuracy
        avg_epoch_task_losses[i] = avg_loss

    epoch_metrics["total_loss"] = epoch_total_loss_sum / len(dataloader) if len(dataloader) > 0 else 0.0
    for loss_key, loss_sum in epoch_other_losses_sum.items():
        epoch_metrics[loss_key] = loss_sum / len(dataloader) if len(dataloader) > 0 else 0.0

    if not is_training and flags.loss_weighting_strategy == "dwa" and dwa_keeper is not None:
        dwa_keeper.prev_losses = avg_epoch_task_losses 

    torch.cuda.empty_cache()
    return epoch_metrics

# ----------------------------------------
# 7) 评估函数 (计算 AUC & F1)
# ----------------------------------------
def evaluate_model_metrics(
    model: nn.Module, # 要评估的模型
    dataloader: DataLoader, # 评估数据集的加载器
    device: torch.device # 计算设备
    ) -> Dict[str, float]: # 返回包含各任务 AUC 和 F1 的字典
    """计算每个任务的 AUC 和 F1 分数，能处理缺失标签。"""
    model.eval() # 设置模型为评估模式

    all_labels: Dict[str, List[int]] = {name: [] for name in TASK_CONFIG["names"]}
    all_probs: Dict[str, List[float]] = {name: [] for name in TASK_CONFIG["names"]}

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="评估指标计算中", leave=False):

            img = batch["image"].to(device, non_blocking=True)
            txt = batch["text_feat"].to(device, non_blocking=True)
            miss = batch["missing_modality"].to(device, non_blocking=True)

            output = model(img_features=img, text_feats=txt, missing_modality=miss, return_all=False)
            logits_list = output.get("logits", [])

            if not isinstance(logits_list, list) or len(logits_list) != NUM_TASKS:
                print(f"警告: 评估过程中模型输出格式异常。跳过此批次。")
                continue

            for i, task_name in enumerate(TASK_CONFIG["names"]):
                label_key = TASK_CONFIG["label_keys"][i]
                labels_batch = batch[label_key].cpu().numpy()
                mask = (labels_batch >= 0)

                if mask.any():
                    logits_task = logits_list[i]

                    if logits_task.dim() > 2 and logits_task.shape[-1] == 1: logits_task = logits_task.squeeze(-1)

                    probs_task = torch.sigmoid(logits_task).cpu().numpy()

                    if probs_task.shape[0] != labels_batch.shape[0]:
                         print(f"警告: 任务 {task_name} 的概率和标签形状不匹配。跳过此批次的该任务评估。")
                         continue
                    
                    all_probs[task_name].extend(probs_task[mask].tolist())
                    all_labels[task_name].extend(labels_batch[mask].astype(int).tolist())

    final_metrics = {}
    for task_name in TASK_CONFIG["names"]:
        labels = all_labels[task_name] # 获取该任务的所有有效标签
        probs = all_probs[task_name] # 获取该任务的所有预测概率
        auc, f1 = float("nan"), float("nan") # 默认指标为 NaN

        # 检查是否有足够样本和类别来计算 AUC 和 F1
        # 需要至少 2 个样本，并且包含至少两个不同的类别 (0 和 1)
        if len(labels) > 1 and len(set(labels)) > 1:
            try:
                # 计算 AUC
                auc = roc_auc_score(labels, probs)
                # 计算 F1 (使用标准 0.5 阈值)
                preds = [1 if p >= 0.5 else 0 for p in probs] # 根据概率生成预测类别
                f1 = f1_score(labels, preds, zero_division=0) # 计算 F1，处理除零情况
            except Exception as e: # 捕获计算过程中可能出现的错误
                print(f"计算任务 '{task_name}' 的指标时出错: {e}")
        elif not labels: # 如果没有收集到任何有效标签
             print(f"警告: 评估期间未找到任务 '{task_name}' 的有效标签。")
        else: # 如果样本数不足或只有一个类别
             print(f"警告: 任务 '{task_name}' 样本数不足或只有一个类别。AUC/F1 设为 NaN。")

        # 将计算出的指标存入字典
        final_metrics[f"{task_name}_auc"] = auc
        final_metrics[f"{task_name}_f1"] = f1

    return final_metrics # 返回包含所有任务 AUC 和 F1 的字典


# ----------------------------------------
# 8) 指标绘图函数 (Metric Plotting)
# ----------------------------------------
def plot_metrics_comparison(
    train_metrics: Dict[str, float], # 训练集评估指标
    val_metrics: Dict[str, float], # 验证集评估指标
    save_dir: str, # 保存图片的目录
    fold_idx: int # 当前 fold 编号
    ):
    """绘制训练集与验证集 AUC 和 F1 分数的对比条形图。"""
    os.makedirs(save_dir, exist_ok=True) # 确保保存目录存在
    task_names = TASK_CONFIG["names"] # 获取任务名称列表
    num_tasks = len(task_names)
    x = np.arange(num_tasks) # 设置 x 轴刻度位置
    width = 0.35 # 设置条形的宽度

    # 分别为 AUC 和 F1 绘制图表
    for metric_suffix in ("auc", "f1"):
        # 从指标字典中提取训练集和验证集的值，处理可能缺失的情况
        train_values = [train_metrics.get(f"{name}_{metric_suffix}", float('nan')) for name in task_names]
        val_values = [val_metrics.get(f"{name}_{metric_suffix}", float('nan')) for name in task_names]

        # 创建图表和坐标轴 (根据任务数量调整图像大小)
        fig, ax = plt.subplots(figsize=(max(8, num_tasks * 1.5), 5))

        # 绘制训练集和验证集的条形图
        rects1 = ax.bar(x - width/2, train_values, width, label='训练集', color='tab:blue')
        rects2 = ax.bar(x + width/2, val_values, width, label='验证集', color='tab:orange')

        # 添加图表元素：Y 轴标签、标题、X 轴刻度标签
        ax.set_ylabel('分数')
        ax.set_title(f'Fold {fold_idx+1} - {metric_suffix.upper()} 分数按任务对比')
        ax.set_xticks(x) # 设置 X 轴刻度位置
        ax.set_xticklabels(task_names, rotation=45, ha="right") # 设置 X 轴刻度标签并旋转
        ax.set_ylim(0, 1.05) # 设置 Y 轴范围
        ax.legend() # 显示图例

        # 在每个条形上方添加数值标签，忽略 NaN 值
        for rects in [rects1, rects2]:
            for rect in rects:
                height = rect.get_height()
                if not np.isnan(height): # 仅当值不是 NaN 时添加标签
                    ax.annotate(f'{height:.3f}', # 格式化数值
                                xy=(rect.get_x() + rect.get_width() / 2, height), # 标签位置 (条形顶部中心)
                                xytext=(0, 3), # 向上偏移 3 个点
                                textcoords="offset points",
                                ha='center', va='bottom', fontsize='small') # 居中对齐

        fig.tight_layout() # 调整布局防止标签重叠
        # 定义保存路径和文件名
        plot_filename = os.path.join(save_dir, f"{metric_suffix}_comparison_fold_{fold_idx+1}.png")
        try:
            # 保存图表
            plt.savefig(plot_filename, dpi=150) # 使用较高分辨率保存
            print(f"已保存指标对比图: {plot_filename}")
        except Exception as e: # 处理保存过程中的错误
            print(f"保存图表 {plot_filename} 时出错: {e}")
        plt.close(fig) # 关闭图表以释放内存

# ----------------------------------------
# 9) 早停机制 (Early Stopping)
# ----------------------------------------
class EarlyStopping:
    """如果验证损失在指定的耐心轮数内没有改善，则停止训练。"""
    def __init__(self, patience: int = 5, delta: float = 0.0, verbose: bool = True):
        """
        Args:
            patience: 容忍验证损失不下降的最大轮数。
            delta: 被视为性能改善的最小损失下降量。
            verbose: 是否打印早停相关信息。
        """
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.counter = 0 # 计录验证损失未改善的轮数
        self.best_score = float('inf') # 记录最佳验证损失 (越小越好)
        self.early_stop = False # 早停标志

    def __call__(self, val_loss: float) -> bool:
        """检查是否应该停止训练。"""
        score = val_loss # 当前验证损失

        # 检查当前损失是否比历史最佳损失改善了至少 delta
        if score < self.best_score - self.delta:
            self.best_score = score # 更新最佳分数
            self.counter = 0 # 重置计数器
            if self.verbose:
                print(f"验证损失下降 ({self.best_score:.6f})。重置早停计数器。")
            self.early_stop = False # 不停止
        else:
            self.counter += 1 # 损失未改善，计数器加一
            if self.verbose:
                print(f"验证损失未从 {self.best_score:.6f} 改善。早停计数器: {self.counter}/{self.patience}")
            if self.counter >= self.patience: # 如果计数器达到耐心值
                print("触发早停。")
                self.early_stop = True # 设置早停标志

        return self.early_stop # 返回是否早停

    def reset_state(self):
        """重置早停状态 (例如，在每个新的 fold 开始时调用)。"""
        self.counter = 0
        self.best_score = float('inf')
        self.early_stop = False


# ----------------------------------------
# 10) Fold 索引加载器 (Fold Indices Loader)
# ----------------------------------------
def _load_fold_indices(
    csv_path: str, # split.csv 文件的路径
    filename_to_idx: Dict[str, int] # 文件名到数据集索引的映射字典
    ) -> Tuple[List[int], List[int]]: # 返回训练集和验证集的索引列表
    """从指定 fold 的 split.csv 文件中加载训练集和验证集的索引。"""
    try:
        # 读取 CSV 文件
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"未找到 Split CSV 文件: {csv_path}")

    required_cols = ["set", "original_filename"] # 检查必需的列是否存在
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"Split CSV 文件必须包含以下列: {required_cols}")

    # 根据 'set' 列筛选出训练集和验证集的文件名
    train_files = df[df["set"] == "train"]["original_filename"].tolist()
    val_files = df[df["set"] == "val"]["original_filename"].tolist()

    # 定义一个内部函数，将文件名列表映射到数据集索引列表
    def map_files_to_indices(files: List[str]) -> List[int]:
        indices = [] # 存储映射后的索引
        missing_files = [] # 存储在映射字典中找不到的文件名
        for f in files:
            idx = filename_to_idx.get(f) # 从字典中查找文件名对应的索引
            if idx is not None: # 如果找到了索引
                indices.append(idx) # 添加到列表
            else: # 如果没找到
                missing_files.append(f) # 添加到未找到列表
        if missing_files: # 如果有未找到的文件，打印警告
            print(f"警告: Split CSV 中的 {len(missing_files)} 个文件在数据集中未找到映射: {missing_files[:5]}...")
        return indices

    # 调用映射函数获取训练集和验证集的索引
    train_indices = map_files_to_indices(train_files)
    val_indices = map_files_to_indices(val_files)

    # 检查是否成功获取到索引
    if not train_indices or not val_indices:
         raise ValueError(f"无法从 {csv_path} 映射得到有效的训练或验证索引。")

    return train_indices, val_indices # 返回索引列表元组

# ----------------------------------------
# 11) 交叉验证循环 (重构版) (Cross-Validation Loop - Refactored)
# ----------------------------------------
def cross_validation_loop(
    dataset: Dataset, # 完整的数据集对象
    model_class: type, # 要实例化的模型类 (例如 MultiTaskMoEModel)
    model_config: Any, # 用于模型类实例化的配置对象或字典
    flags: TrainingFlags # 包含所有超参数和设置的 TrainingFlags 对象
    ):
    """
    执行 k 折交叉验证的函数，使用了重构后的组件。

    Args:
        dataset: 完整的数据集对象。
        model_class: 神经网络模型类。
        model_config: 模型类的配置。
        flags: 包含所有超参数和设置的 TrainingFlags 对象。
    """
    # --- 初始化设置 ---
    # 设置随机种子以保证可复现性
    torch.manual_seed(flags.seed)
    np.random.seed(flags.seed)
    random.seed(flags.seed)
    if torch.cuda.is_available(): # 如果使用 GPU
        torch.cuda.manual_seed_all(flags.seed)
        # 禁用确定性算法和基准测试以可能提高速度，如果需要可重新启用
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # 确定计算设备
    os.makedirs(flags.output_dir, exist_ok=True) # 确保输出目录存在

    # --- 查找 Fold 目录 ---
    try:
        # 获取 folds 目录下所有匹配 'fold_数字' 模式的目录名并排序
        fold_dirs = sorted([d for d in os.listdir(flags.folds_dir) if re.match(r"fold_\d+", d)])
        num_folds = len(fold_dirs) # 获取 fold 数量
        if num_folds == 0: # 如果没有找到 fold 目录
            raise FileNotFoundError(f"在 {flags.folds_dir} 中未找到 Fold 目录")
    except FileNotFoundError: # 如果 folds 目录本身不存在
         raise FileNotFoundError(f"未找到 Folds 目录: {flags.folds_dir}")

    run_name = flags.wandb_run_name or f"cv_{flags.loss_weighting_strategy}_{'pcgrad_' if flags.use_pcgrad else ''}run"
    try:
        wandb.init(
            project=flags.wandb_project, # 项目名称
            name=run_name, # 运行名称
            config={ # 记录超参数和配置
                "num_epochs": flags.num_epochs,
                "num_folds": num_folds,
                "device": str(device),
                "seed": flags.seed,
                "batch_size": flags.batch_size,
                # 记录模型配置 (如果是对象则转为字典)
                "model_config": model_config.__dict__ if hasattr(model_config, '__dict__') else model_config,
                # 记录训练标志 (转为字典)
                "flags": dataclasses.asdict(flags),
                "task_config": TASK_CONFIG, # 记录任务配置
            }
        )
        wandb_active = True # wandb 初始化成功标志
    except Exception as e: # 处理初始化失败的情况
        print(f"初始化 WandB 时出错: {e}。将在没有 WandB 日志记录的情况下继续。")
        wandb_active = False


    # --- 数据集索引映射 ---
    # 创建从 'original_filename' 到数据集索引的映射字典
    try:
        # 假设数据集的每个元素都是包含 'original_filename' 键的字典
        filename_to_idx = {item["original_filename"]: idx for idx, item in enumerate(dataset)}
        if not filename_to_idx: # 如果映射为空
             raise ValueError("无法创建 filename_to_idx 映射。请确保数据集项包含 'original_filename'。")
    except (TypeError, KeyError) as e: # 处理数据集项不支持索引或缺少键的情况
        raise TypeError("数据集项必须支持下标访问 (如字典) 并且包含 'original_filename' 键。") from e


    # --- 交叉验证主循环 ---
    all_fold_results: List[Dict[str, float]] = [] # 存储每个 fold 的结果
    task_names = TASK_CONFIG["names"] # 获取任务名称列表

    for fold_idx, fold_dir_name in enumerate(fold_dirs): # 遍历每个 fold
        print(f"\n===== 开始处理 {fold_dir_name} ({fold_idx + 1}/{num_folds}) =====")
        # 创建当前 fold 的输出子目录
        fold_output_dir = os.path.join(flags.output_dir, fold_dir_name)
        os.makedirs(fold_output_dir, exist_ok=True)

        # --- 加载当前 Fold 的数据索引 ---
        try:
            split_csv_path = os.path.join(flags.folds_dir, fold_dir_name, "split.csv") # 构造 split 文件路径
            # 加载训练集和验证集的索引
            train_indices, val_indices = _load_fold_indices(split_csv_path, filename_to_idx)
        except (FileNotFoundError, ValueError) as e: # 处理加载错误
            print(f"加载 Fold {fold_idx+1} 的索引时出错: {e}。跳过此 Fold。")
            continue # 跳到下一个 fold

        train_subset = Subset(dataset, train_indices) # 创建训练子集
        val_subset = Subset(dataset, val_indices) # 创建验证子集

        train_sampler = ImbalancedDatasetSampler(dataset, train_indices, label_key=TASK_CONFIG["label_keys"][0]) if flags.use_sampler else None

        collate_with_device = lambda batch: collate_fn(batch)

        train_loader = DataLoader(
            train_subset,
            batch_size=flags.batch_size,
            shuffle=(train_sampler is None), # 仅在不使用采样器时打乱顺序
            sampler=train_sampler, # 使用采样器 (如果启用)
            collate_fn=collate_with_device, # 使用自定义的 collate 函数
            num_workers=0, # 根据系统设置工作进程数 (0 表示在主进程中加载)
            pin_memory=torch.cuda.is_available() # 如果使用 GPU，启用内存固定以加速传输
        )
        val_loader = DataLoader(
            val_subset,
            batch_size=flags.batch_size, # 验证时可以使用更大的 batch size
            shuffle=False, # 验证集不打乱顺序
            collate_fn=collate_with_device,
            num_workers=0,
            pin_memory=torch.cuda.is_available()
        )

        try:
            model = model_class(
                config=model_config,
                num_tasks=NUM_TASKS,
                use_uncertainty_weighting=(flags.loss_weighting_strategy == "uw"),
                use_task_norm=flags.use_task_norm,
            ).to(device)

            optimizer = build_optimizer(model, flags)

            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer.original_optimizer if flags.use_pcgrad and hasattr(optimizer, 'original_optimizer') else optimizer,
                mode="min",
                factor=0.5,
                patience=3,
                verbose=True
            )
            task_criteria = {name: crit.to(device) for name, crit in TASK_CONFIG["criteria"].items()}

            # 初始化早停机制
            early_stopper = EarlyStopping(
                patience=flags.early_stopping_patience,
                delta=flags.early_stopping_delta,
                verbose=True
            )
            dwa_keeper = DWAKeeper(NUM_TASKS, flags.dwa_temperature) if flags.loss_weighting_strategy == "dwa" else None

        except Exception as e: # 处理初始化过程中的错误
            print(f"设置 Fold {fold_idx+1} 的模型/优化器时出错: {e}。跳过此 Fold。")
            continue

        best_val_metrics_fold = {f"{name}_acc": 0.0 for name in task_names}

        for epoch in range(flags.num_epochs): # 迭代每个 epoch
            train_epoch_metrics = run_epoch(
                model=model,
                dataloader=train_loader,
                device=device,
                flags=flags,
                optimizer=optimizer, # 传递优化器
                task_criteria=task_criteria,
                dwa_keeper=dwa_keeper, # 传递 DWA keeper
                epoch=epoch,
                fold_idx=fold_idx
            )

            # --- 运行验证 Epoch ---
            val_epoch_metrics = run_epoch(
                model=model,
                dataloader=val_loader,
                device=device,
                flags=flags,
                optimizer=None, # 验证时不传递优化器
                task_criteria=task_criteria,
                dwa_keeper=dwa_keeper, # 传递 DWA keeper 以便更新其状态
                epoch=epoch,
                fold_idx=fold_idx
            )

            # --- 日志记录 (WandB) ---
            # 构造日志字典，包含训练和验证指标，并添加 fold 前缀
            log_payload = {f"fold_{fold_idx+1}/train/{k}": v for k, v in train_epoch_metrics.items()}
            log_payload.update({f"fold_{fold_idx+1}/val/{k}": v for k, v in val_epoch_metrics.items()})
            # 获取当前学习率并记录
            current_lr = (optimizer.original_optimizer if flags.use_pcgrad and hasattr(optimizer, 'original_optimizer') else optimizer).param_groups[0]['lr']
            log_payload[f"fold_{fold_idx+1}/learning_rate"] = current_lr
            log_payload["epoch"] = epoch + 1 # 记录 epoch 编号 (从 1 开始)

            if wandb_active: # 如果 wandb 初始化成功
                wandb.log(log_payload) # 上传日志

            # --- 学习率调度与早停 ---
            # 使用主要任务 (例如第一个任务 'cancer') 的验证损失来驱动调度器和早停
            primary_task_val_loss = val_epoch_metrics.get(f"{task_names[0]}_loss", float('inf'))
            scheduler.step(primary_task_val_loss) # 更新学习率
            if early_stopper(primary_task_val_loss): # 检查是否早停
                break # 如果触发早停，则跳出当前 fold 的 epoch 循环

            # --- 模型检查点 (Checkpointing) ---
            # 根据每个任务在验证集上的最佳准确率保存模型
            for task_name in task_names:
                acc_key = f"{task_name}_acc" # 准确率键名
                current_val_acc = val_epoch_metrics.get(acc_key, 0.0) # 获取当前验证准确率
                # 如果当前准确率优于该任务的历史最佳准确率
                if current_val_acc > best_val_metrics_fold[acc_key]:
                    best_val_metrics_fold[acc_key] = current_val_acc # 更新最佳准确率记录
                    # 定义检查点保存路径
                    checkpoint_path = os.path.join(fold_output_dir, f"best_model_{task_name}_fold{fold_idx+1}.pt")
                    try:
                        # 保存模型状态字典
                        torch.save(model.state_dict(), checkpoint_path)
                        print(f"已保存任务 '{task_name}' 在 fold {fold_idx+1} 的最佳模型 (准确率: {current_val_acc:.4f}) 到 {checkpoint_path}")
                        # 可选：将检查点保存为 wandb artifact
                        # if wandb_active: wandb.save(checkpoint_path, base_path=fold_output_dir)
                    except Exception as e: # 处理保存错误
                        print(f"保存检查点 {checkpoint_path} 时出错: {e}")


            # --- 打印 Epoch 结束时的简报 ---
            print(f"Epoch {epoch+1}/{flags.num_epochs} 结束小结 (Fold {fold_idx+1}):")
            summary_str = []
            # 格式化并打印每个任务的训练/验证准确率和损失
            for name in task_names:
                 tr_acc = train_epoch_metrics.get(f"{name}_acc", -1)
                 val_acc = val_epoch_metrics.get(f"{name}_acc", -1)
                 tr_loss = train_epoch_metrics.get(f"{name}_loss", -1)
                 val_loss = val_epoch_metrics.get(f"{name}_loss", -1)
                 summary_str.append(f"{name[:3]} 准(训/验): {tr_acc:.3f}/{val_acc:.3f}")
                 summary_str.append(f"损(训/验): {tr_loss:.3f}/{val_loss:.3f}")
            print(" | ".join(summary_str))
            print(f"学习率: {current_lr:.2e}") # 打印当前学习率


        # --- Fold 结束: 进行最终评估并绘制指标图 ---
        print(f"Fold {fold_idx+1} 训练结束。正在评估最终模型状态...")
        # 注意：这里评估的是训练结束时的最终模型状态。
        # 如果需要评估在验证集上表现最好的模型，需要在评估前加载相应的 checkpoint。
        final_train_eval_metrics = evaluate_model_metrics(model, train_loader, device) # 在训练集上评估
        final_val_eval_metrics = evaluate_model_metrics(model, val_loader, device) # 在验证集上评估

        print("最终验证集指标 (AUC/F1):")
        for name in task_names:
            auc = final_val_eval_metrics.get(f"{name}_auc", float('nan'))
            f1 = final_val_eval_metrics.get(f"{name}_f1", float('nan'))
            print(f"  {name}: AUC={auc:.4f}, F1={f1:.4f}")

        # 绘制该 fold 的训练集 vs 验证集指标对比图
        plot_metrics_comparison(final_train_eval_metrics, final_val_eval_metrics, fold_output_dir, fold_idx)

        # --- 存储当前 Fold 的结果 ---
        # 包含训练过程中达到的最佳验证准确率和最终评估指标
        fold_summary = {**best_val_metrics_fold} # 使用训练中记录的最佳准确率
        # 添加最终的训练集评估指标
        fold_summary.update({f"final_train_{k}": v for k, v in final_train_eval_metrics.items()})
        # 添加最终的验证集评估指标
        fold_summary.update({f"final_val_{k}": v for k, v in final_val_eval_metrics.items()})
        all_fold_results.append(fold_summary) # 将该 fold 的结果添加到总列表

        # --- 将当前 Fold 的最终结果记录到 WandB ---
        if wandb_active:
            # 使用特定前缀区分 fold 的最终评估结果
            wandb.log({f"fold_{fold_idx+1}/final_eval/{k}": v for k, v in fold_summary.items()})


        # --- 清理当前 Fold 使用的资源 ---
        del model, optimizer, scheduler, train_loader, val_loader, train_subset, val_subset
        torch.cuda.empty_cache() # 清理 GPU 缓存


    # --- 交叉验证结束: 汇总并报告结果 ---
    print("\n=== 交叉验证完成 ===")
    if not all_fold_results: # 如果没有任何 fold 成功完成
        print("没有成功完成的 Fold。")
        if wandb_active: wandb.finish() # 结束 wandb 运行
        return # 提前退出

    # --- 计算并打印各 Fold 的关键结果 ---
    print("各 Fold 结果概要:")
    for i, fold_res in enumerate(all_fold_results, 1): # 遍历每个 fold 的结果字典
        # 格式化打印最佳准确率、最终 AUC 和 F1
        acc_str = " | ".join([f"{name[:3]}_最佳准: {fold_res.get(f'{name}_acc', float('nan')):.4f}" for name in task_names])
        auc_str = " | ".join([f"{name[:3]}_最终AUC: {fold_res.get(f'final_val_{name}_auc', float('nan')):.4f}" for name in task_names])
        f1_str =  " | ".join([f"{name[:3]}_最终F1: {fold_res.get(f'final_val_{name}_f1', float('nan')):.4f}" for name in task_names])
        print(f"Fold {i}: {acc_str}")
        print(f"      {auc_str}")
        print(f"      {f1_str}")

    # --- 计算交叉验证的平均值和标准差 ---
    cv_summary = {} # 存储汇总指标
    print("\n交叉验证汇总指标 (平均值 ± 标准差):")
    # 定义要汇总的指标前缀和后缀
    for metric_prefix in ["final_val", "final_train"]: # 可以添加 "best_val" 等
        for metric_suffix in ["auc", "f1", "acc"]: # 包含准确率用于汇总
            key_pattern = f"{metric_prefix}_{{task_name}}_{metric_suffix}"
            # 特殊处理：对于最终验证准确率，我们通常关心的是训练过程中达到的最佳值
            if metric_prefix == "final_val" and metric_suffix == "acc":
                 key_pattern = "{task_name}_acc" # 使用记录的最佳准确率键

            for task_name in task_names: # 遍历每个任务
                full_key_pattern = key_pattern.format(task_name=task_name)
                # 检查该指标是否存在于 fold 结果中
                if any(full_key_pattern in fr for fr in all_fold_results):
                    # 提取所有 fold 中该指标的值
                    values = [fr.get(full_key_pattern, float('nan')) for fr in all_fold_results]
                    # 过滤掉 NaN 值进行计算
                    valid_values = [v for v in values if not np.isnan(v)]
                    if valid_values: # 如果存在有效值
                        mean_val = np.mean(valid_values) # 计算平均值
                        std_val = np.std(valid_values) # 计算标准差
                        # 将汇总结果存入字典
                        summary_key = f"CV_mean_{full_key_pattern}"
                        cv_summary[summary_key] = mean_val
                        cv_summary[f"CV_std_{full_key_pattern}"] = std_val
                        # 打印汇总结果
                        print(f"  {task_name} ({full_key_pattern}): {mean_val:.4f} ± {std_val:.4f}")
                    else: # 如果所有 fold 都没有该指标的有效值
                         print(f"  {task_name} ({full_key_pattern}): N/A (所有 Fold 均无有效结果)")


    # --- 将交叉验证汇总结果记录到 WandB ---
    if wandb_active:
        wandb.log(cv_summary) # 上传汇总指标
        wandb.finish() # 结束 wandb 运行

    print("\n交叉验证流程结束。")
