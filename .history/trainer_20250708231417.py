from typing import List, Dict, Optional, Any, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm
from sklearn.metrics import roc_auc_score, f1_score

from config import TrainingFlags, TASK_CONFIG, NUM_TASKS
from utils import DWAKeeper
from libmtl.loss import MGDA
from libmtl.weighting import MGDA_UB

# Loss Function
def calculate_multitask_loss(
    model_output: Dict[str, Any],
    batch: Dict[str, Any],
    task_criteria: Dict[str, nn.Module],
    flags: TrainingFlags,
    dwa_weights: Optional[torch.Tensor] = None,
    model: Optional[nn.Module] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:

    logits_list = model_output["logits"]
    device = logits_list[0].device
    task_losses = []
    task_losses_dict = {}

    for i, task_name in enumerate(TASK_CONFIG["names"]):
        logits = logits_list[i].squeeze()
        labels = batch[TASK_CONFIG["label_keys"][i]].float()
        mask = (labels >= 0)
        
        if mask.any():
            loss_i = task_criteria[task_name](logits[mask], labels[mask])
            task_losses.append(loss_i)
            task_losses_dict[f"{task_name}_loss"] = loss_i.item()
        else:
            task_losses.append(torch.tensor(0.0, device=device))
            task_losses_dict[f"{task_name}_loss"] = 0.0

    task_losses_tensor = torch.stack(task_losses)

    # if DWA
    if flags.loss_weighting_strategy == "dwa" and dwa_weights is not None:
        total_loss = (task_losses_tensor * dwa_weights.to(device)).sum()
    elif flags.loss_weighting_strategy == "sum":
        total_loss = task_losses_tensor.sum()
    elif flags.loss_weighting_strategy == "mgda":
        total_loss = MGDA_UB(task_losses_tensor, model)
    else: # Default to sum
        total_loss = task_losses_tensor.sum()

    if "aux_loss" in model_output:
        total_loss += model_output["aux_loss"]
        task_losses_dict["aux_loss"] = model_output["aux_loss"].item()
        
    return total_loss, task_losses_dict

# Epoch Runner
def run_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    flags: TrainingFlags,
    optimizer: Optional[torch.optim.Optimizer] = None,
    task_criteria: Optional[Dict[str, nn.Module]] = None,
    dwa_keeper: Optional[DWAKeeper] = None
    ) -> Dict[str, float]:
    """Runs a single epoch of training or validation."""
    is_training = optimizer is not None
    model.train(is_training)

    epoch_losses = {f"{name}_loss": 0.0 for name in TASK_CONFIG["names"]}
    epoch_total_loss = 0.0
    all_preds = {name: [] for name in TASK_CONFIG["names"]}
    all_labels = {name: [] for name in TASK_CONFIG["names"]}

    dwa_weights = dwa_keeper.compute_weights(dwa_keeper.prev_losses) if is_training and flags.loss_weighting_strategy == "dwa" and dwa_keeper and dwa_keeper.prev_losses is not None else torch.ones(NUM_TASKS, device=device)

    pbar = tqdm(dataloader, desc=f"{'Train' if is_training else 'Val'}", leave=False)
    for batch in pbar:
        # Move batch to device
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device, non_blocking=True)

        with torch.set_grad_enabled(is_training):
            model_output = model(
                img_features=batch["image"],
                text_feats=batch["text_feat"],
                missing_modality=batch["missing_modality"],
                return_all=(flags.use_diversity_reg)
            )
            if flags.loss_weighting_strategy == 'mgda':
                loss, batch_losses_dict = calculate_multitask_loss(model_output, batch, task_criteria, flags, dwa_weights, model)
            else:
                loss, batch_losses_dict = calculate_multitask_loss(model_output, batch, task_criteria, flags, dwa_weights, model)

        if is_training:
            if flags.loss_weighting_strategy == 'mgda':
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            else:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        epoch_total_loss += loss.item()
        for name in TASK_CONFIG["names"]:
            epoch_losses[f"{name}_loss"] += batch_losses_dict.get(f"{name}_loss", 0.0)
            
            # Collect labels and predictions for metric calculation
            mask = (batch[TASK_CONFIG["label_keys"][TASK_CONFIG["names"].index(name)]] >= 0)
            if mask.any():
                labels = batch[TASK_CONFIG["label_keys"][TASK_CONFIG["names"].index(name)]][mask].cpu()
                preds = torch.sigmoid(model_output["logits"][TASK_CONFIG["names"].index(name)].squeeze()[mask]).cpu()
                all_labels[name].append(labels)
                all_preds[name].append(preds)
    
    # Calculate final metrics for the epoch
    metrics = {f"{name}_loss": epoch_losses[f"{name}_loss"] / len(dataloader) for name in TASK_CONFIG["names"]}
    metrics["total_loss"] = epoch_total_loss / len(dataloader)
    
    for name in TASK_CONFIG["names"]:
        if all_labels[name]:
            labels_cat = torch.cat(all_labels[name]).numpy()
            preds_cat = torch.cat(all_preds[name]).numpy()
            if len(set(labels_cat)) > 1: # AUC requires at least two classes
                metrics[f"{name}_auc"] = roc_auc_score(labels_cat, preds_cat)
            metrics[f"{name}_f1"] = f1_score(labels_cat, (preds_cat >= 0.5).astype(int), zero_division=0)
            metrics[f"{name}_acc"] = ((preds_cat >= 0.5).astype(int) == labels_cat).mean()

    if not is_training and flags.loss_weighting_strategy == "dwa" and dwa_keeper:
        avg_task_losses = torch.tensor([metrics[f"{name}_loss"] for name in TASK_CONFIG["names"]], device=device)
        dwa_keeper.prev_losses = avg_task_losses

    return metrics