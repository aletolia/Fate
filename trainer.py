from typing import List, Dict, Optional, Any, Tuple
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm
from sklearn.metrics import roc_auc_score, f1_score

from config import TrainingFlags, TASK_CONFIG, NUM_TASKS
from utils import DWAKeeper, MGDALoss

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

    # Initialize MGDA loss function if using it
    mgda_loss_fn = None
    if is_training and flags.loss_weighting_strategy == 'mgda':
        mgda_loss_fn = MGDALoss(model, device)

    dwa_weights = dwa_keeper.compute_weights(dwa_keeper.prev_losses) if is_training and flags.loss_weighting_strategy == "dwa" and dwa_keeper and dwa_keeper.prev_losses is not None else torch.ones(NUM_TASKS, device=device)

    pbar = tqdm(dataloader, desc=f"{'Train' if is_training else 'Val'}", leave=False)
    error_log = []
    for batch in pbar:
        try:
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
                
                if is_training and flags.loss_weighting_strategy == 'mgda':
                    loss, batch_losses_dict = mgda_loss_fn(model_output, batch, task_criteria)
                else:
                    # Default loss calculation (sum or DWA)
                    task_losses = []
                    batch_losses_dict = {}
                    for i, task_name in enumerate(TASK_CONFIG["names"]):
                        logits = model_output["logits"][i].squeeze()
                        labels = batch[TASK_CONFIG["label_keys"][i]].float()
                        mask = (labels >= 0)
                        
                        if mask.any():
                            loss_i = task_criteria[task_name](logits[mask], labels[mask])
                            task_losses.append(loss_i)
                            batch_losses_dict[f"{task_name}_loss"] = loss_i.item()
                        else:
                            task_losses.append(torch.tensor(0.0, device=device))
                            batch_losses_dict[f"{task_name}_loss"] = 0.0
                    
                    task_losses_tensor = torch.stack(task_losses)
                    if flags.loss_weighting_strategy == "dwa" and dwa_weights is not None:
                        loss = (task_losses_tensor * dwa_weights.to(device)).sum()
                    else: # Default to sum
                        loss = task_losses_tensor.sum()

            if is_training:
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
        except Exception as e:
            error_log.append({
                'file_path': batch.get('file_path', 'N/A'),
                'error': str(e)
            })
            continue
    
    # Calculate final metrics for the epoch
    metrics = {f"{name}_loss": epoch_losses[f"{name}_loss"] / len(dataloader) for name in TASK_CONFIG["names"]}
    metrics["total_loss"] = epoch_total_loss / len(dataloader)
    
    for name in TASK_CONFIG["names"]:
        if all_labels[name]:
            labels_cat = torch.cat(all_labels[name]).numpy()
            preds_cat = torch.cat(all_preds[name]).numpy()
            if len(set(labels_cat)) > 1: # AUC requires at least two classes
                metrics[f"{name}_auc"] = float(roc_auc_score(labels_cat, preds_cat))
            metrics[f"{name}_f1"] = float(f1_score(labels_cat, (preds_cat >= 0.5).astype(int), zero_division=0))
            metrics[f"{name}_acc"] = ((preds_cat >= 0.5).astype(int) == labels_cat).mean()

    if not is_training and flags.loss_weighting_strategy == "dwa" and dwa_keeper:
        avg_task_losses = torch.tensor([metrics[f"{name}_loss"] for name in TASK_CONFIG["names"]], device=device)
        dwa_keeper.prev_losses = avg_task_losses

    return metrics, error_log
