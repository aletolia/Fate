from typing import List, Dict, Optional, Any, Tuple
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm
from sklearn.metrics import roc_auc_score, f1_score

from config import TrainingFlags, TASK_CONFIG, NUM_TASKS
from utils import DWAKeeper, MGDALoss

def run_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    flags: TrainingFlags,
    optimizer: Optional[torch.optim.Optimizer] = None,
    task_criteria: Optional[Dict[str, nn.Module]] = None,
    dwa_keeper: Optional[DWAKeeper] = None,
) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
    """Runs a single epoch of training or validation, displaying live metrics."""
    is_training = optimizer is not None
    model.train(is_training)

    epoch_losses = {f"{task}_loss": 0.0 for task in TASK_CONFIG["names"]}
    all_preds = {task: [] for task in TASK_CONFIG["names"]}
    all_labels = {task: [] for task in TASK_CONFIG["names"]}
    error_log = []

    mgda_loss_fn = MGDALoss(model, device) if is_training and flags.loss_weighting_strategy == 'mgda' else None
    
    # Pre-calculate DWA weights for the entire epoch
    dwa_weights = None
    if is_training and flags.loss_weighting_strategy == "dwa" and dwa_keeper is not None:
        dwa_weights = dwa_keeper.compute_weights(dwa_keeper.prev_losses).to(device)

    # Use a description that reflects the mode (Train/Val)
    pbar_desc = f"{'Train' if is_training else 'Val'}"
    pbar = tqdm(dataloader, desc=pbar_desc, leave=False)

    for batch in pbar:
        try:
            batch = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            with torch.set_grad_enabled(is_training):
                model_output = model(
                    img_features=batch["image"],
                    text_feats=batch["text_feat"],
                    missing_modality=batch["missing_modality"],
                )
                
                # --- Loss Calculation ---
                if mgda_loss_fn:
                    total_loss, batch_losses = mgda_loss_fn(model_output, batch, task_criteria)
                else:
                    task_losses = []
                    batch_losses = {}
                    for i, task_name in enumerate(TASK_CONFIG["names"]):
                        logits = model_output["logits"][i].view(-1)
                        labels = batch[TASK_CONFIG["label_keys"][i]].float().view(-1)
                        mask = labels >= 0
                        
                        if mask.any():
                            loss_i = task_criteria[task_name](logits[mask], labels[mask])
                            task_losses.append(loss_i)
                            batch_losses[f"{task_name}_loss"] = loss_i.item()
                        else: # Handle cases with no valid labels in a batch
                            task_losses.append(torch.tensor(0.0, device=device))
                            batch_losses[f"{task_name}_loss"] = 0.0
                    
                    task_losses_tensor = torch.stack(task_losses)
                    if dwa_weights is not None:
                        total_loss = (task_losses_tensor * dwa_weights).sum()
                    else: # Default to simple sum
                        total_loss = task_losses_tensor.sum()

            if is_training:
                optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                optimizer.step()

            # --- Live Metrics Update ---
            postfix_dict = {'loss': f'{total_loss.item():.4f}'}
            for i, task_name in enumerate(TASK_CONFIG["names"]):
                
                raw_logits = model_output["logits"][i]
                raw_labels = batch[TASK_CONFIG["label_keys"][i]]
                logits = torch.atleast_1d(raw_logits)
                labels = torch.atleast_1d(raw_labels.float())
                
                mask = labels >= 0
                if not mask.any():
                    continue

                # Store preds and labels for epoch-end metrics
                valid_labels = labels[mask].cpu()
                valid_preds = torch.sigmoid(model_output["logits"][i].view(-1)[mask]).cpu()
                all_labels[task_name].append(valid_labels)
                all_preds[task_name].append(valid_preds)
                
                # Update epoch losses
                epoch_losses[f"{task_name}_loss"] += batch_losses.get(f"{task_name}_loss", 0.0)

                # Calculate running accuracy for the progress bar
                # We concatenate all seen labels/preds in the epoch for a more stable running accuracy
                running_labels = torch.cat(all_labels[task_name]).numpy()
                running_preds = torch.cat(all_preds[task_name]).numpy()
                acc = np.mean((running_preds >= 0.5) == running_labels)
                postfix_dict[f'{task_name[:4]}_acc'] = f'{acc:.2%}' # Use abbreviated name for cleaner bar

            print(postfix_dict)
            pbar.set_postfix(**postfix_dict)

        except Exception as e:
            error_log.append({'file_path': batch.get('file_path', 'N/A'), 'error': str(e)})
            print(f"\n[调试信息] 捕获到异常: {e}\n")
            error_log.append({
                'file_path': batch.get('file_path', 'N/A'),
                'error': str(e)
            })
            continue
    
    # --- Final Epoch Metrics Calculation ---
    num_batches = len(dataloader)
    metrics = {key: val / num_batches for key, val in epoch_losses.items()}
    metrics["total_loss"] = sum(metrics.values())
    
    for task_name in TASK_CONFIG["names"]:
        if not all_labels[task_name]:
            continue
        
        labels_cat = torch.cat(all_labels[task_name]).numpy()
        preds_cat = torch.cat(all_preds[task_name]).numpy()
        
        metrics[f"{task_name}_acc"] = np.mean((preds_cat >= 0.5) == labels_cat)
        metrics[f"{task_name}_f1"] = f1_score(labels_cat, (preds_cat >= 0.5), zero_division=0)
        if len(np.unique(labels_cat)) > 1: # AUC requires multiple classes
            metrics[f"{task_name}_auc"] = roc_auc_score(labels_cat, preds_cat)

    # Update DWA keeper with validation losses for the next training epoch
    if not is_training and flags.loss_weighting_strategy == "dwa" and dwa_keeper:
        avg_task_losses = torch.tensor([metrics[f"{name}_loss"] for name in TASK_CONFIG["names"]], device=device)
        dwa_keeper.update_losses(avg_task_losses)

    return metrics, error_log