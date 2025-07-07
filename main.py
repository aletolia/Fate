import os
import re
import random
import dataclasses
from typing import List, Dict, Tuple, Any

import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import wandb
from torch.utils.data import DataLoader, Subset, Dataset

from config import TrainingFlags, TASK_CONFIG, NUM_TASKS
from utils import ImbalancedDatasetSampler, collate_fn, DWAKeeper, EarlyStopping, build_optimizer
from trainer import run_epoch
from model import MultiTaskModelWithPerTaskFusion
from datasets import PathologyMultimodalDataset, load_additional_labels_from_csv
from config import MoEConfig

# Fold Indices Loader
def _load_fold_indices(csv_path: str, filename_to_idx: Dict[str, int]) -> Tuple[List[int], List[int]]:
    df = pd.read_csv(csv_path)
    train_files = df[df["set"] == "train"]["original_filename"]
    val_files = df[df["set"] == "val"]["original_filename"]
    train_indices = [filename_to_idx[f] for f in train_files if f in filename_to_idx]
    val_indices = [filename_to_idx[f] for f in val_files if f in filename_to_idx]
    return train_indices, val_indices

def cross_validation_loop(
    dataset: Dataset, 
    model_class: type, 
    model_config: Any, 
    flags: TrainingFlags
    ):

    torch.manual_seed(flags.seed)
    np.random.seed(flags.seed)
    random.seed(flags.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(flags.output_dir, exist_ok=True)
    
    fold_dirs = sorted([d for d in os.listdir(flags.folds_dir) if re.match(r"fold_\d+", d)])
    if not fold_dirs:
        raise FileNotFoundError(f"No fold directories found in {flags.folds_dir}")

    wandb.init(project=flags.wandb_project, name=flags.wandb_run_name, config=dataclasses.asdict(flags))

    filename_to_idx = {item["original_filename"]: idx for idx, item in enumerate(dataset)}
    all_fold_results = []

    for fold_idx, fold_dir_name in enumerate(fold_dirs):
        print(f"\n===== FOLD {fold_idx + 1}/{len(fold_dirs)} =====")
        fold_output_dir = os.path.join(flags.output_dir, fold_dir_name)
        os.makedirs(fold_output_dir, exist_ok=True)
        
        split_csv_path = os.path.join(flags.folds_dir, fold_dir_name, "split.csv")
        train_indices, val_indices = _load_fold_indices(split_csv_path, filename_to_idx)
        
        train_subset = Subset(dataset, train_indices)
        val_subset = Subset(dataset, val_indices)
        
        train_sampler = ImbalancedDatasetSampler(dataset, train_indices, TASK_CONFIG["label_keys"][0]) if flags.use_sampler else None
        train_loader = DataLoader(train_subset, batch_size=flags.batch_size, shuffle=(train_sampler is None), sampler=train_sampler, collate_fn=collate_fn, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_subset, batch_size=flags.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=4, pin_memory=True)

        model = model_class(config=model_config, num_tasks=NUM_TASKS, use_uncertainty_weighting=(flags.loss_weighting_strategy == "uw")).to(device)
        optimizer = build_optimizer(model, flags)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer.original_optimizer if flags.use_pcgrad else optimizer, 'min', patience=3, factor=0.5, verbose=True)
        task_criteria = {name: crit.to(device) for name, crit in TASK_CONFIG["criteria"].items()}
        early_stopper = EarlyStopping(patience=flags.early_stopping_patience, delta=flags.early_stopping_delta)
        dwa_keeper = DWAKeeper(NUM_TASKS, flags.dwa_temperature) if flags.loss_weighting_strategy == "dwa" else None
        
        best_val_loss = float('inf')

        for epoch in range(flags.num_epochs):
            print(f"--- Epoch {epoch+1}/{flags.num_epochs} ---")
            train_metrics = run_epoch(model, train_loader, device, flags, optimizer, task_criteria, dwa_keeper)
            val_metrics = run_epoch(model, val_loader, device, flags, None, task_criteria, dwa_keeper)
            
            log_payload = {f"train/{k}": v for k, v in train_metrics.items()}
            log_payload.update({f"val/{k}": v for k, v in val_metrics.items()})
            wandb.log(log_payload, step=epoch)

            print(f"Val Loss: {val_metrics['total_loss']:.4f} | Primary Task Val AUC: {val_metrics.get(f'{TASK_CONFIG['names'][0]}_auc', 0):.4f}")
            
            primary_val_loss = val_metrics['total_loss']
            scheduler.step(primary_val_loss)

            if primary_val_loss < best_val_loss:
                best_val_loss = primary_val_loss
                torch.save(model.state_dict(), os.path.join(fold_output_dir, f"best_model_fold_{fold_idx+1}.pt"))
                print(f"Saved best model for fold {fold_idx+1}")

            if early_stopper(primary_val_loss):
                break

        # --- Final Fold Evaluation ---
        all_fold_results.append(val_metrics)
        
        del model, optimizer, scheduler, train_loader, val_loader
        torch.cuda.empty_cache()

    print("\n=== Cross-validation finished. ===")
    wandb.finish()

if __name__ == '__main__':
    CSV_LABEL_PATH = "/home/aletolia/documents/code_notes/visualization/MMP/crossTest/pathReport_modified_cleaned.csv"  # 包含附加标签的CSV文件
    H5_DATA_DIR = "/home/aletolia/documents/code_notes/visualization/MMP/crossTest/output_conch2"                        # 包含H5特征文件的目录
    FOLDS_DIR = "folds_new"

    model_config = MoEConfig()
    flags = TrainingFlags(
        folds_dir=FOLDS_DIR
    )

    print("train config:")
    print(f"data dir: {H5_DATA_DIR}")
    print(f"label path: {CSV_LABEL_PATH}")
    print(f"cross validation folds: {flags.folds_dir}")
    print("-" * 22)

    ln_labels, va_labels, pe_labels = load_additional_labels_from_csv(CSV_LABEL_PATH)

    dataset = PathologyMultimodalDataset(
        data_dir=H5_DATA_DIR,
        lymph_node_status=ln_labels,
        vascular_status=va_labels,
        perineural_status=pe_labels,
        missing_probs={0: 0.0, 1: 0.0}
    )
    print(f"initialized dataset with {len(dataset)} samples")

    cross_validation_loop(
        dataset=dataset,
        model_class=MultiTaskModelWithPerTaskFusion,
        model_config=model_config,
        flags=flags
    )
    print("\nfinished cross-validation training successfully")