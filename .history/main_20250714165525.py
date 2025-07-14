import os
import re
import random
import dataclasses
import argparse
from typing import List, Dict, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
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
from losses import FocalLoss, SymmetricCrossEntropyLoss

# Fold Indices Loader
def _load_fold_indices(csv_path: str, filename_to_idx: Dict[str, int]) -> Tuple[List[int], List[int]]:
    """Loads training and validation indices from a fold's split.csv file."""
    df = pd.read_csv(csv_path)
    train_files = df[df["set"] == "train"]["original_filename"]
    val_files = df[df["set"] == "val"]["original_filename"]
    train_indices = [filename_to_idx[f] for f in train_files if f in filename_to_idx]
    val_indices = [filename_to_idx[f] for f in val_files if f in filename_to_idx]
    return train_indices, val_indices

def _build_loss_criterion(flags: TrainingFlags) -> nn.Module:
    """Builds the loss function based on the provided flags."""
    if flags.loss_function == 'focal':
        print(f"Using Focal Loss (alpha={flags.focal_loss_alpha}, gamma={flags.focal_loss_gamma})")
        return FocalLoss(alpha=flags.focal_loss_alpha, gamma=flags.focal_loss_gamma)
    elif flags.loss_function == 'sce':
        print(f"Using Symmetric Cross Entropy (alpha={flags.sce_alpha}, beta={flags.sce_beta})")
        return SymmetricCrossEntropyLoss(alpha=flags.sce_alpha, beta=flags.sce_beta)
    elif flags.loss_function == 'bce':
        print("Using standard BCEWithLogitsLoss")
        return nn.BCEWithLogitsLoss()
    else:
        raise ValueError(f"Unknown loss function: {flags.loss_function}")

def cross_validation_loop(
    dataset: Dataset, 
    model_class: type, 
    model_config: Any, 
    flags: TrainingFlags,
    ):
    """Main loop for k-fold cross-validation."""
    # --- Setup ---
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

    # --- Loop over folds ---
    for fold_idx, fold_dir_name in enumerate(fold_dirs):
        print(f"\n===== FOLD {fold_idx + 1}/{len(fold_dirs)} =====")
        fold_output_dir = os.path.join(flags.output_dir, fold_dir_name)
        os.makedirs(fold_output_dir, exist_ok=True)
        
        # --- Data Loading for current fold ---
        split_csv_path = os.path.join(flags.folds_dir, fold_dir_name, "split.csv")
        train_indices, val_indices = _load_fold_indices(split_csv_path, filename_to_idx)
        
        train_subset = Subset(dataset, train_indices)
        val_subset = Subset(dataset, val_indices)
        
        train_sampler = ImbalancedDatasetSampler(dataset, train_indices, TASK_CONFIG["label_keys"][0]) if flags.use_sampler else None
        train_loader = DataLoader(train_subset, batch_size=flags.batch_size, shuffle=(train_sampler is None), sampler=train_sampler, collate_fn=collate_fn, num_workers=0, pin_memory=True)
        val_loader = DataLoader(val_subset, batch_size=flags.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0, pin_memory=True)

        # --- Model, Optimizer, and Helpers Initialization ---
        model = model_class(config=model_config, num_tasks=NUM_TASKS, use_cross_attention=flags.use_cross_attention).to(device)
        optimizer = build_optimizer(model, flags)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer.original_optimizer if flags.use_pcgrad else optimizer, 'min', patience=3, factor=0.5, verbose=True)
        
        criterion = _build_loss_criterion(flags)
        task_criteria = {name: criterion.to(device) for name in TASK_CONFIG["names"]}

        early_stopper = EarlyStopping(patience=flags.early_stopping_patience, delta=flags.early_stopping_delta)
        dwa_keeper = DWAKeeper(NUM_TASKS, flags.dwa_temperature) if flags.loss_weighting_strategy == "dwa" else None
        
        best_val_loss = float('inf')

        # --- Epoch Loop ---
        all_error_logs = []
        for epoch in range(flags.num_epochs):
            print(f"--- Epoch {epoch+1}/{flags.num_epochs} ---")
            train_metrics, train_errors = run_epoch(model, train_loader, device, flags, optimizer, task_criteria, dwa_keeper)
            val_metrics, val_errors = run_epoch(model, val_loader, device, flags, None, task_criteria, dwa_keeper)
            
            # --- Logging and Checkpointing ---
            log_payload = {f"train/{k}": v for k, v in train_metrics.items()}
            log_payload.update({f"val/{k}": v for k, v in val_metrics.items()})
            wandb.log(log_payload, step=epoch)

            # --- Epoch End Summary ---
            print(f"  Train Loss: {train_metrics['total_loss']:.4f}")
            for name in TASK_CONFIG["names"]:
                print(f"    {name} -> Acc: {train_metrics.get(name + '_acc', 0):.4f}, AUC: {train_metrics.get(name + '_auc', 0):.4f}, F1: {train_metrics.get(name + '_f1', 0):.4f}, Sens: {train_metrics.get(name + '_sensitivity', 0):.4f}, Spec: {train_metrics.get(name + '_specificity', 0):.4f}")
            
            print(f"  Val Loss: {val_metrics['total_loss']:.4f}")
            for name in TASK_CONFIG["names"]:
                print(f"    {name} -> Acc: {val_metrics.get(name + '_acc', 0):.4f}, AUC: {val_metrics.get(name + '_auc', 0):.4f}, F1: {val_metrics.get(name + '_f1', 0):.4f}, Sens: {val_metrics.get(name + '_sensitivity', 0):.4f}, Spec: {val_metrics.get(name + '_specificity', 0):.4f}")
            
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
        all_error_logs.extend(train_errors)
        all_error_logs.extend(val_errors)
        
        del model, optimizer, scheduler, train_loader, val_loader
        torch.cuda.empty_cache()

    print("\n=== Cross-validation finished. ===")
    if all_error_logs:
        error_df = pd.DataFrame(all_error_logs)
        error_df.to_csv(os.path.join(flags.output_dir, "error_log.csv"), index=False)
        print(f"Saved error log to {os.path.join(flags.output_dir, 'error_log.csv')}")
    wandb.finish()

def main():
    """Main function to parse arguments, set up, and run the training loop."""
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description='Multi-Task Learning Framework for Pathological Analysis')
    parser.add_argument('--csv_label_path', type=str, default='/home/aletolia/documents/code_notes/visualization/MMP/crossTest/pathReport_modified_cleaned.csv', help='Path to the CSV file containing additional labels')
    parser.add_argument('--h5_data_dir', type=str, default='/home/aletolia/documents/code_notes/visualization/MMP/crossTest/output_conch2', help='Directory containing H5 feature files')
    parser.add_argument('--folds_dir', type=str, default='folds_new', help='Directory containing fold definitions')
    
    # Task and Model Options
    parser.add_argument('--use_pcgrad', action='store_true', help='Use PCGrad for gradient optimization')
    parser.add_argument('--use_cross_attention', action='store_true', help='Use cross attention')
    parser.add_argument('--use_diversity_reg', action='store_true', help='Use diversity regularization')
    parser.add_argument('--diversity_beta', type=float, default=1e-3, help='Beta for diversity regularization')
    parser.add_argument('--use_task_norm', action='store_true', help='Use task normalization')
    parser.add_argument('--use_sampler', action='store_true', help='Use imbalanced dataset sampler')

    # Loss Function Options
    parser.add_argument('--loss_function', type=str, default='bce', choices=['bce', 'focal', 'sce'], help='Loss function to use')
    parser.add_argument('--focal_loss_alpha', type=float, default=0.25, help='Alpha for Focal Loss')
    parser.add_argument('--focal_loss_gamma', type=float, default=2.0, help='Gamma for Focal Loss')
    parser.add_argument('--sce_alpha', type=float, default=1.0, help='Alpha for Symmetric Cross Entropy')
    parser.add_argument('--sce_beta', type=float, default=1.0, help='Beta for Symmetric Cross Entropy')
    parser.add_argument('--loss_weighting_strategy', type=str, default='sum', choices=['sum', 'dwa', 'mgda'], help='Loss weighting strategy')

    # Training Hyperparameters
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--early_stopping_patience', type=int, default=5, help='Early stopping patience')
    parser.add_argument('--early_stopping_delta', type=float, default=1e-4, help='Early stopping delta')
    parser.add_argument('--dwa_temperature', type=float, default=2.0, help='DWA temperature')

    # I/O and Logging
    parser.add_argument('--output_dir', type=str, default='./training_output', help='Output directory for trained models')
    parser.add_argument('--wandb_project', type=str, default='multitask_moe_refactored', help='Wandb project name')
    parser.add_argument('--wandb_run_name', type=str, default=None, help='Wandb run name')

    args = parser.parse_args()

    # --- Configuration Setup ---
    model_config = MoEConfig()
    flags = TrainingFlags(
        loss_weighting_strategy=args.loss_weighting_strategy,
        use_pcgrad=args.use_pcgrad,
        use_cross_attention=args.use_cross_attention,
        use_diversity_reg=args.use_diversity_reg,
        diversity_beta=args.diversity_beta,
        use_task_norm=args.use_task_norm,
        use_sampler=args.use_sampler,
        loss_function=args.loss_function,
        focal_loss_alpha=args.focal_loss_alpha,
        focal_loss_gamma=args.focal_loss_gamma,
        sce_alpha=args.sce_alpha,
        sce_beta=args.sce_beta,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        seed=args.seed,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_delta=args.early_stopping_delta,
        dwa_temperature=args.dwa_temperature,
        folds_dir=args.folds_dir,
        output_dir=args.output_dir,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name
    )

    print("train config:")
    print(f"data dir: {args.h5_data_dir}")
    print(f"label path: {args.csv_label_path}")
    print(f"cross validation folds: {flags.folds_dir}")
    print("-" * 22)

    # --- Dataset Loading ---
    ln_labels, va_labels, pe_labels = load_additional_labels_from_csv(args.csv_label_path)

    dataset = PathologyMultimodalDataset(
        data_dir=args.h5_data_dir,
        lymph_node_status=ln_labels,
        vascular_status=va_labels,
        perineural_status=pe_labels,
        missing_probs={0: 0.0, 1: 0.0}
    )
    print(f"initialized dataset with {len(dataset)} samples")

    # --- Run Training ---
    cross_validation_loop(
        dataset=dataset,
        model_class=MultiTaskModelWithPerTaskFusion,
        model_config=model_config,
        flags=flags
    )
    print("\nfinished cross-validation training successfully")

if __name__ == '__main__':
    main()
