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
        return FocalLoss(alpha=flags.focal_loss_alpha, gamma=flags.focal_loss_gamma, reduction="none")
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
    """Main loop for k-fold cross-validation with a two-stage training process: 
       1. Joint training to find the best shared backbone.
       2. Sequential fine-tuning of each task head from that best backbone.
    """
    # --- Setup ---
    torch.manual_seed(flags.seed)
    np.random.seed(flags.seed)
    random.seed(flags.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(flags.output_dir, exist_ok=True)
    
    CURRICULUM_SCHEDULE = {
        1: ["cancer"], 
        5: ["cancer", "lymph"], 
        10: ["cancer", "lymph", "vascular", "perineural"], 
    }

    fold_dirs = sorted([d for d in os.listdir(flags.folds_dir) if re.match(r"fold_\d+", d)])
    if not fold_dirs:
        raise FileNotFoundError(f"No fold directories found in {flags.folds_dir}")

    wandb.init(project=flags.wandb_project, name=flags.wandb_run_name, config=dataclasses.asdict(flags), mode="offline")

    filename_to_idx = {item["original_filename"]: idx for idx, item in enumerate(dataset)}
    
    for fold_idx, fold_dir_name in enumerate(fold_dirs):
        print(f"===== FOLD {fold_idx + 1}/{len(fold_dirs)} ====")
        fold_output_dir = os.path.join(flags.output_dir, fold_dir_name)
        os.makedirs(fold_output_dir, exist_ok=True)
        
        split_csv_path = os.path.join(flags.folds_dir, fold_dir_name, "split.csv")
        train_indices, val_indices = _load_fold_indices(split_csv_path, filename_to_idx)
        
        train_subset = Subset(dataset, train_indices)
        val_subset = Subset(dataset, val_indices)
        
        train_loader = DataLoader(train_subset, batch_size=flags.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0, pin_memory=True)
        val_loader = DataLoader(val_subset, batch_size=flags.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0, pin_memory=True)

        model = model_class(config=model_config, num_tasks=NUM_TASKS, use_cross_attention=flags.use_cross_attention).to(device)
        
        # --- STAGE 1: JOINT TRAINING ---
        print("--- Starting Stage 1: Joint Training to find best backbone ---")
        joint_optimizer = build_optimizer(model, flags)
        joint_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(joint_optimizer.original_optimizer if flags.use_pcgrad else joint_optimizer, mode='max', patience=3, factor=0.5, verbose=True)
        global_early_stopper = EarlyStopping(patience=flags.early_stopping_patience, delta=flags.early_stopping_delta)
        task_stoppers = {name: EarlyStopping(patience=flags.early_stopping_patience, delta=0.001, verbose=False) for name in TASK_CONFIG["names"]}
        criterion = _build_loss_criterion(flags)
        task_criteria = {name: criterion.to(device) for name in TASK_CONFIG["names"]}
        dwa_keeper = DWAKeeper(NUM_TASKS, flags.dwa_temperature) if flags.loss_weighting_strategy == "dwa" else None

        active_tasks = []
        stopped_tasks = set()
        best_avg_val_auc = -1.0

        for epoch in range(flags.num_epochs):
            current_epoch = epoch + 1
            # ... (Curriculum Learning and Epoch Execution Logic is the same) ...
            # This part remains the same as our last correct implementation.

            # --- GLOBAL Early Stopping and Model Saving (based on average val AUC of active tasks) ---
            if not active_tasks:
                print("All tasks have been early-stopped. Ending joint training stage.")
                break

            active_aucs = [val_metrics.get(f"{task}_auc", 0.0) for task in active_tasks]
            avg_val_auc = sum(active_aucs) / len(active_aucs)
            print(f"  Avg Val AUC (for best joint model): {avg_val_auc:.4f}")

            joint_scheduler.step(avg_val_auc)

            if avg_val_auc > best_avg_val_auc:
                best_avg_val_auc = avg_val_auc
                torch.save(model.state_dict(), os.path.join(fold_output_dir, f"best_joint_model_fold_{fold_idx+1}.pt"))
                print(f"Saved best JOINT model for fold {fold_idx+1} (Avg Val AUC: {best_avg_val_auc:.4f})")

            if global_early_stopper(-avg_val_auc):
                print("--- Global early stopping triggered. Ending joint training. ---")
                break

        # --- STAGE 2: SEQUENTIAL FINE-TUNING OF TASK-SPECIFIC HEADS ---
        print("--- Starting Stage 2: Sequential Fine-tuning ---")
        
        best_joint_model_path = os.path.join(fold_output_dir, f"best_joint_model_fold_{fold_idx+1}.pt")
        if not os.path.exists(best_joint_model_path):
            print("Warning: No best joint model found. Skipping fine-tuning.")
            continue

        for task_id, task_to_finetune in enumerate(TASK_CONFIG["names"]):
            print(f"-- Fine-tuning head for task: '{task_to_finetune}' --")
            
            # Reload the best joint model to ensure a clean start for each task
            model.load_state_dict(torch.load(best_joint_model_path))

            # Freeze backbone, unfreeze all heads initially
            backbone_param_names = ["cross_attn_layers", "shared_fusion_module"]
            for name, param in model.named_parameters():
                param.requires_grad = not any(name.startswith(p_name) for p_name in backbone_param_names)

            # Create a new optimizer for the fine-tuning of the specific head
            finetune_params = [p for p in model.parameters() if p.requires_grad]
            finetune_optimizer = torch.optim.AdamW(finetune_params, lr=flags.learning_rate * 0.1) # Use a smaller LR
            finetune_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(finetune_optimizer, 'max', patience=2, factor=0.5)
            
            best_task_finetune_auc = -1.0
            num_finetune_epochs = 10 # More epochs for fine-tuning

            for ft_epoch in range(num_finetune_epochs):
                # Run a training epoch for only the single active task
                run_epoch(model, train_loader, device, flags, [task_to_finetune], finetune_optimizer, task_criteria, None)
                # Run a validation epoch for only the single active task
                ft_val_metrics, _ = run_epoch(model, val_loader, device, flags, [task_to_finetune], None, task_criteria, None)
                
                task_val_auc = ft_val_metrics.get(f"{task_to_finetune}_auc", 0.0)
                print(f"  Fine-tune Epoch {ft_epoch+1}/{num_finetune_epochs} | Task: {task_to_finetune} | Val AUC: {task_val_auc:.4f}")
                finetune_scheduler.step(task_val_auc)

                if task_val_auc > best_task_finetune_auc:
                    best_task_finetune_auc = task_val_auc
                    # Save the model state that is best *for this specific task*
                    save_path = os.path.join(fold_output_dir, f"final_model_best_for_{task_to_finetune}_fold_{fold_idx+1}.pt")
                    torch.save(model.state_dict(), save_path)
                    print(f"    -> Saved best model for '{task_to_finetune}' (Val AUC: {best_task_finetune_auc:.4f})")

        del model, joint_optimizer, finetune_optimizer
        torch.cuda.empty_cache()

    print("\n=== Cross-validation finished. ===")
    # if all_error_logs:
    #     error_df = pd.DataFrame(all_error_logs)
    #     error_df.to_csv(os.path.join(flags.output_dir, "error_log.csv"), index=False)
    #     print(f"Saved error log to {os.path.join(flags.output_dir, 'error_log.csv')}")
    # wandb.finish()

def main():
    """Main function to parse arguments, set up, and run the training loop."""
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description='Multi-Task Learning Framework for Pathological Analysis')
    parser.add_argument('--csv_label_path', type=str, default='/root/pathReport_modified_cleaned.csv', help='Path to the CSV file containing additional labels')
    parser.add_argument('--h5_data_dir', type=str, default='/root/autodl-tmp/output_conch', help='Directory containing H5 feature files')
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