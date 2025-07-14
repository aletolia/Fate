import dataclasses
from typing import Optional
import torch.nn as nn

@dataclasses.dataclass
class TrainingFlags:
    # Multi-Task Options
    loss_weighting_strategy: str = "mgda"
    use_pcgrad: bool = True
    use_diversity_reg: bool = False
    use_cross_attention: bool = True
    diversity_beta: float = 1e-4

    # if use task norm
    use_task_norm: bool = False
    use_sampler: bool = False

    # Loss Function Options
    loss_function: str = 'bce' # 'bce', 'focal', 'sce'
    focal_loss_alpha: float = 0.25
    focal_loss_gamma: float = 2.0
    sce_alpha: float = 1.0
    sce_beta: float = 1.0

    # Parameters
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    batch_size: int = 1
    num_epochs: int = 30
    seed: int = 42

    # Early Stopping & DWA
    early_stopping_patience: int = 5
    early_stopping_delta: float = 1e-4
    dwa_temperature: float = 2.0

    folds_dir: str = "folds"
    output_dir: str = "./training_output"
    wandb_project: str = "multitask_moe_refactored"
    wandb_run_name: Optional[str] = None

TASK_CONFIG = {
    "names": ["cancer", "lymph", "vascular", "perineural"],
    "label_keys": ["label", "lymph_node_label", "vascular_thrombus", "perineural_invasion"],
}
NUM_TASKS = len(TASK_CONFIG["names"])

class MoEConfig:
    def __init__(
        self,
        hidden_size: int = 768,
        moe_intermediate_size: int = 768,
        num_experts: int = 32,                # Total number of experts
        num_task_experts: int = 4,            # Number of task-specific experts to select
        num_generalists: int = 2,             # Number of generalist experts (shared across tasks)
        num_experts_per_tok: int = 4,         # Number of experts to route each token to (k)
        aux_loss_alpha: float = 0.01,         # Scaling factor for the auxiliary load-balancing loss
        router_add_noise: bool = True,        # Whether to add Gumbel noise during routing in training
    ):
        self.hidden_size = hidden_size
        self.moe_intermediate_size = moe_intermediate_size
        self.num_experts = num_experts
        self.num_task_experts = num_task_experts
        self.num_generalists = num_generalists
        self.num_experts_per_tok = num_experts_per_tok
        self.aux_loss_alpha = aux_loss_alpha
        self.router_add_noise = router_add_noise