import dataclasses
from typing import Optional
import torch.nn as nn

@dataclasses.dataclass
class TrainingFlags:
    # Multi-Task Options
    loss_weighting_strategy: str = "mgda"
    use_pcgrad: bool = True
    use_diversity_reg: bool = False
    diversity_beta: float = 1e-4

    # if use task norm
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

class MoEConfig:
    def __init__(
        self,
        hidden_size: int = 768,
        intermediate_size: int = 768,
        moe_intermediate_size: int = 768,
        pretraining_tp: int = 1,
        hidden_act: str = "gelu",
        num_experts_per_tok: int = 4,       
        n_routed_experts: int = 15,         
        scoring_func: str = "softmax",
        aux_loss_alpha: float = 0.01,
        seq_aux: bool = True,
        norm_topk_prob: bool = True,
        n_shared_experts: int = 1,
        expert_mode: str = "both",          
        num_tasks: int = 4,
        task_specific_init: bool = True,
        task_reg_alpha: float = 0.01,
        router_temperature_init: float = 1.5,
        router_temperature_min: float = 0.5,
        router_temperature_decay: float = 1e-5,
        router_add_noise: bool = True,
        router_dropout: float = 0.10,
    ):
        
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.moe_intermediate_size = moe_intermediate_size
        self.pretraining_tp = pretraining_tp
        self.hidden_act = hidden_act
        self.num_experts_per_tok = num_experts_per_tok
        self.n_routed_experts = n_routed_experts
        self.scoring_func = scoring_func
        self.aux_loss_alpha = aux_loss_alpha
        self.seq_aux = seq_aux
        self.norm_topk_prob = norm_topk_prob
        self.n_shared_experts = n_shared_experts
        self.expert_mode = expert_mode
        self.num_tasks = num_tasks
        self.task_specific_init = task_specific_init
        self.task_reg_alpha = task_reg_alpha
        self.router_temperature_init = router_temperature_init
        self.router_temperature_min = router_temperature_min
        self.router_temperature_decay = router_temperature_decay
        self.router_add_noise = router_add_noise
        self.router_dropout = router_dropout