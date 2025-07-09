import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.nn.init as init
from typing import Tuple

class AddAuxiliaryLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, loss):
        ctx.required_aux_loss = loss is not None and loss.requires_grad
        return x

    @staticmethod
    def backward(ctx, grad_output):
        grad_loss = None
        if ctx.required_aux_loss:
            grad_loss = torch.ones(1, dtype=grad_output.dtype, device=grad_output.device)
        return grad_output, grad_loss

class TokenRouterGating(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.top_k: int = config.num_experts_per_tok
        self.add_noise: bool = getattr(config, "router_add_noise", True)
        self.alpha: float = getattr(config, "aux_loss_alpha", 0.01)

    def forward(self, hidden_states: torch.Tensor, expert_gate_weights: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        num_active_experts = expert_gate_weights.size(0)
        flat_tokens = hidden_states.view(-1, self.config.hidden_size)

        logits = F.linear(flat_tokens, expert_gate_weights)
    
        if self.add_noise and self.training:
            gumbel_noise = -torch.empty_like(logits).exponential_().log()
            logits = logits + gumbel_noise

        scores = logits.softmax(dim=-1)
        topk_gates, local_indices = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)
        topk_gates = topk_gates / (topk_gates.sum(dim=-1, keepdim=True) + 1e-6)

        aux_loss = None
        if self.training and self.alpha > 0.0:
            expert_counts = F.one_hot(local_indices.view(-1), num_classes=num_active_experts).float().sum(0)
            expert_importance = scores.mean(0)
            load_balancing_loss = (expert_importance * expert_counts).sum() * self.alpha
            aux_loss = load_balancing_loss

        return local_indices, topk_gates, aux_loss

class DeepseekMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.moe_intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

class HierarchicalTaskMoE(nn.Module):
    def __init__(self, config, num_tasks: int):
        super().__init__()
        self.config = config
        self.num_tasks = num_tasks
        self.hidden_size = config.hidden_size
        self.num_experts = config.num_experts
        self.k = config.num_experts_per_tok
        
        self.num_task_experts = config.num_task_experts
        self.num_generalists = getattr(config, 'num_generalists', 2)
        assert self.num_experts > self.num_generalists, "通才专家数量必须小于总专家数量"

        self.task_router = nn.Linear(self.hidden_size, self.num_experts, bias=False)
        
        self.gate_weight = nn.Parameter(torch.empty(self.num_experts, self.hidden_size))
        self.token_router = TokenRouterGating(config)
        init.kaiming_uniform_(self.gate_weight, a=math.sqrt(5))

        self.task_emb = nn.Embedding(num_tasks, self.hidden_size)
        self.experts = nn.ModuleList([
            DeepseekMLP(config) for _ in range(self.num_experts)
        ])

    def forward(self, hidden_states: torch.Tensor, task_id: int):
        # input format: [B, T, D] or [B, D]
        is_2d_input = hidden_states.dim() == 2
        if is_2d_input:
            hidden_states = hidden_states.unsqueeze(1)
            
        bsz, seq_len, dim = hidden_states.shape
        task_vec = self.task_emb.weight[task_id]

        # experts specific to the task
        task_expert_scores = self.task_router(task_vec)
        _, task_expert_indices = torch.topk(task_expert_scores, self.num_task_experts, sorted=False)
        # generalist experts
        generalist_indices = torch.arange(
            self.num_experts - self.num_generalists, self.num_experts, device=hidden_states.device
        )

        active_expert_indices = torch.cat([task_expert_indices, generalist_indices]).unique()
        active_gate_weights = self.gate_weight[active_expert_indices]

        gate_input = hidden_states + task_vec
        local_topk_indices, topk_gates, aux_loss = self.token_router(gate_input, active_gate_weights)
        global_topk_indices = active_expert_indices[local_topk_indices]

        flat_tokens = hidden_states.view(-1, dim)
        flat_global_indices = global_topk_indices.view(-1)
        
        expanded_tokens = flat_tokens.repeat_interleave(self.k, dim=0)
        dispatched_output = torch.empty_like(expanded_tokens)

        for i, exp_id in enumerate(active_expert_indices):
            mask = (flat_global_indices == exp_id)
            if mask.any():
                dispatched_output[mask] = self.experts[exp_id](expanded_tokens[mask])
        
        dispatched_output = dispatched_output.view(-1, self.k, dim)
        weighted_output = dispatched_output * topk_gates.unsqueeze(-1)
        combined_output = weighted_output.sum(dim=1)
        final_y = combined_output.view(bsz, seq_len, dim)

        if aux_loss is not None and self.training:
            final_y = AddAuxiliaryLoss.apply(final_y, aux_loss)
            
        return final_y