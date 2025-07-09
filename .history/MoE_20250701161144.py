import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import h5py
import numpy as np
from transformers import AutoModel
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
import torch.multiprocessing as mp
import wandb
import random
from math import sqrt
import re
import pandas as pd
import torch.nn.init as init

class AddAuxiliaryLoss(torch.autograd.Function):
    """Helper to propagate auxiliary load‑balancing loss to the main graph."""

    @staticmethod
    def forward(ctx, x, loss):
        assert loss is None or loss.numel() == 1
        ctx.dtype = None if loss is None else loss.dtype
        ctx.required_aux_loss = loss is not None and loss.requires_grad
        return x

    @staticmethod
    def backward(ctx, grad_output):
        grad_loss = None
        if ctx.required_aux_loss:
            grad_loss = torch.ones(1, dtype=ctx.dtype, device=grad_output.device)
        return grad_output, grad_loss

class NoisyTopKGating(nn.Module):
    """A temperature‑annealed, noisy Top‑k router with optional dropout & load‑balancing."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.top_k: int = config.num_experts_per_tok          # k experts per token
        self.num_experts: int = config.n_routed_experts       # total experts
        self.temperature: float = getattr(config, "router_temperature_init", 1.0)
        self.temperature_min: float = getattr(config, "router_temperature_min", 0.5)
        self.temperature_decay: float = getattr(config, "router_temperature_decay", 1e-5)
        self.add_noise: bool = getattr(config, "router_add_noise", True)
        self.drop_prob: float = getattr(config, "router_dropout", 0.0)  # gate dropout probability
        # auxiliary load‑balancing
        self.alpha: float = getattr(config, "aux_loss_alpha", 0.0)
        self.seq_aux: bool = getattr(config, "seq_aux", False)
        self.norm_topk_prob: bool = getattr(config, "norm_topk_prob", True)

        self.hidden_dim: int = config.hidden_size
        self.weight = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim))
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def _anneal_temperature(self):
        if self.training and self.temperature > self.temperature_min:
            self.temperature = max(self.temperature_min, self.temperature - self.temperature_decay)
            
    def forward(self, hidden_states: torch.Tensor):
        if hidden_states.dim() == 2:
            hidden_states = hidden_states.unsqueeze(1)  # (B, 1, D)
        bsz, seq_len, _ = hidden_states.shape
        flat_tokens = hidden_states.view(-1, self.hidden_dim)

        logits = F.linear(flat_tokens, self.weight)  # (T, E)
        logits = logits / self.temperature
        if self.add_noise and self.training:
            gumbel_noise = -torch.empty_like(logits).exponential_().log()  # Gumbel(0,1)
            logits = logits + gumbel_noise

        scores = logits.softmax(dim=-1)                                        # (T, E)
        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)  # each token's k best experts

        if self.norm_topk_prob:
            topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + 1e-20)

        if self.training and self.drop_prob > 0.0:
            mask = torch.rand_like(topk_weight).gt(self.drop_prob)              # keep with prob 1-p
            topk_weight = topk_weight * mask
            topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + 1e-20)

        # auxiliary load‑balancing loss (Shazeer et al., 2017)
        aux_loss = None
        if self.training and self.alpha > 0.0:
            scores_for_aux = scores.detach()          # stop grad wrt logits
            topk_idx_aux = topk_idx.view(bsz, -1)     # (B, S*k)
            if self.seq_aux:  # sequence‑level load‑balancing
                seq_scores = scores_for_aux.view(bsz, seq_len, self.num_experts)
                ce = torch.zeros(bsz, self.num_experts, device=scores.device)
                ce.scatter_add_(1, topk_idx_aux, torch.ones_like(topk_idx_aux, dtype=torch.float))
                ce = ce.div(seq_len * self.top_k / self.num_experts)
                aux_loss = (ce * seq_scores.mean(dim=1)).sum(dim=1).mean() * self.alpha
            else:            # batch‑level load‑balancing
                mask_ce = F.one_hot(topk_idx_aux.view(-1), num_classes=self.num_experts).float()
                ce = mask_ce.mean(0)
                pi = scores_for_aux.mean(0)
                fi = ce * self.num_experts
                aux_loss = (pi * fi).sum() * self.alpha

        self._anneal_temperature()

        return topk_idx, topk_weight, aux_loss

class DeepseekMoE_TaskSpecificExperts(nn.Module):
    def __init__(self, config, num_tasks: int, expert_mode: str = "both"):
        super().__init__()
        assert expert_mode in {"local", "shared", "both"}
        self.config = config
        self.num_tasks = num_tasks
        self.expert_mode = expert_mode
        self.num_experts_per_tok = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts
        self.hidden_size = config.hidden_size

        if expert_mode in ("local", "both"):
            self.gate = NoisyTopKGating(config)
        else:
            self.gate = None

        if expert_mode in ("local", "both"):
            self.task_emb = nn.Embedding(num_tasks, self.hidden_size)
        else:
            self.task_emb = None

        if expert_mode in ("local", "both"):
            self.local_experts = nn.ModuleDict({
                str(tid): nn.ModuleList([
                    DeepseekMLP(config, intermediate_size=config.moe_intermediate_size)
                    for _ in range(self.n_routed_experts)
                ]) for tid in range(num_tasks)
            })
        else:
            self.local_experts = None

        if expert_mode in ("shared", "both"):
            if getattr(config, "n_shared_experts", 1) > 1:
                inter = config.moe_intermediate_size * config.n_shared_experts
            else:
                inter = config.moe_intermediate_size
            self.shared_expert = DeepseekMLP(config, intermediate_size=inter)
        else:
            self.shared_expert = None

        if expert_mode == "both":
            self.combination_gate_proj = nn.Linear(self.hidden_size, 2, bias=False)
        else:
            self.combination_gate_proj = None

    def forward(self, hidden_states: torch.Tensor, task_id: int):
        if not (0 <= task_id < self.num_tasks):
            raise ValueError(f"Invalid task_id {task_id}.")
        task_key = str(task_id)
        bsz, seq_len, dim = hidden_states.shape
        device = hidden_states.device

        local_y = 0.0
        if self.local_experts is not None:
            gate_input = hidden_states + self.task_emb.weight[task_id]  # broadcast
            topk_idx, topk_weight, aux_loss = self.gate(gate_input)

            # flatten token dim
            flat_tokens = hidden_states.view(-1, dim)                   # (T, D)
            flat_idx = topk_idx.view(-1)                                # (T*k)
            flat_weight = topk_weight.view(-1, self.num_experts_per_tok)  # (T, k) -> (T*k)

            # gather & dispatch to experts
            if self.training:
                expanded_tokens = flat_tokens.repeat_interleave(self.num_experts_per_tok, 0)  # (T*k, D)
                local_out = torch.empty_like(expanded_tokens)
                experts = self.local_experts[task_key]
                for exp_id in range(self.n_routed_experts):
                    mask = flat_idx == exp_id
                    if mask.any():
                        out = experts[exp_id](expanded_tokens[mask])
                        local_out[mask] = out
                local_out = local_out.view(-1, self.num_experts_per_tok, dim)  # (T, k, D)
                weighted = local_out * topk_weight.unsqueeze(-1)
                combined = weighted.sum(dim=1)                               # (T, D)
                local_y = combined.view(bsz, seq_len, dim)

                # attach aux loss
                if aux_loss is not None:
                    local_y = AddAuxiliaryLoss.apply(local_y, aux_loss)
            else:
                local_y = self._moe_infer(flat_tokens, flat_idx, topk_weight.view(-1, 1), task_key)
                local_y = local_y.view(bsz, seq_len, dim)

        shared_y = 0.0
        if self.shared_expert is not None:
            shared_y = self.shared_expert(hidden_states)
        if self.expert_mode == "local":
            return local_y
        elif self.expert_mode == "shared":
            return shared_y
        else:  # both
            logits = self.combination_gate_proj(hidden_states)         # (B, S, 2)
            alpha = logits.softmax(dim=-1)                             # ensure sum=1
            alpha_local, alpha_shared = alpha.unbind(-1)               # each (B, S)
            fused = alpha_local.unsqueeze(-1) * local_y + alpha_shared.unsqueeze(-1) * shared_y
            return fused

    @torch.no_grad()
    def _moe_infer(self, flat_tokens, flat_indices, flat_weights, task_key: str):
        experts = self.local_experts[task_key]
        out_acc = torch.zeros_like(flat_tokens)
        k = self.num_experts_per_tok
        sorted_idx = flat_indices.argsort()
        token_ids = sorted_idx // k
        counts = flat_indices.bincount(minlength=self.n_routed_experts).cpu().numpy().cumsum()
        start = 0
        for exp_id, end in enumerate(counts):
            if start == end:
                continue
            sel = sorted_idx[start:end]
            tokens_sel = token_ids[start:end]
            out = experts[exp_id](flat_tokens[tokens_sel])
            out.mul_(flat_weights[sel])
            out_acc.scatter_reduce_(0, tokens_sel.unsqueeze(1).expand_as(out), out, reduce="sum")
            start = end
        return out_acc

ACT2FN = {
    "relu": F.relu,
    "gelu": F.gelu,
    "tanh": torch.tanh,
}

class DeepseekMLP(nn.Module):
    def __init__(self, config, hidden_size=None, intermediate_size=None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size if hidden_size is None else hidden_size
        self.intermediate_size = config.intermediate_size if intermediate_size is None else intermediate_size

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        if self.config.pretraining_tp > 1:
            slice_size = self.intermediate_size // self.config.pretraining_tp
            gate_proj_slices = self.gate_proj.weight.split(slice_size, dim=0)
            up_proj_slices = self.up_proj.weight.split(slice_size, dim=0)
            down_proj_slices = self.down_proj.weight.split(slice_size, dim=1)

            gate_proj = torch.cat(
                [F.linear(x, gate_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1
            )
            up_proj = torch.cat(
                [F.linear(x, up_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1
            )

            intermediate_states = (self.act_fn(gate_proj) * up_proj).split(slice_size, dim=2)
            down_proj = sum([
                F.linear(intermediate_states[i], down_proj_slices[i])
                for i in range(self.config.pretraining_tp)
            ])
        else:
            down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj