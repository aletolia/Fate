import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any
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

class Fate(nn.Module):
    def __init__(
        self,
        img_dim: int = 768,
        text_dim: int = 768,
        dim: int = 768,
        heads: int = 8,
        dropout: float = 0.1,
        num_ts_tokens: int = 1,
        add_ts_tokens_in_first_half: bool = False,
        learn_ts_tokens: bool = False,
        layer_idx: int = 0,
        total_layers: int = 1,
        fusion_mode: str = "identity",
        residual_mode: str = "identity",
        use_task_norm: bool = False,
        num_tasks: int = 4,
        **kw,
    ):
        super().__init__()

        self.residual_mode = residual_mode.lower()
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim**-0.5
        self.layer_idx = layer_idx
        self.total_layers = total_layers
        self.add_ts_tokens_to_text = add_ts_tokens_in_first_half
        self.fusion_mode = fusion_mode

        self.num_tasks = num_tasks
        self.num_ts_tokens = num_ts_tokens
        self.learn_ts_tokens = learn_ts_tokens

        self.img_proj = nn.Linear(img_dim, dim)
        self.text_proj = nn.Linear(text_dim, dim)
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

        if fusion_mode == "concat" or fusion_mode == "full":
            self.fusion_layer = nn.Linear(dim * 2, dim)
        elif fusion_mode == "weighted":
            self.register_parameter("fusion_weights", nn.Parameter(torch.ones(2) / 2.0))
        elif fusion_mode == "attention":
             self.fusion_attention_proj = nn.Linear(dim, 1)
        else:
            self.fusion_layer = None
            if fusion_mode not in ["identity", "none"]:
                 print(f"OptimizedMultiModalCrossAttention Info: fusion_mode '{fusion_mode}' does not require specific fusion layer init here.")

        if self.residual_mode == "gated":
            self.alpha = nn.Parameter(torch.zeros(1))
        elif self.residual_mode == "highway":
            self.gate_proj = nn.Linear(dim, dim)

        if self.add_ts_tokens_to_text and self.num_ts_tokens > 0:
            self.task_specific_tokens = nn.Parameter(
                torch.randn(num_tasks, num_ts_tokens, dim) * 0.02
            )
            if not self.learn_ts_tokens:
                self.task_specific_tokens.requires_grad = False
        else:
             self.task_specific_tokens = None

        self.task_norm = None

        if self.fusion_mode not in ["moe", "identity", "none"]:
            self.res_proj = nn.Linear(self.dim, self.dim)
        else:
             self.res_proj = None

        self._init_weights()

    def _init_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Linear):
                if 'proj' in name or ('fusion' in name and hasattr(self, 'fusion_layer') and m is self.fusion_layer) or ('gate' in name and self.residual_mode == "highway"):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _compute_cross_attention(self, q, k, v):
        bsz, q_len, _ = q.shape
        _, kv_len, _ = k.shape

        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        q = q.view(bsz, q_len, self.heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, kv_len, self.heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, kv_len, self.heads, self.head_dim).transpose(1, 2)

        # scaled dot-product attention
        dropout_p = self.dropout.p if self.training else 0.0
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p, is_causal=False)

        out = out.transpose(1, 2).contiguous().view(bsz, q_len, self.dim)
        return out

    def _add_residual(self, y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if self.residual_mode in ("none", "hyper"):
            return y
        elif self.residual_mode == "identity":
            if y.shape == x.shape:
                 return y + x
            else:
                 return y
        elif self.residual_mode == "gated":
             if y.shape == x.shape:
                  return y + self.alpha * x
        elif self.residual_mode == "highway":
             if y.shape == x.shape:
                  gate = torch.sigmoid(self.gate_proj(x))
                  return (1.0 - gate) * x + gate * y
        else:
            raise ValueError(f"Unknown residual mode: {self.residual_mode}")

    def forward(
        self,
        img_features: torch.Tensor,
        text_feats: Optional[torch.Tensor],
        task_id: int = 0, 
    ):
        if img_features.dim() == 2:
            img_features = img_features.unsqueeze(0)
        bsz = img_features.size(0)

        img_proj = self.img_proj(img_features) # Shape: [B, L_i, D]

        if text_feats is None:
            text_feats_proj = torch.zeros(
                bsz, 1, self.dim, device=img_proj.device, dtype=img_proj.dtype
            )
            effective_text_len = 1 # For residual matching if needed
        else:
            if text_feats.dim() == 1:
                text_feats = text_feats.unsqueeze(0)
            if text_feats.dim() == 2:
                 text_feats = text_feats.unsqueeze(1)
            text_feats_proj = self.text_proj(text_feats) # Shape: [B, L_t, D]
            effective_text_len = text_feats_proj.shape[1]

        text_query = text_feats_proj
        
        if self.task_specific_tokens is not None and self.add_ts_tokens_to_text:
            valid_task_id = task_id % self.num_tasks
            ts_tokens_for_task = self.task_specific_tokens[valid_task_id] # Shape [num_ts, D]
            ts_tokens = ts_tokens_for_task.unsqueeze(0).expand(bsz, -1, -1)
            text_query = torch.cat([ts_tokens, text_feats_proj], dim=1) # Shape: [B, num_ts + L_t, D]

        text_using_img_ctx = self._compute_cross_attention(text_query, img_proj, img_proj)
        img_using_text_ctx = self._compute_cross_attention(img_proj, text_query, text_query)

        text_ctx_with_residual = self._add_residual(text_using_img_ctx, text_query)
        img_ctx_with_residual = self._add_residual(img_using_text_ctx, img_proj)

        text_ctx_final = text_ctx_with_residual
        img_ctx_final = img_ctx_with_residual

        aggregated_feature = None # Default to None
        if self.fusion_mode == "full" or self.fusion_mode == "concat":
            if self.fusion_layer is None:
                 raise RuntimeError(f"Fusion mode '{self.fusion_mode}' requires fusion_layer, but it's None.")
            text_pool = text_ctx_final.mean(dim=1) # [B, D]
            img_pool = img_ctx_final.mean(dim=1)   # [B, D]
            aggregated_feature = self.fusion_layer(torch.cat([text_pool, img_pool], dim=-1)) # [B, D]

        elif self.fusion_mode == "weighted":
            weights = F.softmax(self.fusion_weights, dim=0)
            text_pool = text_ctx_final.mean(dim=1)
            img_pool = img_ctx_final.mean(dim=1)
            aggregated_feature = weights[0] * text_pool + weights[1] * img_pool # [B, D]

        elif self.fusion_mode == "attention":
            if self.fusion_attention_proj is None:
                 raise RuntimeError("Fusion mode 'attention' requires fusion_attention_proj, but it's None.")
            text_pool = text_ctx_final.mean(dim=1) # (B, D)
            img_pool = img_ctx_final.mean(dim=1) # (B, D)
            t2i_score = self.fusion_attention_proj(text_pool) # (B, 1)
            i2t_score = self.fusion_attention_proj(img_pool) # (B, 1)
            scores = torch.cat([t2i_score, i2t_score], dim=-1) # (B, 2)
            attn_w = F.softmax(scores, dim=-1) # (B, 2)
            aggregated_feature = attn_w[:, 0].unsqueeze(-1) * text_pool + attn_w[:, 1].unsqueeze(-1) * img_pool # [B, D]

        if aggregated_feature is not None and self.res_proj is not None:
             res_base_text = text_feats_proj.mean(1, keepdim=True) # [B, 1, D]
             res_base_img = img_proj.mean(1, keepdim=True)      # [B, 1, D]
             residual_base = (res_base_img + res_base_text) / 2.0 # [B, 1, D]
             projected_residual = self.res_proj(residual_base)    # [B, 1, D]

             if aggregated_feature.dim() == 2:
                  aggregated_feature = aggregated_feature.unsqueeze(1) # -> [B, 1, D]

             if aggregated_feature.shape == projected_residual.shape:
                 aggregated_feature = self._add_residual(aggregated_feature, projected_residual)
             else:
                  print(f"Warning: Final residual shape mismatch agg:{aggregated_feature.shape} vs res:{projected_residual.shape}. Skipping.")

        return {
            "text_to_img": text_ctx_with_residual, # Shape [B, num_ts + L_t, D]
            "img_to_text": img_ctx_with_residual, # Shape [B, L_i, D]
            "aggregated_feature": aggregated_feature, # Shape [B, D] or [B, 1, D] or None
            "layer_idx": self.layer_idx,
        }