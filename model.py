import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Dict, Any
from MoE import HierarchicalTaskMoE as DeepseekMoE_TaskSpecificExperts
from attn import Fate
from utils import MultiScaleLatentQueryFusion

# MTAN from https://github.com/lorenmt/mtan
class TaskSpecificAttentionLayer(nn.Module):
    def __init__(self, dim: int, att_hidden_dim: Optional[int] = None):
        super().__init__()
        if att_hidden_dim is None:
            att_hidden_dim = dim // 2

        self.attention_net = nn.Sequential(
            nn.Linear(dim, att_hidden_dim),
            nn.ReLU(),
            nn.Linear(att_hidden_dim, dim),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.attention_net(x)

class MultiTaskModelWithPerTaskFusion(nn.Module):
    def __init__(
        self,
        num_layers: int = 1,
        img_dim: int = 768,
        text_dim: int = 768,
        hidden_dim: int = 768,
        num_tasks: int = 4,
        dropout: float = 0.3,
        residual_mode: str = "identity",
        fusion_mode: str = "identity",
        fusion_heads: int = 8,
        classification_type: str = "binary",
        use_uncertainty_weighting: bool = False,
        use_task_norm: bool = False,
        use_cross_attention: bool = True,
        latent_scales: List[int] = [64, 128, 256],
        config = None,
    ):
        super().__init__()
        
        # Parameters
        self.num_tasks = num_tasks
        self.classification_type = classification_type
        self.use_cross_attention = use_cross_attention
        self.hidden_dim = hidden_dim
        self.latent_scales = latent_scales
        expert_mode_attr = getattr(config, 'expert_mode', 'both')

        # Loss weighting
        self.use_uw = use_uncertainty_weighting
        if self.use_uw:
            self.log_sigma = nn.Parameter(torch.zeros(num_tasks))

        # Cross-Attention Layers
        self.cross_attn_layers = nn.ModuleList(
            [
                Fate(
                    img_dim=img_dim if i == 0 else hidden_dim,
                    text_dim=text_dim if i == 0 else hidden_dim,
                    dim=hidden_dim,
                    layer_idx=i, total_layers=num_layers,
                    residual_mode=residual_mode, fusion_mode=fusion_mode,
                    num_tasks=num_tasks, use_task_norm=False, # Disable TaskNorm in Fate
                    heads=fusion_heads, dropout=dropout, num_ts_tokens=0,
                    learn_ts_tokens=False, add_ts_tokens_in_first_half=False,
                )
                for i in range(num_layers)
            ]
        )
        # Fusion
        self.shared_fusion_module = MultiScaleLatentQueryFusion(
            dim=hidden_dim,
            latent_scales=self.latent_scales,
            fusion_heads=fusion_heads,
            dropout=dropout
        )

        # Task-specific normalization layer
        if use_task_norm:
            self.task_norm_layers = nn.ModuleList(
                [nn.LayerNorm(hidden_dim) for _ in range(self.num_tasks)]
            )
        else:
            self.task_norm_layers = None

        self.mtan_attention_layers = nn.ModuleList(
            [TaskSpecificAttentionLayer(dim=hidden_dim) for _ in range(self.num_tasks)]
        )

        # MoE Layers
        self.post_fusion_moe = DeepseekMoE_TaskSpecificExperts(
            config, num_tasks=self.num_tasks
        )
        
        classifier_in_dim = sum(self.latent_scales) * hidden_dim
        if classification_type == "all":
            self.task_classifiers = nn.ModuleList()
            num_binary_tasks = 1
            for _ in range(num_binary_tasks):
                self.task_classifiers.append(nn.Linear(classifier_in_dim, 1))
        else:
            out_dim = 1 if classification_type == "binary" else hidden_dim
            self.task_classifiers = nn.ModuleList([nn.Linear(classifier_in_dim, out_dim) for _ in range(num_tasks)])

        self.num_layers = num_layers
        self.last_output: Dict[str, Any] = {}

    def forward(
        self,
        img_features: torch.Tensor,
        text_feats: torch.Tensor,
        missing_modality: int = 0,
        return_all: bool = False,
    ) -> Dict[str, Any]:
        cur_img, cur_txt = img_features, text_feats
        final_attn_outputs: Optional[Dict[str, torch.Tensor]] = None

        # Caculate attention outputs
        if self.use_cross_attention:
            for i, block in enumerate(self.cross_attn_layers):
                attn_out = block(img_features=cur_img, text_feats=cur_txt)
                final_attn_outputs = attn_out
                img_context, text_context = attn_out.get('img_to_text'), attn_out.get('text_to_img')
                
                if i < self.num_layers - 1:
                    if img_context is not None: cur_img = img_context
                    if text_context is not None:
                        cur_txt = text_context.mean(dim=1) if text_context.dim() > 2 else text_context
        else:
            final_attn_outputs = {
                'img_to_text': cur_img,
                'text_to_img': cur_txt
            }
        
        task_feats: List[torch.Tensor] = []
        if final_attn_outputs is not None and \
           'img_to_text' in final_attn_outputs and \
           'text_to_img' in final_attn_outputs:
            
            i2r_outputs = final_attn_outputs['img_to_text']
            r2i_outputs = final_attn_outputs['text_to_img']

            # Fusion
            shared_fused_feature = self.shared_fusion_module(
                i2r_att=i2r_outputs,
                r2i_att=r2i_outputs
            )

            for t in range(self.num_tasks):
                # Apply Task-Specific Normalization
                if self.task_norm_layers is not None:
                    task_normed_feature = self.task_norm_layers[t](shared_fused_feature)
                else:
                    task_normed_feature = shared_fused_feature

                # MTAN now operates on the task-normed feature
                attention_mask = self.mtan_attention_layers[t](task_normed_feature)
                task_attended_feature = task_normed_feature * attention_mask
                
                # MoE
                expert_output = self.post_fusion_moe(task_attended_feature, task_id=t)
                task_specific_feature = task_attended_feature + expert_output
                task_feats.append(task_specific_feature)

        else:
            device = next(self.parameters()).device
            task_feats = [torch.zeros(1, self.hidden_dim, device=device) for _ in range(self.num_tasks)]
        
        logits: List[torch.Tensor] = []
        probs: List[torch.Tensor] = []

        for idx, clf in enumerate(self.task_classifiers):
            feat = task_feats[idx]
            feat_flattened = feat.view(feat.size(0), -1)
            logit = clf(feat_flattened)
            logits.append(logit)
            
            if self.classification_type == "binary" and logit.size(-1) == 1:
                probs.append(torch.sigmoid(logit))
            elif logit.size(-1) > 0:
                probs.append(F.softmax(logit, dim=-1))
            else:
                probs.append(torch.tensor([], device=logit.device))

        output = {"logits": logits, "probs": probs, "task_feats": task_feats}
        if return_all and final_attn_outputs is not None:
            output["attn_out"] = final_attn_outputs

        self.last_output = output
        return output

    def compute_loss(
        self,
        logits_list: List[torch.Tensor],
        labels_list: List[torch.Tensor],
        valid_mask_list: Optional[List[torch.Tensor]] = None,
        loss_fn=F.binary_cross_entropy_with_logits,
        epoch: int = 0,
        dwa_weights: Optional[List[float]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        task_losses = []
        for t, (logit, label) in enumerate(zip(logits_list, labels_list)):
            if logit is None or logit.numel() == 0:
                device = labels_list[t].device if labels_list[t] is not None else ('cuda' if torch.cuda.is_available() else 'cpu')
                task_losses.append(torch.tensor(0.0, device=device))
                continue

            effective_logit = logit
            effective_label = label.float().to(effective_logit.device)

            if loss_fn == F.binary_cross_entropy_with_logits and effective_logit.size(-1) == 1:
                if effective_label.dim() == 1:
                    effective_label = effective_label.unsqueeze(-1)
            
            if effective_logit.shape[0] != effective_label.shape[0]:
                task_losses.append(torch.tensor(0.0, device=effective_logit.device))
                continue
            
            l_unreduced = loss_fn(effective_logit, effective_label, reduction='none')

            if valid_mask_list is not None and len(valid_mask_list) > t and valid_mask_list[t] is not None:
                mask = valid_mask_list[t].to(effective_logit.device)
                if not mask.any():
                    task_losses.append(torch.tensor(0.0, device=effective_logit.device))
                    continue
                mask_expanded = mask.unsqueeze(-1).expand_as(l_unreduced) if l_unreduced.dim() > mask.dim() else mask
                l = (l_unreduced * mask_expanded).sum() / mask_expanded.sum().clamp(min=1e-8)
            else:
                l = l_unreduced.mean()
            
            task_losses.append(l)
        
        task_losses_tensor = torch.stack(task_losses)
        
        if self.use_uw:
            sigma_sq = torch.exp(self.log_sigma.to(task_losses_tensor.device) * 2)
            weighted_losses = task_losses_tensor / (2 * sigma_sq) + torch.log(sigma_sq.sqrt())
            total_loss = weighted_losses.sum()
        elif dwa_weights is not None:
            w = torch.tensor(dwa_weights, device=task_losses_tensor.device, dtype=task_losses_tensor.dtype)
            total_loss = (task_losses_tensor * w).sum()
        else:
            total_loss = task_losses_tensor.sum()

        return total_loss, task_losses_tensor