import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Sequence, Tuple, Dict, Any
from MoE import DeepseekMoE_TaskSpecificExperts
from attn import Fate
from utils import ProposedFusionModule

class MultiTaskModelWithPerTaskFusion(nn.Module):
    def __init__(
        self,
        num_layers: int = 1,
        img_dim: int = 768,
        text_dim: int = 768,
        hidden_dim: int = 768,
        num_tasks: int = 4,
        dropout: float = 0.3,
        prompt_length: int = 32,
        prompt_type: str = "input",
        learnt_p: bool = True,
        residual_mode: str = "identity",
        fusion_mode: str = "identity",
        num_latents: int = 512,
        compress_heads: int = 8,
        film_hidden_dim: Optional[int] = None,
        use_projection: bool = True,
        fusion_heads: int = 8,
        classification_type: str = "binary",
        use_uncertainty_weighting: bool = False,
        use_task_norm: bool = False,
        config = None,
    ):
        super().__init__()
        
        # Parameters
        self.num_tasks = num_tasks
        self.classification_type = classification_type
        self.hidden_dim = hidden_dim
        self.original_fusion_mode = fusion_mode

        self.use_uw = use_uncertainty_weighting
        if self.use_uw:
            self.log_sigma = nn.Parameter(torch.zeros(num_tasks))

        self.apply_moe_post_attn = (self.original_fusion_mode == "moe")
        self.report2wsi_moe = None
        self.wsi2report_moe = None

        # Initialize MoE layers
        expert_mode_attr = getattr(config, 'expert_mode', 'both')
        self.report2wsi_moe = DeepseekMoE_TaskSpecificExperts(
            config, num_tasks=self.num_tasks, expert_mode=expert_mode_attr
        )
        self.wsi2report_moe = DeepseekMoE_TaskSpecificExperts(
            config, num_tasks=self.num_tasks, expert_mode=expert_mode_attr
        )
        print(f"Initialized Post-Attention MoE Layers for {self.num_tasks} tasks.")

        cross_attn_fusion_mode = "identity" if self.apply_moe_post_attn else self.original_fusion_mode

        self.cross_attn_layers = nn.ModuleList(
            [
                Fate(
                    img_dim=img_dim if i == 0 else hidden_dim,
                    text_dim=text_dim if i == 0 else hidden_dim,
                    dim=hidden_dim,
                    layer_idx=i,
                    total_layers=num_layers,
                    residual_mode=residual_mode,
                    fusion_mode=cross_attn_fusion_mode,
                    num_tasks=num_tasks,
                    use_task_norm=use_task_norm,
                    heads=fusion_heads,
                    dropout=dropout,
                    num_ts_tokens=0,
                    learn_ts_tokens=False,
                    add_ts_tokens_in_first_half=False,
                )
                for i in range(num_layers)
            ]
        )

        self.task_fusion_modules = nn.ModuleList(
            [
                ProposedFusionModule(
                    dim=hidden_dim,
                    num_latents=num_latents,
                    compress_heads=compress_heads,
                    film_hidden_dim=film_hidden_dim,
                    use_projection=use_projection,
                    fusion_heads=fusion_heads,
                    dropout=dropout
                )
                for _ in range(num_tasks)
            ]
        )

        if classification_type == "all":
             self.task_classifiers = nn.ModuleList()
             num_binary_tasks = 1
             for _ in range(num_binary_tasks):
                 self.task_classifiers.append(nn.Linear(hidden_dim, 1))
        else:
            out_dim = 1 if classification_type == "binary" else hidden_dim # Adjust if other classification types exist
            self.task_classifiers = nn.ModuleList([nn.Linear(hidden_dim, out_dim) for _ in range(num_tasks)])

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

        for i, block in enumerate(self.cross_attn_layers):
            attn_out = block(img_features=cur_img, text_feats=cur_txt)
            final_attn_outputs = attn_out

            img_context = attn_out.get('img_to_text')
            text_context = attn_out.get('text_to_img')

            if i < self.num_layers - 1:
                 if img_context is not None:
                      cur_img = img_context
                 if text_context is not None:
                      if text_context.dim() > 2 and text_context.shape[1] > 0:
                           cur_txt = text_context.mean(dim=1)
                      elif text_context.dim() == 2:
                           cur_txt = text_context

        task_fused_features: List[torch.Tensor] = []
        if final_attn_outputs is not None and \
           'img_to_text' in final_attn_outputs and \
           'text_to_img' in final_attn_outputs:

            i2r_outputs_pre_moe = final_attn_outputs['img_to_text']
            r2i_outputs_pre_moe = final_attn_outputs['text_to_img']
            print(f"i2r_outputs_pre_moe shape: {i2r_outputs_pre_moe.shape}, r2i_outputs_pre_moe shape: {r2i_outputs_pre_moe.shape}")
            current_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            for t in range(self.num_tasks):

                if True:
                    i2r_input_for_fusion = self.wsi2report_moe(i2r_outputs_pre_moe, t)
                    r2i_input_for_fusion = self.report2wsi_moe(r2i_outputs_pre_moe, t)
                    print(f"i2r_input_for_fusion shape: {i2r_input_for_fusion.shape}, r2i_input_for_fusion shape: {r2i_input_for_fusion.shape}")

                    i2r_input_for_fusion = i2r_input_for_fusion + i2r_outputs_pre_moe
                    r2i_input_for_fusion = r2i_input_for_fusion + r2i_outputs_pre_moe

                fused_feature_t = self.task_fusion_modules[t](
                    i2r_input_for_fusion,
                    r2i_input_for_fusion
                )

                if fused_feature_t.shape != (1, self.hidden_dim):
                     print(f"Warning: Task {t} fusion output shape {fused_feature_t.shape} unexpected. Trying to adapt.")
                     if fused_feature_t.numel() == self.hidden_dim:
                          fused_feature_t = fused_feature_t.view(1, self.hidden_dim)
                     else:
                          fused_feature_t = torch.zeros(1, self.hidden_dim, device=current_device)

                task_fused_features.append(fused_feature_t) # Append [1, Dim] tensor

        else:
             device = next(self.parameters()).device
             task_fused_features = [torch.zeros(1, self.hidden_dim, device=device)
                                    for _ in range(self.num_tasks)]

        logits: List[torch.Tensor] = []
        probs: List[torch.Tensor] = []
        task_feats: List[torch.Tensor] = task_fused_features

        if len(task_feats) != self.num_tasks:
            device = img_features.device
            while len(task_feats) < self.num_tasks:
                  task_feats.append(torch.zeros(1, self.hidden_dim, device=device))

        for idx, clf in enumerate(self.task_classifiers):
            feat = task_feats[idx] # Should be [1, hidden_dim] now
            if feat.dim() != 2 or feat.shape[0] != 1 or feat.shape[1] != self.hidden_dim:
                 # Add more robust error handling or default tensor
                 print(f"ERROR: Task {idx} feature has unexpected shape {feat.shape} before classifier.")
                 feat = torch.zeros(1, self.hidden_dim, device=feat.device) # Default to zeros

            logit = clf(feat)
            logits.append(logit)

            if self.classification_type == "binary" and logit.size(-1) == 1:
                probs.append(torch.sigmoid(logit))
            elif logit.size(-1) > 0:
                probs.append(F.softmax(logit, dim=-1))
            else:
                 probs.append(torch.tensor([], device=logit.device))

        output = {"logits": logits, "probs": probs, "task_feats": task_feats}

        if return_all and final_attn_outputs is not None:
            output["attn_out_pre_moe"] = final_attn_outputs

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

            if effective_logit.shape[0] != effective_label.shape[0] or effective_logit.shape[-1] != effective_label.shape[-1]:
                 print(f"Task {t} shape mismatch: logit {effective_logit.shape}, label {effective_label.shape}. Skipping loss.")
                 task_losses.append(torch.tensor(0.0, device=effective_logit.device))
                 continue

            apply_mask = False
            reduction = 'mean'
            if valid_mask_list is not None and len(valid_mask_list) > t and valid_mask_list[t] is not None:
                 mask = valid_mask_list[t].to(effective_logit.device)
                 if not mask.any():
                     task_losses.append(torch.tensor(0.0, device=effective_logit.device))
                     continue
                 reduction='none'
                 apply_mask = True

            l_unreduced = loss_fn(effective_logit, effective_label, reduction=reduction)

            if apply_mask:
                 mask_expanded = mask
                 if l_unreduced.dim() > mask.dim(): # e.g., loss is [B, C], mask is [B]
                      mask_expanded = mask.unsqueeze(-1).expand_as(l_unreduced)
                 l = (l_unreduced * mask_expanded).sum() / mask_expanded.sum().clamp(min=1e-8) # Masked mean
            else:
                 l = l_unreduced # Already mean if reduction was 'mean'

            task_losses.append(l)

        if len(task_losses) != self.num_tasks:
             print(f"Warning: Number of computed losses {len(task_losses)} != num_tasks {self.num_tasks}. Padding with zeros.")
             device = task_losses[0].device if task_losses else ('cuda' if torch.cuda.is_available() else 'cpu')
             while len(task_losses) < self.num_tasks:
                   task_losses.append(torch.tensor(0.0, device=device))

        task_losses_tensor = torch.stack(task_losses)

        if self.use_uw:
            log_sigma_dev = self.log_sigma.to(task_losses_tensor.device)
            sigma_sq = torch.exp(log_sigma_dev * 2)
            weighted_losses = task_losses_tensor / (2 * sigma_sq + 1e-8)
            regularization = torch.log(sigma_sq.sqrt() + 1e-8).sum()
            total_loss = weighted_losses.sum() + regularization
        elif dwa_weights is not None:
            w = torch.tensor(dwa_weights, device=task_losses_tensor.device, dtype=task_losses_tensor.dtype)
            if w.shape != task_losses_tensor.shape:
                 raise ValueError(f"DWA weights shape {w.shape} mismatch task losses shape {task_losses_tensor.shape}")
            total_loss = (task_losses_tensor * w).sum()
        else:
            total_loss = task_losses_tensor.sum() # Simple sum if no weighting

        return total_loss, task_losses_tensor