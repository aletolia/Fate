from typing import List, Optional, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Sampler, Dataset
from config import TASK_CONFIG, TrainingFlags
from Pytorch_PCGrad.pcgrad import PCGrad

# ===============================================================================
# MGDA (Multiple Gradient Descent Algorithm) Implementation
# Source: https://papers.nips.cc/paper/2018/hash/432aca3a1e345e339f35a30c8f65edce-Abstract.html
# ===============================================================================

def _min_norm_element_from2(v1v1, v1v2, v2v2):
    if v1v2 >= v1v1:
        gamma = 0.999
        cost = v1v1
        return gamma, cost
    if v1v2 >= v2v2:
        gamma = 0.001
        cost = v2v2
        return gamma, cost
    gamma = -1.0 * ((v1v2 - v2v2) / (v1v1 + v2v2 - 2 * v1v2))
    cost = v2v2 + gamma * (v1v2 - v2v2)
    return gamma, cost

def _min_norm_2d(grad_mat):
    dmin = 1e8
    for i in range(grad_mat.size()[0]):
        for j in range(i + 1, grad_mat.size()[0]):
            c, d = _min_norm_element_from2(grad_mat[i, i], grad_mat[i, j], grad_mat[j, j])
            if d < dmin:
                dmin = d
                sol = [(i, j), c, d]
    return sol

def _projection2simplex(y):
    m = len(y)
    sorted_y = torch.sort(y, descending=True)[0]
    tmpsum = 0.0
    tmax_f = (torch.sum(y) - 1.0) / m
    for i in range(m - 1):
        tmpsum += sorted_y[i]
        tmax = (tmpsum - 1) / (i + 1.0)
        if tmax > sorted_y[i + 1]:
            tmax_f = tmax
            break
    return torch.max(y - tmax_f, torch.zeros(m).to(y.device))

def _next_point(cur_val, grad, n):
    proj_grad = grad - (torch.sum(grad) / n)
    tm1 = -1.0 * cur_val[proj_grad < 0] / proj_grad[proj_grad < 0]
    tm2 = (1.0 - cur_val[proj_grad > 0]) / (proj_grad[proj_grad > 0])

    t = torch.ones(1).to(grad.device)
    if (tm1 > 1e-7).sum() > 0:
        t = torch.min(t, torch.min(tm1[tm1 > 1e-7]))
    if (tm2 > 1e-7).sum() > 0:
        t = torch.min(t, torch.min(tm2[tm2 > 1e-7]))

    next_point = proj_grad * t + cur_val
    next_point = _projection2simplex(next_point)
    return next_point

def _find_min_norm_element(grads):
    MAX_ITER = 250
    STOP_CRIT = 1e-5

    grad_mat = grads.mm(grads.t())
    init_sol = _min_norm_2d(grad_mat)

    n = grads.size()[0]
    sol_vec = torch.zeros(n).to(grads.device)
    sol_vec[init_sol[0][0]] = init_sol[1]
    sol_vec[init_sol[0][1]] = 1 - init_sol[1]

    if n < 3:
        return sol_vec

    iter_count = 0
    while iter_count < MAX_ITER:
        grad_dir = -1.0 * torch.matmul(grad_mat, sol_vec)
        new_point = _next_point(sol_vec, grad_dir, n)
        v1v1 = torch.sum(sol_vec.unsqueeze(1) * sol_vec.unsqueeze(0) * grad_mat)
        v1v2 = torch.sum(sol_vec.unsqueeze(1) * new_point.unsqueeze(0) * grad_mat)
        v2v2 = torch.sum(new_point.unsqueeze(1) * new_point.unsqueeze(0) * grad_mat)

        nc, nd = _min_norm_element_from2(v1v1, v1v2, v2v2)
        new_sol_vec = nc * sol_vec + (1 - nc) * new_point
        change = new_sol_vec - sol_vec
        if torch.sum(torch.abs(change)) < STOP_CRIT:
            return sol_vec
        sol_vec = new_sol_vec
        iter_count += 1
    return sol_vec

class MGDALoss(nn.Module):
    def __init__(self, model, device, mgda_gn_type='none'):
        super(MGDALoss, self).__init__()
        self.model = model
        self.device = device
        self.mgda_gn_type = mgda_gn_type

    def _get_grads(self, losses):
        """
        Compute gradients of each task loss with respect to the shared representation.
        """
        shared_params = [p for p in self.model.parameters() if p.requires_grad]
        
        grads = []
        for i, loss in enumerate(losses):
            self.model.zero_grad()
            loss.backward(retain_graph=True)
            
            grad_vec = []
            for param in shared_params:
                if param.grad is not None:
                    grad_vec.append(param.grad.view(-1))
            grads.append(torch.cat(grad_vec))
        
        return torch.stack(grads)

    def _gradient_normalizers(self, grads, loss_data):
        if self.mgda_gn_type == 'l2':
            gn = grads.pow(2).sum(-1).sqrt()
        elif self.mgda_gn_type == 'loss':
            gn = loss_data
        elif self.mgda_gn_type == 'loss+':
            gn = loss_data * grads.pow(2).sum(-1).sqrt()
        elif self.mgda_gn_type == 'none':
            gn = torch.ones_like(loss_data).to(self.device)
        else:
            raise ValueError(f'Unsupported normalization type {self.mgda_gn_type}')
        
        # Add a small epsilon to avoid division by zero
        gn = gn + 1e-8
        
        grads = grads / gn.unsqueeze(1)
        return grads

    def forward(self, model_output, batch, task_criteria):
        task_losses = []
        task_losses_dict = {}

        for i, task_name in enumerate(TASK_CONFIG["names"]):
            logits = model_output["logits"][i].squeeze()
            labels = batch[TASK_CONFIG["label_keys"][i]].float()
            mask = (labels >= 0)
            
            if mask.any():
                loss_i = task_criteria[task_name](logits[mask], labels[mask])
                task_losses.append(loss_i)
                task_losses_dict[f"{task_name}_loss"] = loss_i.item()
            else:
                # Still append a zero tensor to maintain task count
                task_losses.append(torch.tensor(0.0, device=self.device, requires_grad=True))
                task_losses_dict[f"{task_name}_loss"] = 0.0
        
        task_losses_tensor = torch.stack(task_losses)
        
        # Get gradients of each task loss
        grads = self._get_grads(task_losses)
        
        # Normalize gradients
        loss_data = task_losses_tensor.detach().clone()
        grads = self._gradient_normalizers(grads, loss_data)
        
        # Find the optimal weights
        sol = _find_min_norm_element(grads)
        
        # Compute the weighted loss
        weighted_loss = torch.sum(sol * task_losses_tensor)
        
        return weighted_loss, task_losses_dict


# =================================
# ===== multi-head attention ======
# =================================
def multi_head_attention_forward(
    query: torch.Tensor,              
    key: torch.Tensor,                
    value: torch.Tensor,              
    num_heads: int,                   
    embed_dim: int,                   
    q_proj: nn.Module,                
    k_proj: nn.Module,                
    v_proj: nn.Module,                
    out_proj: nn.Module,              
    dropout_p: float = 0.0,           
    attn_mask: Optional[torch.Tensor] = None, 
    is_causal: bool = False           
    ) -> torch.Tensor:

    B, Tq, D = query.shape
    Tkv = key.shape[1]
    head_dim = embed_dim // num_heads

    q = q_proj(query)
    k = k_proj(key)
    v = v_proj(value)

    q = q.view(B, Tq, num_heads, head_dim).transpose(1, 2) 
    k = k.view(B, Tkv, num_heads, head_dim).transpose(1, 2) 
    v = v.view(B, Tkv, num_heads, head_dim).transpose(1, 2) 
    
    if attn_mask is not None and attn_mask.ndim == 2: 
         attn_mask = attn_mask.unsqueeze(1).unsqueeze(2) 

    attn_output = F.scaled_dot_product_attention(
        q, k, v,
        attn_mask=attn_mask, 
        dropout_p=dropout_p if dropout_p > 0.0 else 0.0,
        is_causal=is_causal
    ) 

    attn_output = attn_output.transpose(1, 2).contiguous().view(B, Tq, embed_dim) 
    output = out_proj(attn_output) 
    return output

class CrossAttentionLayer(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = dim
        self.num_heads = num_heads
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim)

        self.attn_dropout = dropout
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, kv: torch.Tensor, kv_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, Tq, D = query.shape
        Tkv = kv.shape[1]

        if Tkv == 0:
            if Tq == 0:
                 return torch.zeros_like(query) 
            return query

        query_norm = self.norm_q(query)
        kv_norm = self.norm_kv(kv)

        attn_output = multi_head_attention_forward(
            query=query_norm,
            key=kv_norm,
            value=kv_norm,
            num_heads=self.num_heads,
            embed_dim=self.embed_dim,
            q_proj=self.q_proj,
            k_proj=self.k_proj,
            v_proj=self.v_proj,
            out_proj=self.out_proj,
            dropout_p=self.attn_dropout,
            attn_mask=kv_mask, 
            is_causal=False
        )

        output = query + self.resid_dropout(attn_output)

        return output

# =================================
# ====== Fusion utils =============
# =================================
class PerceiverIOCompressor(nn.Module):
    def __init__(self, dim: int, num_latents: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.num_latents = num_latents
        self.latent_query = nn.Parameter(torch.randn(1, num_latents, dim))
        self.compress_attention = CrossAttentionLayer(dim, num_heads, dropout)
        
    def forward(self, x: torch.Tensor, x_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B = x.shape[0]
        
        latent_queries = self.latent_query.expand(B, -1, -1)
        compressed_latents = self.compress_attention(
            query=latent_queries,
            kv=x,
            kv_mask=x_mask
        )
        
        return compressed_latents 

class FiLMLayer(nn.Module):
    def __init__(self, target_dim: int, condition_dim: int, film_hidden_dim: Optional[int] = None):
        super().__init__()
        self.target_dim = target_dim
        self.condition_dim = condition_dim

        if film_hidden_dim is None:
            film_hidden_dim = target_dim 

        self.generator = nn.Sequential(
            nn.Linear(condition_dim, film_hidden_dim),
            nn.ReLU(),
            nn.Linear(film_hidden_dim, 2 * target_dim) 
        )

    def forward(self, target_features: torch.Tensor, condition_features: torch.Tensor) -> torch.Tensor:
        B, N, D_target = target_features.shape

        if condition_features.shape[1] != 1:
             if condition_features.shape[1] == 1 :
                condition_features = condition_features.squeeze(1) 
             else:
                 condition_features = condition_features.mean(dim=1) 

        B_cond, S_cond, D_cond = condition_features.shape
        if S_cond == 1:
            condition_features_reduced = condition_features.squeeze(1) # Handles (B, 1, D) input
        else:
            condition_features_reduced = condition_features.mean(dim=1) # Handles (B, S>1, D) input
        params = self.generator(condition_features_reduced) # Expects (B, D)
        gamma = params[:, :self.target_dim] 
        beta = params[:, self.target_dim:]   

        gamma = gamma.unsqueeze(1)
        beta = beta.unsqueeze(1)
        
        modulated_features = target_features * gamma + beta
        return modulated_features

class ProposedFusionModule(nn.Module):
    def __init__(self,
                 dim: int = 768,
                 num_latents: int = 512,         
                 compress_heads: int = 8,
                 film_hidden_dim: Optional[int] = None,
                 use_projection: bool = True,    
                 fusion_heads: int = 8,
                 dropout: float = 0.1):
        super().__init__()
        self.dim = dim
        self.num_latents = num_latents
        self.use_projection = use_projection
        self.compressor = PerceiverIOCompressor(dim, num_latents, compress_heads, dropout)
        self.film_modulator = FiLMLayer(dim, dim, film_hidden_dim)

        if self.use_projection:
            self.proj_latent = nn.Linear(dim, dim) 
            self.proj_r2i = nn.Linear(dim, dim)    
            self.norm_proj_latent = nn.LayerNorm(dim)
            self.norm_proj_r2i = nn.LayerNorm(dim)

        self.fusion_cross_attention = CrossAttentionLayer(dim, fusion_heads, dropout)
        self.final_pool = nn.Identity() 

    def forward(self,
                i2r_att: torch.Tensor,                
                r2i_att: torch.Tensor,                
                i2r_mask: Optional[torch.Tensor] = None 
               ) -> torch.Tensor:                     
        
        print(f"i2r_att shape: {i2r_att.shape}, r2i_att shape: {r2i_att.shape}")
        compressed_i2r = self.compressor(i2r_att, i2r_mask)
        print(f"compressed_i2r shape: {compressed_i2r.shape}")
        modulated_i2r = self.film_modulator(compressed_i2r, r2i_att)
        print(f"modulated_i2r shape: {modulated_i2r.shape}")
        query_token = r2i_att 
        kv_tokens = modulated_i2r 

        if self.use_projection:
            query_token = self.norm_proj_r2i(self.proj_r2i(query_token))
            kv_tokens = self.norm_proj_latent(self.proj_latent(kv_tokens))

        fused_representation = self.fusion_cross_attention(
            query=query_token,
            kv=kv_tokens,
            kv_mask=None 
        )

        final_output = fused_representation.squeeze(1)
        final_output = self.final_pool(final_output)

        return final_output
    
# 备用方案 1
# Pyramid Fusion Module
class MultiScaleFusionModule(nn.Module):
    """
    Implements fusion using multi-scale compressors to capture features at different resolutions.
    """
    def __init__(self,
                 dim: int = 768,
                 latent_scales: List[int] = [64, 128, 256], # A list of different num_latents
                 compress_heads: int = 8,
                 film_hidden_dim: Optional[int] = None,
                 fusion_heads: int = 8,
                 dropout: float = 0.1):
        super().__init__()
        
        self.compressors = nn.ModuleList(
            [PerceiverIOCompressor(dim, num_latents, compress_heads, dropout) for num_latents in latent_scales]
        )
        merged_dim = dim * len(latent_scales)
        
        self.merge_projection = nn.Sequential(
            nn.Linear(merged_dim, dim),
            nn.LayerNorm(dim)
        )
        
        self.film_modulator = FiLMLayer(dim, dim, film_hidden_dim)
        self.fusion_cross_attention = CrossAttentionLayer(dim, fusion_heads, dropout)

    def forward(self,
                i2r_att: torch.Tensor,
                r2i_att: torch.Tensor,
                i2r_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        
        # 1. Compress the input i2r_att at multiple scales in parallel
        multi_scale_latents = [compressor(i2r_att, i2r_mask) for compressor in self.compressors]
        
        # 2. Concatenate the results along the sequence dimension
        # Before: [(B, 64, D), (B, 128, D), (B, 256, D)]
        # After: (B, 64 + 128 + 256, D)
        concatenated_latents = torch.cat(multi_scale_latents, dim=1)
        
        # 3. Project the merged representation back to the original dimension
        # This step learns to effectively combine the different scales
        merged_representation = self.merge_projection(concatenated_latents)
        
        # 4. Modulate the multi-scale representation using FiLM
        modulated_representation = self.film_modulator(merged_representation, r2i_att)
        
        # 5. Final fusion using cross-attention
        fused_representation = self.fusion_cross_attention(
            query=r2i_att,
            kv=modulated_representation,
        )
        
        return fused_representation.squeeze(1)
# 备用方案 2
# token2token fusion
class TokenToTokenFiLMLayer(nn.Module):
    """
    Applies FiLM modulation in a token-to-token fashion using cross-attention.
    Each target token queries the generated modulation parameters.
    """
    def __init__(self, target_dim: int, condition_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.target_dim = target_dim
        
        # Generator now produces a sequence of modulation parameters
        self.generator = nn.Linear(condition_dim, 2 * target_dim)
        
        # Cross-attention to align target tokens with modulation parameters
        self.attention = CrossAttentionLayer(target_dim, num_heads, dropout)

    def forward(self, target_features: torch.Tensor, condition_features: torch.Tensor) -> torch.Tensor:
        # target_features: (B, N, D_target) -> The sequence to be modulated
        # condition_features: (B, S, D_cond) -> The sequence providing context
        
        # 1. Generate a sequence of modulation parameters (gammas and betas) from condition_features
        # The sequence length 'S' of the condition is preserved
        mod_params = self.generator(condition_features) # (B, S, 2 * D_target)
        gammas = mod_params[..., :self.target_dim]     # (B, S, D_target)
        betas = mod_params[..., self.target_dim:]      # (B, S, D_target)
        
        # 2. Use attention to create custom modulation for each target token
        # Query: Each token in the target sequence
        # Key/Value: The sequence of generated gammas and betas
        aligned_gammas = self.attention(query=target_features, kv=gammas)
        aligned_betas = self.attention(query=target_features, kv=betas)
        
        # 3. Apply the fine-grained, aligned modulation
        modulated_features = target_features * (1 + aligned_gammas) + aligned_betas
        return modulated_features

class TokenToTokenFusionModule(nn.Module):
    def __init__(self,
                 dim: int = 768,
                 num_latents: int = 512,
                 compress_heads: int = 8,
                 fusion_heads: int = 8,
                 dropout: float = 0.1):
        super().__init__()
        self.compressor = PerceiverIOCompressor(dim, num_latents, compress_heads, dropout)
        
        # Use the new, more powerful FiLM layer
        self.film_modulator = TokenToTokenFiLMLayer(dim, dim, fusion_heads, dropout)
        
        self.fusion_cross_attention = CrossAttentionLayer(dim, fusion_heads, dropout)

    def forward(self,
                i2r_att: torch.Tensor,
                r2i_att: torch.Tensor,
                i2r_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        
        # 1. Compress i2r_att (same as before)
        compressed_i2r = self.compressor(i2r_att, i2r_mask)
        
        # 2. Modulate using the token-to-token mechanism
        # Each token in `compressed_i2r` will query `r2i_att` to get its own gamma/beta
        modulated_i2r = self.film_modulator(
            target_features=compressed_i2r, 
            condition_features=r2i_att
        )
        
        # 3. Final fusion (same as before)
        fused_representation = self.fusion_cross_attention(
            query=r2i_att,
            kv=modulated_i2r,
        )
        
        return fused_representation.squeeze(1)
# ==========================
# ===== dataset utils ======
# ==========================
class ImbalancedDatasetSampler(Sampler):
    """Samples elements randomly from a given list of indices for an imbalanced dataset."""
    def __init__(self, dataset: Dataset, indices: List[int], label_key: str = "label"):
        self.indices = indices
        self.num_samples = len(indices)
        
        # Calculate class frequencies for the given indices
        label_freq = {}
        for i in indices:
            label = int(dataset[i][label_key])
            label_freq[label] = label_freq.get(label, 0) + 1 
        
        # Calculate weights as the inverse of class frequency
        weights = [1.0 / label_freq[int(dataset[i][label_key])] for i in self.indices]
        self.weights = torch.DoubleTensor(weights)

    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples

# Collate function
def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Custom collate function to handle batching of heterogeneous data."""
    processed = {}
    batch_size = len(batch)

    # Process image and text features
    if "image" in batch[0]:
        processed["image"] = torch.stack([b["image"] for b in batch])
    if "text_feat" in batch[0]:
        processed["text_feat"] = torch.stack([b["text_feat"] for b in batch])
    
    # Process missing modality flags
    processed["missing_modality"] = torch.stack([b["missing_modality"] for b in batch]) if "missing_modality" in batch[0] else torch.zeros(batch_size, dtype=torch.long)
    
    # Process file paths
    if "file_path" in batch[0]:
        processed["file_path"] = [b.get("file_path") for b in batch]

    # Process labels for each task, handling missing labels with -1
    for key in TASK_CONFIG["label_keys"]:
        arr = torch.full((batch_size,), -1, dtype=torch.long)
        for i, b in enumerate(batch):
            if key in b:
                arr[i] = int(b[key])
        processed[key] = arr

    return processed
# ============================
# ====== training utils ======
# ============================
class DWAKeeper:
    """Keeps track of losses for Dynamic Weight Averaging (DWA)."""
    def __init__(self, num_tasks: int, temperature: float = 2.0):
        self.num_tasks = num_tasks
        self.temperature = temperature
        self.prev_losses: Optional[torch.Tensor] = None

    def compute_weights(self, current_avg_losses: torch.Tensor) -> torch.Tensor:
        if self.prev_losses is None:
            weights = torch.ones(self.num_tasks, device=current_avg_losses.device)
        else:
            loss_ratio = current_avg_losses / (self.prev_losses.to(current_avg_losses.device) + 1e-8)
            exp_term = torch.exp(loss_ratio / self.temperature)
            weights = self.num_tasks * exp_term / (torch.sum(exp_term) + 1e-8)
        
        self.prev_losses = current_avg_losses.detach().clone()
        return weights

    def reset(self):
        self.prev_losses = None

# Optimizer Builder
def build_optimizer(model: nn.Module, flags: TrainingFlags) -> torch.optim.Optimizer:
    """Builds an AdamW optimizer, with optional PCGrad wrapper."""
    if flags.use_pcgrad:
        return PCGrad(
            torch.optim.AdamW(model.parameters(), lr=flags.learning_rate, weight_decay=flags.weight_decay),
            reduction='mean'
        )
    return torch.optim.AdamW(model.parameters(), lr=flags.learning_rate, weight_decay=flags.weight_decay)

# Early Stopping
class EarlyStopping:
    """Stops training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience: int = 5, delta: float = 0.0, verbose: bool = True):
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = float('inf')
        self.early_stop = False

    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_score - self.delta:
            self.best_score = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                print("--- Early stopping triggered ---")
                self.early_stop = True
        return self.early_stop

    def reset_state(self):
        self.counter = 0
        self.best_score = float('inf')
        self.early_stop = False