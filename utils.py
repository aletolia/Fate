from typing import List, Optional, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Sampler, Dataset
from config import TASK_CONFIG, TrainingFlags
from Pytorch_PCGrad.pcgrad import PCGrad

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
    if head_dim * num_heads != embed_dim:
        raise ValueError("embed_dim 必须能被 num_heads 整除")

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