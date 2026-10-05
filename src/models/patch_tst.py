"""
Patch Time Series Transformer (PatchTST) Model Suite:
1. PatchTST: Channel-Independent 24h diurnal patching with joint multi-horizon linear head.
2. PatchTSTWithTimeEmbedding: Integrates transferred decadal time embeddings from DenseNet.
3. PatchTSTRoPE: Slices patches with Rotary Positional Embeddings (RoPE).
4. ModularPatchTST: Flexible architecture supporting self-learned time embeddings,
   learned linear channel mixing, and future transformer self-attention.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.dense_cross_transformer import TimeEmbedding, PositionalEncoding

# =========================================================================
# 1. Rotary Positional Embeddings (RoPE) Components
# =========================================================================

def apply_rotary_pos_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1]
    x1 = x[..., :d // 2]
    x2 = x[..., d // 2:]
    x_rot = torch.cat([-x2, x1], dim=-1)
    return x * cos + x_rot * sin

class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 512, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int):
        return self.cos_cached[:, :, :seq_len, :], self.sin_cached[:, :, :seq_len, :]

class RoPEAttention(nn.Module):
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.15):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        assert self.head_dim * nhead == d_model, "d_model must be divisible by nhead"
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.rope = RotaryEmbedding(self.head_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        q = self.q_proj(x).view(B, S, self.nhead, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.nhead, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.nhead, self.head_dim).transpose(1, 2)
        
        cos, sin = self.rope(q, S)
        q = apply_rotary_pos_emb(q, cos, sin)
        k = apply_rotary_pos_emb(k, cos, sin)
        
        attn_out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout.p if self.training else 0.0
        )
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, D)
        return self.out_proj(attn_out)

class RoPETransformerEncoderLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float = 0.15):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = RoPEAttention(d_model, nhead, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x + self.dropout1(self.attn(self.norm1(x)))
        out = h + self.mlp(self.norm2(h))
        return out

# =========================================================================
# 2. PatchTST Baseline
# =========================================================================

class PatchTST(nn.Module):
    def __init__(
        self,
        history_len: int = 720,
        horizon_len: int = 336,
        patch_len: int = 24,
        stride: int = 12,
        num_features: int = 12,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.15
    ):
        super().__init__()
        self.history_len = history_len
        self.horizon_len = horizon_len
        self.patch_len = patch_len
        self.stride = stride
        self.num_features = num_features
        self.num_patches = (history_len - patch_len) // stride + 1
        
        self.patch_embedding = nn.Linear(patch_len, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=self.num_patches + 50)
        self.dropout = nn.Dropout(dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.head = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(self.num_patches * d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, horizon_len)
        )
        
        self.future_fusion = nn.Sequential(
            nn.Linear(6, 32),
            nn.GELU(),
            nn.Linear(32, 1)
        )

    def forward(
        self, 
        x_hist: torch.Tensor, 
        y_clim: torch.Tensor, 
        future_solar: torch.Tensor = None
    ) -> torch.Tensor:
        B, L, C = x_hist.shape
        x = x_hist.permute(0, 2, 1).contiguous().view(B * C, L)
        x_patches = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        x_emb = self.patch_embedding(x_patches)
        x_emb = self.pos_encoder(x_emb)
        x_emb = self.dropout(x_emb)
        
        x_out = self.transformer_encoder(x_emb)
        out = self.head(x_out)
        out = out.view(B, C, self.horizon_len)
        
        pred_res = out[:, 0, :] + 0.1 * torch.mean(out[:, 1:, :], dim=1)
        
        if future_solar is not None:
            solar_res = self.future_fusion(future_solar).squeeze(-1)
            pred_res = pred_res + solar_res
            
        return y_clim + pred_res

# =========================================================================
# 3. PatchTST + Transferred Time Embeddings (Champion Architecture)
# =========================================================================

class PatchTSTWithTimeEmbedding(nn.Module):
    def __init__(
        self,
        history_len: int = 720,
        horizon_len: int = 336,
        patch_len: int = 24,
        stride: int = 12,
        num_features: int = 12,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.15,
        time_embed_dim: int = 64
    ):
        super().__init__()
        self.history_len = history_len
        self.horizon_len = horizon_len
        self.patch_len = patch_len
        self.stride = stride
        self.num_features = num_features
        self.num_patches = (history_len - patch_len) // stride + 1
        
        self.patch_embedding = nn.Linear(patch_len, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=self.num_patches + 50)
        self.dropout = nn.Dropout(dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.head = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(self.num_patches * d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, horizon_len)
        )
        
        self.time_emb = TimeEmbedding(embed_dim=time_embed_dim)
        
        self.future_fusion = nn.Sequential(
            nn.Linear(time_embed_dim + 6, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
        nn.init.zeros_(self.future_fusion[-1].weight)
        nn.init.zeros_(self.future_fusion[-1].bias)

    def forward(
        self,
        x_hist: torch.Tensor,
        y_clim: torch.Tensor,
        future_solar: torch.Tensor,
        future_time: dict
    ) -> torch.Tensor:
        B, L, C = x_hist.shape
        x = x_hist.permute(0, 2, 1).contiguous().view(B * C, L)
        x_patches = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        x_emb = self.patch_embedding(x_patches)
        x_emb = self.pos_encoder(x_emb)
        x_emb = self.dropout(x_emb)
        
        x_out = self.transformer_encoder(x_emb)
        out = self.head(x_out)
        out = out.view(B, C, self.horizon_len)
        
        pred_res = out[:, 0, :] + 0.1 * torch.mean(out[:, 1:, :], dim=1)
        
        fut_t_emb = self.time_emb(
            future_time["hour"], future_time["month"], future_time["season"],
            future_time["day"], future_time["weekday"], future_time["year"]
        )
        
        fut_cond = torch.cat([fut_t_emb, future_solar], dim=-1)
        time_solar_res = self.future_fusion(fut_cond).squeeze(-1)
        
        return y_clim + pred_res + time_solar_res

# =========================================================================
# 4. PatchTST + RoPE + Transferred Time Embeddings
# =========================================================================

class PatchTSTRoPE(nn.Module):
    def __init__(
        self,
        history_len: int = 720,
        horizon_len: int = 336,
        patch_len: int = 24,
        stride: int = 12,
        num_features: int = 12,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.15,
        time_embed_dim: int = 64
    ):
        super().__init__()
        self.history_len = history_len
        self.horizon_len = horizon_len
        self.patch_len = patch_len
        self.stride = stride
        self.num_features = num_features
        self.num_patches = (history_len - patch_len) // stride + 1
        
        self.patch_embedding = nn.Linear(patch_len, d_model)
        self.dropout = nn.Dropout(dropout)
        
        self.layers = nn.ModuleList([
            RoPETransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        
        self.head = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(self.num_patches * d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, horizon_len)
        )
        
        self.time_emb = TimeEmbedding(embed_dim=time_embed_dim)
        
        self.future_fusion = nn.Sequential(
            nn.Linear(time_embed_dim + 6, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
        nn.init.zeros_(self.future_fusion[-1].weight)
        nn.init.zeros_(self.future_fusion[-1].bias)

    def forward(
        self,
        x_hist: torch.Tensor,
        y_clim: torch.Tensor,
        future_solar: torch.Tensor,
        future_time: dict
    ) -> torch.Tensor:
        B, L, C = x_hist.shape
        x = x_hist.permute(0, 2, 1).contiguous().view(B * C, L)
        x_patches = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        x_emb = self.patch_embedding(x_patches)
        x_emb = self.dropout(x_emb)
        
        for layer in self.layers:
            x_emb = layer(x_emb)
            
        out = self.head(x_emb)
        out = out.view(B, C, self.horizon_len)
        pred_res = out[:, 0, :] + 0.1 * torch.mean(out[:, 1:, :], dim=1)
        
        fut_t_emb = self.time_emb(
            future_time["hour"], future_time["month"], future_time["season"],
            future_time["day"], future_time["weekday"], future_time["year"]
        )
        fut_cond = torch.cat([fut_t_emb, future_solar], dim=-1)
        time_solar_res = self.future_fusion(fut_cond).squeeze(-1)
        
        return y_clim + pred_res + time_solar_res

# =========================================================================
# 5. ModularPatchTST (Progressive Experiments Suite)
# =========================================================================

class ModularPatchTST(nn.Module):
    def __init__(
        self,
        history_len: int = 720,
        horizon_len: int = 336,
        patch_len: int = 24,
        stride: int = 12,
        num_features: int = 12,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.15,
        time_embed_dim: int = 64,
        use_learned_channel_mixing: bool = False,
        use_future_transformer: bool = False,
    ):
        super().__init__()
        self.history_len = history_len
        self.horizon_len = horizon_len
        self.patch_len = patch_len
        self.stride = stride
        self.num_features = num_features
        self.use_learned_channel_mixing = use_learned_channel_mixing
        self.use_future_transformer = use_future_transformer
        self.num_patches = (history_len - patch_len) // stride + 1
        
        self.patch_embedding = nn.Linear(patch_len, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=self.num_patches + 50)
        self.dropout = nn.Dropout(dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.head = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(self.num_patches * d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, horizon_len)
        )
        
        if self.use_learned_channel_mixing:
            self.channel_mixer = nn.Linear(num_features, 1, bias=False)
            with torch.no_grad():
                self.channel_mixer.weight.zero_()
                self.channel_mixer.weight[0, 0] = 1.0
                
        self.time_emb = TimeEmbedding(embed_dim=time_embed_dim)
        
        cond_dim = time_embed_dim + 6
        if self.use_future_transformer:
            fut_layer = nn.TransformerEncoderLayer(
                d_model=cond_dim, nhead=2, dim_feedforward=128, dropout=dropout, activation="gelu", batch_first=True
            )
            self.future_transformer = nn.TransformerEncoder(fut_layer, num_layers=2)
            self.future_fusion = nn.Linear(cond_dim, 1)
        else:
            self.future_fusion = nn.Sequential(
                nn.Linear(cond_dim, 64),
                nn.LayerNorm(64),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(64, 1)
            )
        nn.init.zeros_(self.future_fusion[-1].weight if not self.use_future_transformer else self.future_fusion.weight)
        nn.init.zeros_(self.future_fusion[-1].bias if not self.use_future_transformer else self.future_fusion.bias)

    def forward(
        self,
        x_hist: torch.Tensor,
        y_clim: torch.Tensor,
        future_solar: torch.Tensor,
        future_time: dict
    ) -> torch.Tensor:
        B, L, C = x_hist.shape
        x = x_hist.permute(0, 2, 1).contiguous().view(B * C, L)
        x_patches = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        x_emb = self.patch_embedding(x_patches)
        x_emb = self.pos_encoder(x_emb)
        x_emb = self.dropout(x_emb)
        
        x_out = self.transformer_encoder(x_emb)
        out = self.head(x_out)
        out = out.view(B, C, self.horizon_len)
        
        if self.use_learned_channel_mixing:
            out_perm = out.permute(0, 2, 1)
            pred_res = self.channel_mixer(out_perm).squeeze(-1)
        else:
            pred_res = out[:, 0, :] + 0.1 * torch.mean(out[:, 1:, :], dim=1)
            
        fut_t_emb = self.time_emb(
            future_time["hour"], future_time["month"], future_time["season"],
            future_time["day"], future_time["weekday"], future_time["year"]
        )
        fut_cond = torch.cat([fut_t_emb, future_solar], dim=-1)
        
        if self.use_future_transformer:
            fut_trans = self.future_transformer(fut_cond)
            time_solar_res = self.future_fusion(fut_trans).squeeze(-1)
        else:
            time_solar_res = self.future_fusion(fut_cond).squeeze(-1)
            
        return y_clim + pred_res + time_solar_res
