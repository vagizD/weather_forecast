"""
DenseCrossTransformer: Modern Deep Learning Architecture for 336-Hour Weather Forecasting.

Key Architectural Innovations:
1. Piecewise Linear Embeddings (PLE):
   - Quantile-based piecewise linear encoding for 14 physical continuous features
     (T, Td, P, dp3h, dp24h, dt24h, u_wind, v_wind, vpd, q_spec, theta, cloud, shortwave, solar_elevation).
   - Allows neural network to learn sharp physical non-linear thresholds (freezing, heat wave, frontal drops).
2. Categorical Time Embeddings:
   - Trainable embeddings for Hour (0-23), Month (1-12), Season (1-4), Day (1-31), Weekday (0-6),
     and Year (0-35, explicitly modeling decadal global warming trends).
3. DenseNet Feature Cross-Tower:
   - Dense residual MLP with skip-connections fusing PLE numerical representations and Time Embeddings
     into rich, grounded hourly tokens h_t in R^{d_model}.
4. Multi-Scale Temporal Transformer Backbone:
   - Self-attention across daily patched tokens capturing inter-day weather patterns.
5. Future Conditioning & Cross-Attention Head:
   - Conditioned on future time embeddings (including year) and astronomical solar geometry.
   - Future tokens cross-attend to historical day patches, predicting the hourly residual anomaly.
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class PiecewiseLinearEncoding(nn.Module):
    def __init__(self, bins: torch.Tensor, embed_dim: int):
        super().__init__()
        # bins: (num_features, num_bins)
        self.register_buffer("bins", bins)
        self.num_features, self.num_bins = bins.shape
        in_dim = self.num_features * (self.num_bins - 1)
        self.proj = nn.Sequential(
            nn.Linear(in_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, num_features)
        B, T, F = x.shape
        x_exp = x.unsqueeze(-1)  # (B, T, F, 1)
        b_left = self.bins[:, :-1]   # (F, Bins-1)
        b_right = self.bins[:, 1:]   # (F, Bins-1)
        b_width = b_right - b_left + 1e-6
        
        # Piecewise linear activations in [0, 1]
        v = torch.clamp((x_exp - b_left) / b_width, 0.0, 1.0)  # (B, T, F, Bins-1)
        v_flat = v.reshape(B, T, -1)
        return self.proj(v_flat)

class TimeEmbedding(nn.Module):
    def __init__(self, embed_dim: int = 64):
        super().__init__()
        d_sub = 16
        self.hour_embed = nn.Embedding(24, d_sub)
        self.month_embed = nn.Embedding(13, d_sub)
        self.season_embed = nn.Embedding(5, d_sub)
        self.day_embed = nn.Embedding(32, d_sub)
        self.weekday_embed = nn.Embedding(7, d_sub)
        # Year embedding (0 to 35) to capture decadal climate change and global warming trends
        self.year_embed = nn.Embedding(36, d_sub)
        
        total_dim = d_sub * 6  # 96
        self.proj = nn.Sequential(
            nn.Linear(total_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU()
        )

    def forward(self, hour: torch.Tensor, month: torch.Tensor, season: torch.Tensor, 
                day: torch.Tensor, weekday: torch.Tensor, year: torch.Tensor) -> torch.Tensor:
        # Each is (B, T)
        h = self.hour_embed(hour)
        m = self.month_embed(month)
        s = self.season_embed(season)
        d = self.day_embed(day)
        w = self.weekday_embed(weekday)
        y = self.year_embed(torch.clamp(year, 0, 35))
        
        concat = torch.cat([h, m, s, d, w, y], dim=-1)
        return self.proj(concat)

class DenseNetCrossTower(nn.Module):
    """Dense residual MLP cross-network fusing PLE numerical and categorical time embeddings."""
    def __init__(self, in_dim: int, d_model: int, dropout: float = 0.15):
        super().__init__()
        self.dense1 = nn.Sequential(
            nn.Linear(in_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.dense2 = nn.Sequential(
            nn.Linear(in_dim + d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.out_proj = nn.Sequential(
            nn.Linear(in_dim + d_model + d_model, d_model),
            nn.LayerNorm(d_model)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, in_dim)
        h1 = self.dense1(x)
        h2 = self.dense2(torch.cat([x, h1], dim=-1))
        out = self.out_proj(torch.cat([x, h1, h2], dim=-1))
        return out

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]

class DenseCrossTransformer(nn.Module):
    def __init__(
        self,
        bins: torch.Tensor,
        history_len: int = 720,
        horizon_len: int = 336,
        patch_len: int = 24,
        stride: int = 12,
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
        self.d_model = d_model
        
        # 1. Feature Encoders
        self.ple = PiecewiseLinearEncoding(bins, embed_dim=128)
        self.time_emb = TimeEmbedding(embed_dim=64)
        
        # 2. DenseNet Cross Tower: (128 + 64) -> d_model (128)
        self.cross_tower = DenseNetCrossTower(in_dim=192, d_model=d_model, dropout=dropout)
        
        # 3. Patching Configuration
        self.num_patches = (history_len - patch_len) // stride + 1
        self.patch_proj = nn.Linear(patch_len * d_model, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=self.num_patches + 50)
        self.dropout = nn.Dropout(dropout)
        
        # 4. Transformer Encoder Backbone
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 5. Future Embedding & Solar Encoder
        self.future_time_emb = TimeEmbedding(embed_dim=64)
        self.future_solar_proj = nn.Sequential(
            nn.Linear(2, 32),
            nn.GELU(),
            nn.Linear(32, 64)
        )
        self.future_dense = nn.Sequential(
            nn.Linear(128, d_model),
            nn.LayerNorm(d_model),
            nn.GELU()
        )
        
        # 6. Future Cross-Attention Head: Queries are future tokens, Keys/Values are history day patches
        self.cross_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True)
        self.step_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(
        self,
        x_cont: torch.Tensor,          # (B, 720, 14)
        x_time: dict,                  # dict of (B, 720) tensors: hour, month, season, day, weekday, year
        y_clim: torch.Tensor,          # (B, 336)
        future_time: dict,             # dict of (B, 336) tensors: hour, month, season, day, weekday, year
        future_solar: torch.Tensor     # (B, 336, 2): solar_elevation, extraterrestrial_flux
    ) -> torch.Tensor:
        B = x_cont.shape[0]
        
        # A. Low-level Embeddings for each of 720 hours
        ple_out = self.ple(x_cont)      # (B, 720, 128)
        time_out = self.time_emb(
            x_time["hour"], x_time["month"], x_time["season"],
            x_time["day"], x_time["weekday"], x_time["year"]
        )  # (B, 720, 64)
        
        fused = torch.cat([ple_out, time_out], dim=-1)  # (B, 720, 192)
        h_tokens = self.cross_tower(fused)              # (B, 720, d_model)
        
        # B. Temporal Patching: (B, 720, d_model) -> (B, NumPatches, patch_len * d_model)
        patches = h_tokens.unfold(dimension=1, size=self.patch_len, step=self.stride)
        patches = patches.permute(0, 1, 3, 2).reshape(B, self.num_patches, -1)
        
        patch_emb = self.patch_proj(patches)            # (B, NumPatches, d_model)
        patch_emb = self.pos_encoder(patch_emb)
        patch_emb = self.dropout(patch_emb)
        
        # Transformer Self-Attention over history patches
        enc_out = self.transformer_encoder(patch_emb)   # (B, NumPatches, d_model)
        
        # C. Future Horizon Embeddings (Time + Solar Physics for 336 hours)
        fut_time_emb = self.future_time_emb(
            future_time["hour"], future_time["month"], future_time["season"],
            future_time["day"], future_time["weekday"], future_time["year"]
        )  # (B, 336, 64)
        fut_solar_emb = self.future_solar_proj(future_solar)  # (B, 336, 64)
        fut_fused = torch.cat([fut_time_emb, fut_solar_emb], dim=-1)  # (B, 336, 128)
        fut_tokens = self.future_dense(fut_fused)                     # (B, 336, d_model)
        
        # D. Cross-Attention: Future tokens attend to history day patches
        cross_out, _ = self.cross_attn(query=fut_tokens, key=enc_out, value=enc_out)  # (B, 336, d_model)
        pred_anomaly = self.step_head(cross_out).squeeze(-1)                           # (B, 336)
        
        # E. Physical Climatology Addition
        y_final = y_clim + pred_anomaly
        return y_final

class RegularizedDenseCrossTransformer(nn.Module):
    """
    DenseCrossTransformer upgraded with:
    - 64-bin Piecewise Linear Encoding (PLE) (882 input dimensions)
    - Strong dropout (0.25) across PLE, Cross-Tower, and Transformer
    - Zero-initialized residual output head ensuring stable residual training
    """
    def __init__(
        self,
        bins: torch.Tensor,
        history_len: int = 720,
        horizon_len: int = 336,
        patch_len: int = 24,
        stride: int = 12,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.25
    ):
        super().__init__()
        self.history_len = history_len
        self.horizon_len = horizon_len
        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model
        
        # 1. 64-bin PLE with dropout
        self.register_buffer("bins", bins)
        self.num_features, self.num_bins = bins.shape
        in_dim = self.num_features * (self.num_bins - 1)
        self.ple_proj = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # 2. Time Embeddings
        self.time_emb = TimeEmbedding(embed_dim=64)
        
        # 3. DenseNet Cross Tower (128 + 64 -> d_model)
        self.cross_tower = DenseNetCrossTower(in_dim=192, d_model=d_model, dropout=dropout)
        
        # 4. Patching Configuration
        self.num_patches = (history_len - patch_len) // stride + 1
        self.patch_proj = nn.Sequential(
            nn.Linear(patch_len * d_model, d_model),
            nn.LayerNorm(d_model)
        )
        self.pos_encoder = PositionalEncoding(d_model, max_len=self.num_patches + 50)
        self.dropout = nn.Dropout(dropout)
        
        # 5. Transformer Encoder Backbone
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 6. Future Conditioning
        self.future_time_emb = TimeEmbedding(embed_dim=64)
        self.future_solar_proj = nn.Sequential(
            nn.Linear(2, 32),
            nn.GELU(),
            nn.Linear(32, 64)
        )
        self.future_dense = nn.Sequential(
            nn.Linear(128, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # 7. Cross Attention Head
        self.cross_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True)
        self.step_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
        
        # ZERO INITIALIZE the final linear layer
        nn.init.zeros_(self.step_head[-1].weight)
        nn.init.zeros_(self.step_head[-1].bias)

    def ple_encode(self, x: torch.Tensor) -> torch.Tensor:
        B, T, F = x.shape
        x_exp = x.unsqueeze(-1)
        b_left = self.bins[:, :-1]
        b_right = self.bins[:, 1:]
        b_width = b_right - b_left + 1e-6
        v = torch.clamp((x_exp - b_left) / b_width, 0.0, 1.0)
        v_flat = v.reshape(B, T, -1)
        return self.ple_proj(v_flat)

    def forward(
        self,
        x_cont: torch.Tensor,
        x_time: dict,
        y_clim: torch.Tensor,
        future_time: dict,
        future_solar: torch.Tensor
    ) -> torch.Tensor:
        B = x_cont.shape[0]
        ple_out = self.ple_encode(x_cont)
        time_out = self.time_emb(
            x_time["hour"], x_time["month"], x_time["season"],
            x_time["day"], x_time["weekday"], x_time["year"]
        )
        fused = torch.cat([ple_out, time_out], dim=-1)
        h_tokens = self.cross_tower(fused)
        
        patches = h_tokens.unfold(dimension=1, size=self.patch_len, step=self.stride)
        patches = patches.permute(0, 1, 3, 2).reshape(B, self.num_patches, -1)
        
        patch_emb = self.patch_proj(patches)
        patch_emb = self.pos_encoder(patch_emb)
        patch_emb = self.dropout(patch_emb)
        
        enc_out = self.transformer_encoder(patch_emb)
        
        fut_time_emb = self.future_time_emb(
            future_time["hour"], future_time["month"], future_time["season"],
            future_time["day"], future_time["weekday"], future_time["year"]
        )
        fut_solar_emb = self.future_solar_proj(future_solar)
        fut_fused = torch.cat([fut_time_emb, fut_solar_emb], dim=-1)
        fut_tokens = self.future_dense(fut_fused)
        
        cross_out, _ = self.cross_attn(query=fut_tokens, key=enc_out, value=enc_out)
        pred_anomaly = self.step_head(cross_out).squeeze(-1)
        
        return y_clim + pred_anomaly

