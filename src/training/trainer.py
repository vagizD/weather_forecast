"""
Training and evaluation loops for baseline models, PatchTST, and DenseCrossTransformer.
"""

from typing import Dict, List, Optional
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.config import DEVICE
from src.training.loss import CompositeWeatherLoss

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    model_type: str = "pt",
    device: torch.device = DEVICE,
    clip_grad: float = 1.0
) -> float:
    """Train a neural network model for a single epoch."""
    model.train()
    total_loss = 0.0
    batches = 0
    
    for b in loader:
        optimizer.zero_grad()
        y_t = b["y_true"].to(device)
        y_c = b["y_clim"].to(device)
        
        if model_type == "pt":
            pt_h = b["pt_hist"].to(device)
            pt_s = b["pt_solar_fut"].to(device)
            pred = model(pt_h, y_c, pt_s)
        elif model_type == "pt_time":
            pt_h = b["pt_hist"].to(device)
            pt_s = b["pt_solar_fut"].to(device)
            fut_t = {k: v.to(device) for k, v in b["future_time"].items()}
            pred = model(pt_h, y_c, pt_s, fut_t)
        elif model_type == "dct":
            xc = b["x_cont"].to(device)
            fut_s = b["future_solar"].to(device)
            xt = {k: v.to(device) for k, v in b["x_time"].items()}
            fut_t = {k: v.to(device) for k, v in b["future_time"].items()}
            pred = model(xc, xt, y_c, fut_t, fut_s)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
            
        loss, _ = loss_fn(pred, y_t)
        loss.backward()
        if clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)
        optimizer.step()
        
        total_loss += loss.item()
        batches += 1
        
    return total_loss / max(1, batches)

def evaluate_windows(
    predict_fn,
    df: pd.DataFrame,
    windows: List[Dict],
    history_len: int = 168
) -> Dict[str, float]:
    """Evaluate a predictor function across rolling windows."""
    maes = []
    rmses = []
    for w in windows:
        curr_start = w["start"]
        curr_end = w["end"]
        target_df = df[(df["time"] >= curr_start) & (df["time"] < curr_end)]
        if len(target_df) != 336:
            continue
        y_true = target_df["target_tmpf"].values
        timestamps = pd.to_datetime(target_df["time"].values)
        
        y_pred = predict_fn(curr_start, timestamps)
        
        mae = float(np.mean(np.abs(y_true - y_pred)))
        rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        maes.append(mae)
        rmses.append(rmse)
        
    return {
        "mean_mae": float(np.mean(maes)),
        "std_mae": float(np.std(maes)),
        "mean_rmse": float(np.mean(rmses)),
        "std_rmse": float(np.std(rmses)),
    }
