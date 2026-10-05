"""
Custom Loss Functions for Weather Time-Series Forecasting.
Combines point fidelity, temporal derivative preservation, and diurnal extrema penalization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class CompositeWeatherLoss(nn.Module):
    """
    Composite Loss for Long-Horizon (336h) Temperature Forecasting:
      L_total = L_point + lambda_slope * L_slope + lambda_extrema * L_extrema
    
    Prevents the model from collapsing into an over-smoothed, flat curve in Week 2.
    """
    def __init__(self, lambda_slope: float = 0.5, lambda_extrema: float = 0.25, loss_type: str = "huber"):
        super().__init__()
        self.lambda_slope = lambda_slope
        self.lambda_extrema = lambda_extrema
        self.loss_type = loss_type

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """
        Args:
            y_pred: (Batch, Horizon) e.g. (B, 336)
            y_true: (Batch, Horizon) e.g. (B, 336)
        Returns:
            total_loss: scalar tensor
            metrics: dict of sub-losses for logging
        """
        # 1. Point Fidelity Loss
        if self.loss_type == "huber":
            l_point = F.smooth_l1_loss(y_pred, y_true, beta=1.0)
        else:
            l_point = F.mse_loss(y_pred, y_true)

        # 2. Temporal Derivative (Slope) Loss
        # Enforces realistic hour-to-hour ramps (solar heating & night cooling)
        diff_pred = y_pred[:, 1:] - y_pred[:, :-1]
        diff_true = y_true[:, 1:] - y_true[:, :-1]
        l_slope = F.mse_loss(diff_pred, diff_true)

        # 3. Diurnal Extrema (Peak/Trough) Loss
        # Reshape to (Batch, 14 days, 24 hours)
        B, H = y_pred.shape
        if H == 336:
            y_p_days = y_pred.view(B, 14, 24)
            y_t_days = y_true.view(B, 14, 24)

            max_p = torch.max(y_p_days, dim=-1).values
            max_t = torch.max(y_t_days, dim=-1).values
            min_p = torch.min(y_p_days, dim=-1).values
            min_t = torch.min(y_t_days, dim=-1).values

            l_max = F.mse_loss(max_p, max_t)
            l_min = F.mse_loss(min_p, min_t)
            l_extrema = 0.5 * (l_max + l_min)
        else:
            l_extrema = torch.tensor(0.0, device=y_pred.device)

        total_loss = l_point + self.lambda_slope * l_slope + self.lambda_extrema * l_extrema

        loss_dict = {
            "l_total": total_loss.item(),
            "l_point": l_point.item(),
            "l_slope": l_slope.item(),
            "l_extrema": l_extrema.item() if isinstance(l_extrema, torch.Tensor) else l_extrema
        }
        return total_loss, loss_dict
