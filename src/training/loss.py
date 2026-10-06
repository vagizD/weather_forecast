"""
Custom Loss Functions for Weather Time-Series Forecasting.
Combines point fidelity, temporal derivative preservation, and diurnal extrema penalization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class StudentTLoss(nn.Module):
    """
    Negative Log-Likelihood Loss for Student-t Distributed Residuals:
    p(e) ~ t_nu(0, sigma)
    -log p(e) = ((nu + 1) / 2) * log(1 + e^2 / (nu * sigma^2)) + const
    """
    def __init__(self, nu: float = 8.44, sigma: float = 4.38):
        super().__init__()
        self.nu = nu
        self.sigma_sq = sigma ** 2
        self.scale = (nu + 1) / 2.0

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        err_sq = (pred - target) ** 2
        return torch.mean(self.scale * torch.log(1.0 + err_sq / (self.nu * self.sigma_sq)))


class GEDLoss(nn.Module):
    """
    Negative Log-Likelihood Loss for Generalized Error Distribution (Subbotin):
    p(e) ~ exp(- |e/alpha|^beta)
    -log p(e) = |e/alpha|^beta + const
    """
    def __init__(self, beta: float = 1.51, eps: float = 1e-6):
        super().__init__()
        self.beta = beta
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        err = torch.abs(pred - target) + self.eps
        return torch.mean(err ** self.beta)


class CompositeWeatherLoss(nn.Module):
    """
    Composite Loss for Long-Horizon (336h) Temperature Forecasting:
      L_total = L_point + lambda_slope * L_slope + lambda_extrema * L_extrema
    
    Prevents the model from collapsing into an over-smoothed, flat curve in Week 2.
    """
    def __init__(
        self,
        lambda_slope: float = 0.3,
        lambda_extrema: float = 0.2,
        loss_type: str = "student_t",
        nu: float = 8.44,
        sigma: float = 4.38,
        beta: float = 1.51
    ):
        super().__init__()
        self.lambda_slope = lambda_slope
        self.lambda_extrema = lambda_extrema
        self.loss_type = loss_type

        if loss_type == "student_t":
            self.point_fn = StudentTLoss(nu=nu, sigma=sigma)
        elif loss_type == "ged":
            self.point_fn = GEDLoss(beta=beta)
        elif loss_type in ["laplace", "l1"]:
            self.point_fn = nn.L1Loss()
        elif loss_type == "huber":
            self.point_fn = nn.SmoothL1Loss(beta=1.0)
        else:
            self.point_fn = nn.MSELoss()

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """
        Args:
            y_pred: (Batch, Horizon) e.g. (B, 336)
            y_true: (Batch, Horizon) e.g. (B, 336)
        Returns:
            total_loss: scalar tensor
            metrics: dict of sub-losses for logging
        """
        # 1. Point Fidelity Loss (MLE or standard)
        l_point = self.point_fn(y_pred, y_true)

        # 2. Temporal Derivative (Slope) Loss
        # Enforces realistic hour-to-hour ramps (solar heating & night cooling)
        diff_pred = y_pred[:, 1:] - y_pred[:, :-1]
        diff_true = y_true[:, 1:] - y_true[:, :-1]
        l_slope = F.l1_loss(diff_pred, diff_true)

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
