"""
Knowledge Distillation losses adapted for survival analysis.

Reference: "Distilling the Knowledge in a Neural Network" (Hinton et al., 2015)
Adaptation: scalar risk predictions are converted to 2-class logits for KL distillation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DistillKL(nn.Module):
    """KL-divergence distillation loss (Hinton 2015).

    Args:
        T: temperature for softening the logits.
    """

    def __init__(self, T: float = 2.0):
        super().__init__()
        self.T = T

    def forward(self, y_s: torch.Tensor, y_t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            y_s: student logits [B, C]
            y_t: teacher logits [B, C]
        Returns:
            scalar KL divergence loss
        """
        p_s = F.log_softmax(y_s / self.T, dim=1)
        p_t = F.softmax(y_t / self.T, dim=1)
        loss = F.kl_div(p_s, p_t, reduction="batchmean") * (self.T**2)
        return loss


def risk_distill_loss(
    student_risk: torch.Tensor,
    teacher_risk: torch.Tensor,
    T: float = 2.0,
) -> torch.Tensor:
    """Distill a scalar risk prediction via KL divergence.

    Converts scalar risk -> 2D logits: [risk, -risk] and applies KL.

    Args:
        student_risk: student risk predictions [B]
        teacher_risk: teacher risk predictions [B] (detached internally)
        T: temperature

    Returns:
        scalar KL distillation loss
    """
    stu = torch.stack([student_risk / T, -student_risk / T], dim=1)
    tea = torch.stack([teacher_risk.detach() / T, -teacher_risk.detach() / T], dim=1)
    loss = F.kl_div(
        F.log_softmax(stu, dim=1),
        F.softmax(tea, dim=1),
        reduction="batchmean",
    ) * (T**2)
    return loss


def mse_distill_loss(
    student_risk: torch.Tensor,
    teacher_risk: torch.Tensor,
) -> torch.Tensor:
    """Direct MSE distillation on scalar risk predictions.

    Simpler alternative to KL — no temperature or logit conversion needed.

    Args:
        student_risk: student risk predictions [B]
        teacher_risk: teacher risk predictions [B] (detached internally)

    Returns:
        scalar MSE loss
    """
    return F.mse_loss(student_risk, teacher_risk.detach())
