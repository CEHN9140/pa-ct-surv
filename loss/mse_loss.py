"""MSE loss for scalar survival-risk distillation."""

import torch
import torch.nn.functional as F


def mse_distill_loss(
    student_risk: torch.Tensor,
    teacher_risk: torch.Tensor,
) -> torch.Tensor:
    """Compute MSE between student and detached teacher risk predictions."""
    return F.mse_loss(student_risk, teacher_risk.detach())
