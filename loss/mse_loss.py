"""MSE loss for scalar survival-risk distillation."""

import torch
import torch.nn.functional as F


def mse_distill_loss(
    student_risk: torch.Tensor,
    teacher_risk: torch.Tensor,
) -> torch.Tensor:
    """Compute MSE between student and detached teacher risk predictions."""
    return F.mse_loss(student_risk, teacher_risk.detach())


def normalized_mse_distill_loss(
    student_risk: torch.Tensor,
    teacher_risk: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute MSE after batch-wise z-score normalization of risk values.

    Cox risk scores are only meaningful up to their scale, so this version
    focuses the distillation loss on the relative risk pattern.
    """
    student_risk = student_risk.reshape(-1)
    teacher_risk = teacher_risk.detach().reshape(-1)

    student_mean = student_risk.mean()
    teacher_mean = teacher_risk.mean()
    student_std = torch.sqrt(student_risk.var(unbiased=False) + eps)
    teacher_std = torch.sqrt(teacher_risk.var(unbiased=False) + eps)

    student_normalized = (student_risk - student_mean) / student_std
    teacher_normalized = (teacher_risk - teacher_mean) / teacher_std
    return F.mse_loss(student_normalized, teacher_normalized)
