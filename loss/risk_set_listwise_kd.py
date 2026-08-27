"""Risk-set listwise knowledge distillation loss for survival analysis."""

import torch
import torch.nn.functional as F


def risk_set_listwise_kd(
    student_risk: torch.Tensor,
    teacher_risk: torch.Tensor,
    time: torch.Tensor,
    event: torch.Tensor,
    temperature: float = 2.0,
) -> torch.Tensor:
    """Distill teacher relative-risk distributions within Cox risk sets."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    student_risk = student_risk.reshape(-1)
    teacher_risk = teacher_risk.detach().reshape(-1)
    time = time.reshape(-1)
    event = event.reshape(-1)

    if not (
        len(student_risk)
        == len(teacher_risk)
        == len(time)
        == len(event)
    ):
        raise ValueError("student_risk, teacher_risk, time, and event must have the same length")

    losses = []
    for event_index in torch.where(event > 0)[0]:
        risk_set = time >= time[event_index]
        teacher_prob = torch.softmax(
            teacher_risk[risk_set] / temperature,
            dim=0,
        )
        student_log_prob = torch.log_softmax(
            student_risk[risk_set] / temperature,
            dim=0,
        )
        losses.append(
            F.kl_div(student_log_prob, teacher_prob, reduction="sum")
        )

    if not losses:
        return student_risk.sum() * 0.0

    return torch.stack(losses).mean() * temperature**2
