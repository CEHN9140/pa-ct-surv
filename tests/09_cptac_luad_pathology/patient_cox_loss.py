"""Experiment09: the same Breslow Cox likelihood averaged over actual patients."""
from loss import cox_loss


def cox_patient_loss(risk,time,event):
    # Reuse the original risk sets/ties; eps=0 avoids retaining an event divisor.
    # The original function already returns a differentiable zero for no events.
    return cox_loss(risk,time,event,eps=0.0)*event.reshape(-1).bool().sum()/risk.numel()
