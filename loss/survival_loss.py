"""Scalar-risk survival objectives; event=1, censoring=0, higher score=higher risk."""

import math

import torch
import torch.nn.functional as F


def cox_loss(risk, time, event, eps=1e-8):
    """Event-averaged negative Cox partial log-likelihood with Breslow ties."""
    risk, time, event = risk.view(-1), time.view(-1), event.view(-1).bool()
    if event.sum() == 0:
        return risk.sum() * 0.0

    loss = risk.new_tensor(0.0)
    for t in torch.unique(time[event]):
        event_mask = (time == t) & event
        risk_set = time >= t
        d_t = event_mask.float().sum()
        loss -= risk[event_mask].sum() - d_t * torch.logsumexp(risk[risk_set], dim=0)
    return loss / (event.float().sum() + eps)


def pairwise_ranking_loss(risk, time, event):
    """Mean logistic loss on Harrell-comparable pairs, including event/censor ties.

    An observed event is compared with later observations and same-time censored
    observations. Same-time event/event pairs and censor/censor pairs are excluded.
    No valid pairs produce zero with a valid gradient path.
    """
    risk, time, event = risk.reshape(-1), time.reshape(-1), event.reshape(-1).bool()
    if risk.numel() != time.numel() or risk.numel() != event.numel():
        raise ValueError("risk, time and event must contain the same number of patients")
    comparable = event[:, None] & (
        (time[:, None] < time[None, :])
        | ((time[:, None] == time[None, :]) & ~event[None, :])
    )
    if not comparable.any():
        return risk.sum() * 0.0
    differences = risk[None, :] - risk[:, None]
    return F.softplus(differences[comparable]).mean()


def cox_rank_loss(risk, time, event, rank_weight=1.0):
    """Cox + rank_weight * logistic pairwise loss; this is not DeepHit."""
    if not math.isfinite(rank_weight) or rank_weight < 0:
        raise ValueError("rank_weight must be finite and nonnegative")
    loss = cox_loss(risk, time, event)
    if rank_weight == 0:
        return loss
    return loss + rank_weight * pairwise_ranking_loss(risk, time, event)


def survival_loss(risk, time, event, loss_type="cox", rank_weight=1.0):
    """Shared dispatch for objectives that use one scalar risk per patient."""
    if loss_type == "cox":
        return cox_loss(risk, time, event)
    if loss_type == "cox_rank":
        return cox_rank_loss(risk, time, event, rank_weight)
    raise ValueError(f"Unsupported scalar-risk survival loss: {loss_type}")
