import torch
import torch.nn as nn


class MeanPool(nn.Module):
    """Mean-pooling MIL baseline with a direct linear Cox risk head."""

    def __init__(self, in_dim=1024):
        super().__init__()
        self.risk_head = nn.Linear(in_dim, 1)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(0)
        pooled = x.mean(dim=1)
        out = self.risk_head(pooled)
        return out.squeeze(-1), pooled, None
