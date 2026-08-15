import torch
import torch.nn as nn


class MeanPool(nn.Module):
    """Mean-pooling MIL baseline."""

    def __init__(self, in_dim=1024, hidden_dim=500):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
        )
        self.risk_head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(0)
        pooled = self.projector(x).mean(dim=1)
        out = self.risk_head(pooled)
        return out.squeeze(-1), pooled, None
