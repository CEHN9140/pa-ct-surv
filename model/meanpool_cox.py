import torch
import torch.nn as nn


class MeanPool(nn.Module):
    """Mean-pooling MIL baseline."""

    def __init__(self, in_dim=1024, hidden_dim=500, n_bins=None):
        super().__init__()
        self.n_bins = n_bins
        self.projector = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
        )
        head_out = n_bins if n_bins is not None else 1
        self.risk_head = nn.Linear(hidden_dim, head_out)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(0)
        pooled = self.projector(x).mean(dim=1)
        out = self.risk_head(pooled)
        if self.n_bins is not None:
            hazards = torch.sigmoid(out)
            S = torch.cumprod(1 - hazards, dim=1)
            return hazards, S, pooled, None
        return out.squeeze(-1), pooled, None
