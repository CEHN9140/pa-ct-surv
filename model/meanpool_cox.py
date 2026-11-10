import torch
import torch.nn as nn


class MeanPool(nn.Module):
    """Mean-pool raw patch features before the patient-level risk head."""

    def __init__(self, in_dim=1024, dropout=0.0, n_bins=None):
        super().__init__()
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if n_bins is not None and n_bins <= 0:
            raise ValueError("n_bins must be positive when provided")
        self.n_bins = n_bins
        self.post_compression = nn.Sequential(
            nn.Linear(in_dim, 512), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(dropout),
        )
        self.risk_head = nn.Linear(256, n_bins if n_bins is not None else 1)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(0)
        pooled = self.post_compression(x.mean(dim=1))
        output = self.risk_head(pooled)
        if self.n_bins is None:
            return output.squeeze(-1), pooled, None
        hazards = torch.sigmoid(output)
        survival = torch.cumprod(1.0 - hazards, dim=1)
        return -survival.sum(dim=1), pooled, None, hazards, survival
