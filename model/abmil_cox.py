import torch
import torch.nn as nn
import torch.nn.functional as F


class ABMIL(nn.Module):
    """Ilse-style attention MIL adapted to a Cox survival risk head."""

    def __init__(self, in_dim=1024, hidden_dim=500, attention_dim=128,
                 n_bins=None):
        super().__init__()
        self.n_bins = n_bins
        self.projector = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
        )
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1),
        )
        head_out = n_bins if n_bins is not None else 1
        self.risk_head = nn.Linear(hidden_dim, head_out)

    def _pool_attention(self, H, logits):
        weights = F.softmax(logits, dim=1)
        return torch.bmm(weights.transpose(1, 2), H).squeeze(1), weights

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(0)
        H = self.projector(x)
        logits = self.attention(H)
        pooled, weights = self._pool_attention(H, logits)
        out = self.risk_head(pooled)
        if self.n_bins is not None:
            hazards = torch.sigmoid(out)
            S = torch.cumprod(1 - hazards, dim=1)
            return hazards, S, pooled, weights
        return out.squeeze(-1), pooled, weights


class ABMIL_TopK(ABMIL):
    """ABMIL with attention pooling restricted to the top-k instances."""

    def __init__(self, in_dim=1024, hidden_dim=500, attention_dim=128,
                 k=None, n_bins=None):
        if k is None or k <= 0:
            raise ValueError("ABMIL_TopK requires a positive k")
        super().__init__(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            attention_dim=attention_dim,
            n_bins=n_bins,
        )
        self.k = k

    def _pool_attention(self, H, logits):
        if H.size(1) <= self.k:
            weights = F.softmax(logits, dim=1)
            return torch.bmm(weights.transpose(1, 2), H).squeeze(1), weights

        topk_logits, topk_indices = torch.topk(
            logits.squeeze(-1), self.k, dim=1
        )
        gather_idx = topk_indices.unsqueeze(-1).expand(
            -1, -1, H.size(-1)
        )
        H_topk = torch.gather(H, dim=1, index=gather_idx)
        topk_weights = F.softmax(topk_logits, dim=1).unsqueeze(-1)
        pooled = torch.bmm(
            topk_weights.transpose(1, 2), H_topk
        ).squeeze(1)

        weights = torch.zeros_like(logits)
        weights.scatter_(1, topk_indices.unsqueeze(-1), topk_weights)
        return pooled, weights


class ProjABMIL(nn.Module):
    """ABMIL with a learnable dimensionality-reduction head before the MIL
    projector.

    Reduces 1024-d UNI features to proj_dim via a trainable linear layer
    (or MLP), then feeds them into ABMIL. The projection is optimized jointly
    with the rest of the model, so it learns the most survival-relevant
    low-dim subspace rather than a fixed one.
    """

    def __init__(self, in_dim=1024, proj_dim=256, proj_type="linear",
                 hidden_dim=500, attention_dim=128, dropout=0.25, k=None,
                 n_bins=None):
        super().__init__()
        self.proj_dim = proj_dim

        if proj_type == "linear":
            self.proj = nn.Sequential(
                nn.Linear(in_dim, proj_dim),
                nn.LayerNorm(proj_dim),
                nn.ReLU(),
            )
        elif proj_type == "mlp":
            self.proj = nn.Sequential(
                nn.Linear(in_dim, proj_dim * 2),
                nn.LayerNorm(proj_dim * 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(proj_dim * 2, proj_dim),
                nn.LayerNorm(proj_dim),
                nn.ReLU(),
            )
        else:
            raise ValueError(f"Unknown proj_type: {proj_type}")

        if k is None:
            self.abmil = ABMIL(
                in_dim=proj_dim,
                hidden_dim=hidden_dim,
                attention_dim=attention_dim,
                n_bins=n_bins,
            )
        else:
            self.abmil = ABMIL_TopK(
                in_dim=proj_dim,
                hidden_dim=hidden_dim,
                attention_dim=attention_dim,
                k=k,
                n_bins=n_bins,
            )

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(0)
        x = self.proj(x)
        return self.abmil(x)
