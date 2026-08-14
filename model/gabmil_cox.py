import torch
import torch.nn as nn
import torch.nn.functional as F


def _pool_topk(H, logits, k):
    if k is None or k <= 0 or H.size(1) <= k:
        weights = F.softmax(logits, dim=1)
        return torch.bmm(weights.transpose(1, 2), H).squeeze(1), weights
    topk_logits, topk_indices = torch.topk(logits.squeeze(-1), k, dim=1)
    gather_idx = topk_indices.unsqueeze(-1).expand(-1, -1, H.size(-1))
    H_topk = torch.gather(H, dim=1, index=gather_idx)
    topk_weights = F.softmax(topk_logits, dim=1).unsqueeze(-1)
    pooled = torch.bmm(topk_weights.transpose(1, 2), H_topk).squeeze(1)
    weights = torch.zeros_like(logits)
    weights.scatter_(1, topk_indices.unsqueeze(-1), topk_weights)
    return pooled, weights


class GatedABMIL(nn.Module):
    """Gated-attention MIL for survival analysis."""

    def __init__(self, in_dim=1024, hidden_dim=500, attention_dim=128,
                 k=None, n_bins=None):
        super().__init__()
        self.k = k
        self.n_bins = n_bins
        self.projector = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
        )
        self.attention_V = nn.Sequential(
            nn.Linear(hidden_dim, attention_dim),
            nn.Tanh(),
        )
        self.attention_U = nn.Sequential(
            nn.Linear(hidden_dim, attention_dim),
            nn.Sigmoid(),
        )
        self.attention_w = nn.Linear(attention_dim, 1)
        head_out = n_bins if n_bins is not None else 1
        self.risk_head = nn.Linear(hidden_dim, head_out)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(0)
        H = self.projector(x)
        logits = self.attention_w(self.attention_V(H) * self.attention_U(H))
        pooled, weights = _pool_topk(H, logits, self.k)
        out = self.risk_head(pooled)
        if self.n_bins is not None:
            hazards = torch.sigmoid(out)
            S = torch.cumprod(1 - hazards, dim=1)
            return hazards, S, pooled, weights
        return out.squeeze(-1), pooled, weights
