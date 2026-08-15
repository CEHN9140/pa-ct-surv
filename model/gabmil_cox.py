import torch
import torch.nn as nn
import torch.nn.functional as F


class GABMIL(nn.Module):
    """Gated attention MIL for survival analysis."""

    def __init__(self, in_dim=1024, hidden_dim=500, attention_dim=128):
        super().__init__()
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
        self.risk_head = nn.Linear(hidden_dim, 1)

    def _pool_attention(self, H, logits):
        weights = F.softmax(logits, dim=1)
        return torch.bmm(weights.transpose(1, 2), H).squeeze(1), weights

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(0)
        H = self.projector(x)
        logits = self.attention_w(self.attention_V(H) * self.attention_U(H))
        pooled, weights = self._pool_attention(H, logits)
        out = self.risk_head(pooled)
        return out.squeeze(-1), pooled, weights


class GABMIL_TopK(GABMIL):
    """GABMIL with attention pooling restricted to the top-k instances."""

    def __init__(self, in_dim=1024, hidden_dim=500, attention_dim=128, k=None):
        if k is None or k <= 0:
            raise ValueError("GABMIL_TopK requires a positive k")
        super().__init__(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            attention_dim=attention_dim,
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
