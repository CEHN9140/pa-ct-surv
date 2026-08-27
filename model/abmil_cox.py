import torch
import torch.nn as nn
import torch.nn.functional as F


class ABMIL(nn.Module):
    """Ilse-style attention MIL adapted to a Cox survival risk head."""

    def __init__(
        self,
        in_dim=1024,
        hidden_dim=512,
        attention_dim=128,
        dropout=0.0,
        attention_branches=1,
    ):
        super().__init__()
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if attention_branches <= 0:
            raise ValueError("attention_branches must be positive")
        self.attention_branches = attention_branches
        self.projector = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, attention_branches),
        )
        self.risk_head = nn.Linear(hidden_dim * attention_branches, 1)

    def pool_attention(self, H, logits):
        weights = F.softmax(logits.transpose(1, 2), dim=2)
        pooled = torch.bmm(weights, H)
        return pooled.reshape(H.size(0), -1), weights.transpose(1, 2)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(0)
        H = self.projector(x)
        logits = self.attention(H)
        pooled, weights = self.pool_attention(H, logits)
        out = self.risk_head(pooled)
        return out.squeeze(-1), pooled, weights


class ABMIL_TopK(ABMIL):
    """ABMIL with attention pooling restricted to the top-k instances."""

    def __init__(
        self,
        in_dim=1024,
        hidden_dim=512,
        attention_dim=128,
        k=None,
        dropout=0.0,
        attention_branches=1,
    ):
        if k is None or k <= 0:
            raise ValueError("ABMIL_TopK requires a positive k")
        super().__init__(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            attention_dim=attention_dim,
            dropout=dropout,
            attention_branches=attention_branches,
        )
        self.k = k

    def pool_attention(self, H, logits):
        if H.size(1) <= self.k:
            return super().pool_attention(H, logits)

        branch_logits = logits.transpose(1, 2)
        topk_logits, topk_indices = torch.topk(
            branch_logits, self.k, dim=2
        )
        topk_weights = F.softmax(topk_logits, dim=2)
        H_by_branch = H.unsqueeze(1).expand(
            -1, self.attention_branches, -1, -1
        )
        gather_idx = topk_indices.unsqueeze(-1).expand(
            -1, -1, -1, H.size(-1)
        )
        H_topk = torch.gather(H_by_branch, dim=2, index=gather_idx)
        pooled = torch.sum(topk_weights.unsqueeze(-1) * H_topk, dim=2)
        pooled = pooled.reshape(H.size(0), -1)

        weights = torch.zeros_like(branch_logits)
        weights.scatter_(2, topk_indices, topk_weights)
        return pooled, weights.transpose(1, 2)
