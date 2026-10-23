import torch
import torch.nn as nn
import torch.nn.functional as F


class GABMIL(nn.Module):
    """Gated attention MIL with matched pre/post-pooling projections."""

    def __init__(
        self,
        in_dim=1024,
        hidden_dim=512,
        attention_dim=128,
        post_hidden_dim=256,
        dropout=0.0,
        n_bins=None,
    ):
        super().__init__()
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if n_bins is not None and n_bins <= 0:
            raise ValueError("n_bins must be positive when provided")
        self.n_bins = n_bins

        self.compression = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)
        )
        self.attention_V = nn.Sequential(
            nn.Linear(hidden_dim, attention_dim),
            nn.Tanh(),
        )
        self.attention_U = nn.Sequential(
            nn.Linear(hidden_dim, attention_dim),
            nn.Sigmoid(),
        )
        self.attention_dropout = nn.Dropout(dropout)
        self.attention_w = nn.Linear(attention_dim, 1)
        self.post_compression = nn.Sequential(
            nn.Linear(hidden_dim, post_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.risk_head = nn.Linear(
            post_hidden_dim, n_bins if n_bins is not None else 1
        )

    def pool_attention(self, H, logits):
        weights = F.softmax(logits, dim=1)
        pooled = torch.bmm(weights.transpose(1, 2), H).squeeze(1)
        return pooled, weights

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(0)
        H = self.compression(x)
        gated = self.attention_V(H) * self.attention_U(H)
        gated = self.attention_dropout(gated)
        logits = self.attention_w(gated)
        pooled, weights = self.pool_attention(H, logits)
        pooled = self.post_compression(pooled)
        output = self.risk_head(pooled)
        if self.n_bins is None:
            return output.squeeze(-1), pooled, weights

        hazards = torch.sigmoid(output)
        survival = torch.cumprod(1.0 - hazards, dim=1)
        risk = -survival.sum(dim=1)
        return risk, pooled, weights, hazards, survival


class GABMIL_TopK(GABMIL):
    """GABMIL with attention pooling restricted to the top-k instances."""

    def __init__(
        self,
        in_dim=1024,
        hidden_dim=512,
        attention_dim=128,
        k=None,
        dropout=0.0,
        n_bins=None,
    ):
        if k is None or k <= 0:
            raise ValueError("GABMIL_TopK requires a positive k")
        super().__init__(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            attention_dim=attention_dim,
            dropout=dropout,
            n_bins=n_bins,
        )
        self.k = k

    def pool_attention(self, H, logits):
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
