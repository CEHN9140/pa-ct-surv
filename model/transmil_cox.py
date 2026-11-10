import math

import torch
import torch.nn as nn

try:
    from nystrom_attention import NystromAttention
except ModuleNotFoundError:
    NystromAttention = None


class TransLayer(nn.Module):
    def __init__(self, norm_layer=nn.LayerNorm, dim=512, dropout=0.1):
        super().__init__()
        if NystromAttention is None:
            raise ImportError(
                "TransMIL requires nystrom-attention. Install it in the training "
                "environment with: pip install nystrom-attention"
            )
        self.norm = norm_layer(dim)
        self.attn = NystromAttention(
            dim=dim,
            dim_head=dim // 8,
            heads=8,
            num_landmarks=dim // 2,  # number of landmarks
            pinv_iterations=6,  # number of moore-penrose iterations for approximating pinverse. 6 was recommended by the paper
            residual=True,  # whether to do an extra residual with the value or not. supposedly faster convergence if turned on
            dropout=dropout,
        )

    def forward(self, x):
        x = x + self.attn(self.norm(x))

        return x


class PPEG(nn.Module):
    def __init__(self, dim=512):
        super(PPEG, self).__init__()
        self.proj = nn.Conv2d(dim, dim, 7, 1, 7 // 2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5 // 2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3 // 2, groups=dim)

    def forward(self, x, H, W):
        B, _, C = x.shape
        cls_token, feat_token = x[:, 0], x[:, 1:]
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        x = self.proj(cnn_feat) + cnn_feat + self.proj1(cnn_feat) + self.proj2(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        x = torch.cat((cls_token.unsqueeze(1), x), dim=1)
        return x


class TransMILCox(nn.Module):
    """Official TransMIL encoder with a patient-level survival prediction head."""

    def __init__(self, in_dim=1024, dropout=0.0, n_bins=None):
        super().__init__()
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if n_bins is not None and n_bins <= 0:
            raise ValueError("n_bins must be positive when provided")
        self.n_bins = n_bins
        self.compression = nn.Sequential(nn.Linear(in_dim, 512), nn.ReLU())
        self.pos_layer = PPEG(dim=512)
        self.cls_token = nn.Parameter(torch.randn(1, 1, 512))
        self.layer1 = TransLayer(dim=512, dropout=dropout)
        self.layer2 = TransLayer(dim=512, dropout=dropout)
        self.norm = nn.LayerNorm(512)
        self.post_compression = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(dropout),
        )
        self.risk_head = nn.Linear(256, n_bins if n_bins is not None else 1)

    def forward(self, data):
        if data.dim() == 2:
            data = data.unsqueeze(0)
        h = self.compression(data.float())

        # ---->pad
        H = h.shape[1]
        _H = _W = math.ceil(math.sqrt(H))
        add_length = _H * _W - H
        h = torch.cat([h, h[:, :add_length, :]], dim=1)

        # ---->cls_token
        B = h.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1).to(h.device)
        h = torch.cat((cls_tokens, h), dim=1)

        # ---->Translayer x1
        h = self.layer1(h)

        # ---->PPEG
        h = self.pos_layer(h, _H, _W)

        # ---->Translayer x2
        h = self.layer2(h)

        # ---->cls_token
        h = self.norm(h)[:, 0]

        # ---->predict
        pooled = self.post_compression(h)
        output = self.risk_head(pooled)
        if self.n_bins is None:
            return output.squeeze(-1), pooled, None
        hazards = torch.sigmoid(output)
        survival = torch.cumprod(1.0 - hazards, dim=1)
        return -survival.sum(dim=1), pooled, None, hazards, survival
