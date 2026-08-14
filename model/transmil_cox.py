import numpy as np
import torch
import torch.nn as nn

try:
    from nystrom_attention import NystromAttention
except ModuleNotFoundError:
    NystromAttention = None


class TransLayer(nn.Module):
    def __init__(self, norm_layer=nn.LayerNorm, dim=512):
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
            dropout=0.1,
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


class TransMIL_cox(nn.Module):
    def __init__(self, n_classes, max_patches=0, eval_max_patches=0):
        super(TransMIL_cox, self).__init__()
        self.pos_layer = PPEG(dim=512)
        self._fc1 = nn.Sequential(nn.Linear(1024, 512), nn.ReLU())
        self.cls_token = nn.Parameter(torch.randn(1, 1, 512))
        self.n_classes = n_classes
        self.max_patches = int(max_patches or 0)
        self.eval_max_patches = int(eval_max_patches or 0)
        self.layer1 = TransLayer(dim=512)
        self.layer2 = TransLayer(dim=512)
        self.norm = nn.LayerNorm(512)
        self._fc2 = nn.Linear(512, 1)

    def _select_patches(self, h):
        limit = self.max_patches if self.training else self.eval_max_patches
        if limit <= 0 or h.size(1) <= limit:
            return h

        if self.training:
            idx = torch.randperm(h.size(1), device=h.device)[:limit]
        else:
            idx = torch.linspace(
                0, h.size(1) - 1, steps=limit, device=h.device
            ).long()
        return h.index_select(dim=1, index=idx)

    def forward(self, **kwargs):
        h = kwargs["data"].float()  # [B, n, 1024]
        h = self._select_patches(h)

        h = self._fc1(h)  # [B, n, 512]

        # ---->pad
        H = h.shape[1]
        _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
        add_length = _H * _W - H
        h = torch.cat([h, h[:, :add_length, :]], dim=1)  # [B, N, 512]

        # ---->cls_token
        B = h.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1).to(h.device)
        h = torch.cat((cls_tokens, h), dim=1)

        # ---->Translayer x1
        h = self.layer1(h)  # [B, N, 512]

        # ---->PPEG
        h = self.pos_layer(h, _H, _W)  # [B, N, 512]

        # ---->Translayer x2
        h = self.layer2(h)  # [B, N, 512]

        # ---->cls_token
        h = self.norm(h)[:, 0]

        # ---->predict
        risks = self._fc2(h).squeeze(-1)  # [B]
        return risks
