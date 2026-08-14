import torch
import torch.nn as nn


class ConcatFusion(nn.Module):
    """简单的 concat 融合 + MLP。"""

    def __init__(self, dim1=512, dim2=512, mmhid=256, dropout_rate=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim1 + dim2, mmhid),
            nn.LayerNorm(mmhid),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(mmhid, 1),
        )

    def forward(self, ct_fea, pa_fea):
        return self.net(torch.cat([ct_fea, pa_fea], dim=1)).squeeze(-1)


class BilinearFusion(nn.Module):
    def __init__(
        self,
        dim1=128,
        dim2=128,
        mmhid=128,
        dropout_rate=0.25,
        skip=True,
        use_bilinear=True,
        in_dim=512,
    ):
        super(BilinearFusion, self).__init__()
        self.skip = skip
        self.use_bilinear = use_bilinear
        self.in_dim = in_dim
        self.inner_dim1 = dim1
        self.inner_dim2 = dim2

        # Project high-dim input (512) to low-dim bilinear space (128)
        self.proj1 = nn.Sequential(
            nn.Linear(in_dim, dim1), nn.LayerNorm(dim1), nn.ReLU()
        ) if in_dim != dim1 else nn.Identity()
        self.proj2 = nn.Sequential(
            nn.Linear(in_dim, dim2), nn.LayerNorm(dim2), nn.ReLU()
        ) if in_dim != dim2 else nn.Identity()

        self.linear_h1 = nn.Sequential(nn.Linear(dim1, dim1), nn.ReLU())
        self.linear_h2 = nn.Sequential(nn.Linear(dim2, dim2), nn.ReLU())

        if use_bilinear:
            self.linear_z1 = nn.Bilinear(dim1, dim2, dim1)
            self.linear_z2 = nn.Bilinear(dim1, dim2, dim2)
        else:
            self.linear_z1 = nn.Linear(dim1 + dim2, dim1)
            self.linear_z2 = nn.Linear(dim1 + dim2, dim2)

        self.linear_o1 = nn.Sequential(
            nn.Linear(dim1, dim1), nn.ReLU(), nn.Dropout(dropout_rate)
        )
        self.linear_o2 = nn.Sequential(
            nn.Linear(dim2, dim2), nn.ReLU(), nn.Dropout(dropout_rate)
        )

        self.post_fusion_dropout = nn.Dropout(p=dropout_rate)
        input_dim = (dim1 + 1) * (dim2 + 1)
        skip_dim = dim1 + dim2 + 2 if skip else 0
        self.encoder1 = nn.Sequential(
            nn.Linear(input_dim, mmhid),
            # nn.BatchNorm1d(mmhid),
            nn.LayerNorm(mmhid),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
        )
        self.encoder2 = nn.Sequential(
            nn.Linear(mmhid + skip_dim, mmhid),
            # nn.BatchNorm1d(mmhid),
            nn.LayerNorm(mmhid),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
        )

    def forward(self, vec1, vec2):
        vec1 = self.proj1(vec1)
        vec2 = self.proj2(vec2)
        h1 = self.linear_h1(vec1)
        h2 = self.linear_h2(vec2)

        z1 = (
            self.linear_z1(vec1, vec2)
            if self.use_bilinear
            else self.linear_z1(torch.cat([vec1, vec2], dim=1))
        )
        z2 = (
            self.linear_z2(vec1, vec2)
            if self.use_bilinear
            else self.linear_z2(torch.cat([vec1, vec2], dim=1))
        )

        o1 = self.linear_o1(torch.sigmoid(z1) * h1)
        o2 = self.linear_o2(torch.sigmoid(z2) * h2)

        ones1 = torch.ones(o1.size(0), 1, device=o1.device)
        ones2 = torch.ones(o2.size(0), 1, device=o2.device)
        o1 = torch.cat([o1, ones1], dim=1)
        o2 = torch.cat([o2, ones2], dim=1)

        o12 = torch.bmm(o1.unsqueeze(2), o2.unsqueeze(1)).flatten(start_dim=1)
        out = self.post_fusion_dropout(o12)
        out = self.encoder1(out)

        if self.skip:
            out = torch.cat([out, o1, o2], dim=1)

        out = self.encoder2(out)
        return out


class GatedFusion(nn.Module):
    """Learnable gate between CT and pathology features."""

    def __init__(self, dim1=512, dim2=512, mmhid=256, dropout_rate=0.3):
        super().__init__()
        self.pa_projection = (
            nn.Identity()
            if dim1 == dim2
            else nn.Sequential(
                nn.Linear(dim2, dim1),
                nn.LayerNorm(dim1),
                nn.ReLU(inplace=True),
            )
        )
        self.gate = nn.Sequential(
            nn.Linear(dim1 + dim2, mmhid),
            nn.LayerNorm(mmhid),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(mmhid, dim1),
            nn.Sigmoid(),
        )
        self.head = nn.Sequential(
            nn.Linear(dim1, mmhid),
            nn.LayerNorm(mmhid),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(mmhid, 1),
        )

    def forward(self, ct_fea, pa_fea):
        gate = self.gate(torch.cat([ct_fea, pa_fea], dim=1))
        pa_fea = self.pa_projection(pa_fea)
        fused = gate * ct_fea + (1 - gate) * pa_fea
        return self.head(fused).squeeze(-1)


class CrossAttnFusion(nn.Module):
    """Attention over two modality tokens followed by a survival head.

    A single query token attending to a single key/value token has a fixed
    attention weight of one and is therefore not meaningful cross-attention.
    Stacking both modalities as tokens lets the attention layer learn their
    self- and cross-modal interactions.
    """

    def __init__(self, dim=512, num_heads=4, mmhid=256, dropout_rate=0.2):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(
                f"dim ({dim}) must be divisible by num_heads ({num_heads})"
            )
        self.cross_ct = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads,
            dropout=dropout_rate, batch_first=True,
        )
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Sequential(
            nn.Linear(dim * 2, mmhid),
            nn.LayerNorm(mmhid),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(mmhid, 1),
        )

    def forward(self, ct_fea, pa_fea):
        # [B, D] → two modality tokens [B, 2, D].
        tokens = torch.stack([ct_fea, pa_fea], dim=1)
        attended, _ = self.cross_ct(tokens, tokens, tokens, need_weights=False)
        tokens = self.norm(tokens + attended)
        return self.head(tokens.flatten(start_dim=1)).squeeze(-1)
