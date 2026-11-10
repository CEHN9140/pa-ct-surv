import torch
import torch.nn as nn
import torch.nn.functional as F


class ConcatFusion(nn.Module):
    """简单的 concat 融合 + MLP。"""

    def __init__(self, dim1=256, dim2=256, mmhid=256, dropout_rate=0.3,
                 head_hidden_dim=128):
        super().__init__()
        layers = [
            nn.Linear(dim1 + dim2, mmhid),
            nn.LayerNorm(mmhid),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
        ]
        if head_hidden_dim is not None:
            layers.extend([
                nn.Linear(mmhid, head_hidden_dim), nn.LayerNorm(head_hidden_dim),
                nn.ReLU(inplace=True), nn.Dropout(dropout_rate),
            ])
        layers.append(nn.Linear(mmhid if head_hidden_dim is None else head_hidden_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, ct_fea, pa_fea):
        return self.net(torch.cat([ct_fea, pa_fea], dim=1)).squeeze(-1)


class CrossAttentionFusion(nn.Module):
    """One CT query attends to pathology patches, then concatenates with CT.

    Inputs are patient CT features [B, dim1] and pre-pooling pathology features
    [B, N, dim2]. Return risk, guided pathology feature, and [B, N, 1] weights.
    Single-head attention and zero attention dropout match MCAT_pact's coattn.
    """

    def __init__(self, dim1=256, dim2=512, mmhid=256, dropout_rate=0.25,
                 head_hidden_dim=128, l2_normalize=False):
        super().__init__()
        self.l2_normalize = l2_normalize
        self.pa_projection = nn.Sequential(
            nn.Linear(dim2, dim1), nn.ReLU(), nn.Dropout(dropout_rate),
        )
        self.coattn = nn.MultiheadAttention(dim1, num_heads=1, batch_first=True)
        self.head = ConcatFusion(dim1, dim1, mmhid, dropout_rate, head_hidden_dim)

    def forward(self, ct_fea, pa_patches):
        if (ct_fea.ndim != 2 or pa_patches.ndim != 3
                or ct_fea.size(0) != pa_patches.size(0) or pa_patches.size(1) == 0):
            raise ValueError("Expected CT [B, D] and nonempty pathology patches [B, N, P]")
        if self.l2_normalize:
            ct_fea = F.normalize(ct_fea, p=2, dim=-1, eps=1e-12)
        patches = self.pa_projection(pa_patches)
        guided, weights = self.coattn(ct_fea.unsqueeze(1), patches, patches)
        guided = guided.squeeze(1)
        if self.l2_normalize:
            guided = F.normalize(guided, p=2, dim=-1, eps=1e-12)
        return self.head(ct_fea, guided), guided, weights.transpose(1, 2)


class BilinearFusion(nn.Module):
    """Full PA x CT outer product with signed sqrt and L2 normalization."""

    def __init__(self, dim1=256, dim2=256, in_dim=256, dropout_rate=0.0):
        super().__init__()
        self.proj1 = nn.Linear(in_dim, dim1) if dim1 != in_dim else nn.Identity()
        self.proj2 = nn.Linear(in_dim, dim2) if dim2 != in_dim else nn.Identity()
        self.dropout = nn.Dropout(dropout_rate)
        self.head = nn.Linear(dim1 * dim2, 1)

    def forward(self, ct_fea, pa_fea):
        ct_fea = self.proj1(ct_fea)
        pa_fea = self.proj2(pa_fea)
        outer = torch.bmm(pa_fea.unsqueeze(2), ct_fea.unsqueeze(1))
        fused = outer.flatten(start_dim=1)
        fused = fused.sign() * torch.sqrt(fused.abs() + 1e-8)
        fused = F.normalize(fused, p=2, dim=1, eps=1e-8)
        return self.head(self.dropout(fused)).squeeze(-1)


class GatedFusion(nn.Module):
    """Learnable gate between CT and pathology features."""

    def __init__(self, dim1=256, dim2=256, mmhid=256, dropout_rate=0.3,
                 head_hidden_dim=128):
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
        head_dim = mmhid if head_hidden_dim is None else head_hidden_dim
        self.head = nn.Sequential(
            nn.Linear(dim1, head_dim),
            nn.LayerNorm(head_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(head_dim, 1),
        )

    def forward(self, ct_fea, pa_fea):
        gate = self.gate(torch.cat([ct_fea, pa_fea], dim=1))
        pa_fea = self.pa_projection(pa_fea)
        fused = gate * ct_fea + (1 - gate) * pa_fea
        return self.head(fused).squeeze(-1)
