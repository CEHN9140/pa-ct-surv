"""Trainable pathology Q / seven-family Radiomics K,V fusion teacher."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from torch import nn
import torch.nn.functional as F

from model.abmil_cox import ABMIL
from model.fusion import ConcatFusion


class RadiomicsTeacher(nn.Module):
    def __init__(self, group_indices, dropout=.25, fusion_type='crossattention'):
        super().__init__()
        if fusion_type == 'pa_only':
            self.fusion_type = fusion_type
            self.pa_branch = ABMIL(in_dim=1280, dropout=dropout)
            self.pa_branch.risk_head = nn.Identity()
            # Match the fused head's hidden layers; only its input lacks CT256.
            self.head = nn.Sequential(nn.Linear(256, 256), nn.LayerNorm(256),
                nn.ReLU(), nn.Dropout(dropout), nn.Linear(256, 128), nn.LayerNorm(128),
                nn.ReLU(), nn.Dropout(dropout), nn.Linear(128, 1))
            return
        flattened = [int(i) for group in group_indices for i in group]
        if (len(group_indices) != 7 or sorted(flattened) != list(range(len(flattened)))
                or sum(bool(group) for group in group_indices) < 2):
            raise ValueError('Need seven disjoint groups covering CT features, at least two nonempty')
        if fusion_type not in ['crossattention', 'concat']:
            raise ValueError('Use crossattention, concat or pa_only')
        self.group_indices = [list(group) for group in group_indices]
        self.input_dim = len(flattened)
        self.fusion_type = fusion_type
        self.pa_branch = ABMIL(in_dim=1280, dropout=dropout)
        self.pa_branch.risk_head = nn.Identity()  # fused Cox only; no unused risk-head parameters
        self.ct_encoders = nn.ModuleList([
            nn.Sequential(nn.Linear(len(group), 256), nn.ReLU(), nn.Dropout(dropout))
            if group else nn.Identity() for group in self.group_indices])
        self.register_buffer('ct_padding_mask', torch.tensor([not bool(group) for group in group_indices]))
        self.pa_projection = nn.Sequential(nn.Linear(512, 256), nn.ReLU(), nn.Dropout(dropout))
        # Match test15's standard one-head MHA; its attention-weight dropout is0.
        self.coattn = nn.MultiheadAttention(256, num_heads=1, batch_first=True)
        self.query_pool = nn.Sequential(nn.Linear(256, 128), nn.Tanh(), nn.Dropout(dropout), nn.Linear(128, 1))
        self.head = ConcatFusion(256, 256, mmhid=256, dropout_rate=dropout, head_hidden_dim=128)
        if fusion_type == 'concat':
            # Construct first so common PA/CT/head initializations match at the same seed.
            self.pa_projection = self.coattn = self.query_pool = None

    def forward(self, ct, pa, return_attention=False):
        if pa.ndim != 3 or pa.shape[1:] != (25, 1280):
            raise ValueError('Expected PA [B,25,1280]')
        if self.fusion_type != 'pa_only' and (
                ct.ndim != 2 or ct.shape != (pa.size(0), self.input_dim)):
            raise ValueError('Expected paired CT [B,D] and PA [B,25,1280]')
        if not torch.isfinite(pa).all() or (self.fusion_type != 'pa_only' and not torch.isfinite(ct).all()):
            raise ValueError('Nonfinite teacher inputs')
        patches = self.pa_branch.compression(F.normalize(pa, p=2, dim=-1, eps=1e-12))
        patient, pa_weights = self.pa_branch.pool_attention(patches, self.pa_branch.attention(patches))
        patient = self.pa_branch.post_compression(patient)
        if self.fusion_type == 'pa_only':
            risk = self.head(patient).squeeze(-1)
            if not return_attention:
                return risk
            return risk, dict(attention=None, ct_tokens=None, guided_ct=None,
                              pathology_feature=patient, pathology_weights=pa_weights,
                              query_pool_weights=None)
        tokens = torch.stack([encoder(ct[:, group]) if group else ct.new_zeros(ct.size(0), 256)
                              for group, encoder in zip(self.group_indices, self.ct_encoders)], dim=1)
        if self.fusion_type == 'crossattention':
            query = self.pa_projection(patches)
            retrieved, attention = self.coattn(query, tokens, tokens,
                key_padding_mask=self.ct_padding_mask.unsqueeze(0).expand(ct.size(0), -1))
            pool_weights = F.softmax(self.query_pool(retrieved), dim=1)
            guided = (retrieved * pool_weights).sum(dim=1)
        else:
            guided = tokens[:, ~self.ct_padding_mask].mean(dim=1)
            attention = pool_weights = None
        risk = self.head(guided, patient)
        if not return_attention:
            return risk
        return risk, dict(attention=attention, ct_tokens=tokens, guided_ct=guided,
                          pathology_feature=patient, pathology_weights=pa_weights,
                          query_pool_weights=pool_weights, ct_padding_mask=self.ct_padding_mask)
