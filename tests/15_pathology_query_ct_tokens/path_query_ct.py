"""Experiment-local pathology Q / CT layer4 KV teacher."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.build import CT_Model, Pa_Model
from model.fusion import ConcatFusion
from pamt_alignment import pamt_alignment_loss


class QKNormAttention(nn.MultiheadAttention):
    """Single-head QKNorm; preserve the original MHA projection initialization/keys."""

    def __init__(self):
        super().__init__(256, num_heads=1, batch_first=True)
        # QKNorm Eq. (3); this ROI64 control always has 512 CT context positions.
        self.scale = nn.Parameter(torch.tensor(math.log2(512**2 - 512)))

    def forward(self, query, key, value):
        weights, biases = self.in_proj_weight.chunk(3), self.in_proj_bias.chunk(3)
        q = F.normalize(F.linear(query, weights[0], biases[0]), p=2, dim=-1, eps=1e-12)
        k = F.normalize(F.linear(key, weights[1], biases[1]), p=2, dim=-1, eps=1e-12)
        v = F.linear(value, weights[2], biases[2])
        attention = (self.scale * (q @ k.transpose(-2, -1))).softmax(dim=-1)
        return self.out_proj(attention @ v), attention


class PathQueryCTFusion(nn.Module):
    def __init__(self, dropout=.25, qk_norm=False, fusion_control='crossattention'):
        super().__init__()
        if fusion_control not in {'crossattention', 'pathology_only', 'uniform_ct'}:
            raise ValueError(f'Unknown fusion control: {fusion_control}')
        if qk_norm and fusion_control != 'crossattention':
            raise ValueError('Fusion controls require standard attention initialization')
        self.fusion_control = fusion_control
        self.pa_projection = nn.Sequential(nn.Linear(512, 256), nn.ReLU(), nn.Dropout(dropout))
        self.ct_projection = nn.Sequential(nn.Linear(512, 256), nn.ReLU(), nn.Dropout(dropout))
        self.coattn = (QKNormAttention() if qk_norm
                       else nn.MultiheadAttention(256, num_heads=1, batch_first=True))
        self.query_pool = nn.Sequential(
            nn.Linear(256, 128), nn.Tanh(), nn.Dropout(dropout), nn.Linear(128, 1),
        )
        self.head = ConcatFusion(256, 256, mmhid=256, dropout_rate=dropout, head_hidden_dim=128)
        if fusion_control == 'pathology_only':
            # Construct first so retained PA/head initialization matches the teacher.
            self.pa_projection = self.ct_projection = self.coattn = self.query_pool = None

    def forward(self, pa_patches, ct_tokens, pa_patient, alignment_topk=None):
        if self.fusion_control != 'crossattention' and alignment_topk is not None:
            raise ValueError('Fusion controls require fused Cox only, without alignment')
        if self.fusion_control == 'pathology_only':
            guided_ct = torch.zeros_like(pa_patient)
            attention = pa_patient.new_zeros(pa_patient.size(0), pa_patches.size(1), ct_tokens.size(1))
            pool_weights = pa_patient.new_zeros(pa_patient.size(0), pa_patches.size(1), 1)
            return self.head(guided_ct, pa_patient), guided_ct, attention, pool_weights
        query = self.pa_projection(pa_patches)
        ct = self.ct_projection(ct_tokens)
        if self.fusion_control == 'uniform_ct':
            dim = self.coattn.embed_dim
            value = F.linear(ct, self.coattn.in_proj_weight[2 * dim:], self.coattn.in_proj_bias[2 * dim:])
            retrieved_ct = self.coattn.out_proj(value.mean(dim=1, keepdim=True))
            retrieved_ct = retrieved_ct.expand(-1, query.size(1), -1)
            attention = ct.new_full((ct.size(0), query.size(1), ct.size(1)), 1 / ct.size(1))
        else:
            retrieved_ct, attention = self.coattn(query, ct, ct)
        pool_weights = F.softmax(self.query_pool(retrieved_ct), dim=1)
        guided_ct = (pool_weights * retrieved_ct).sum(dim=1)
        output = self.head(guided_ct, pa_patient), guided_ct, attention, pool_weights
        if alignment_topk is not None:
            return (*output, pamt_alignment_loss(query, ct, topk=alignment_topk))
        return output


class PathQueryCTModel(nn.Module):
    def __init__(self, ct_pretrained_path=None, dropout=.25, alignment_topk=None,
                 pa_l2_normalize=True, all_patch_queries=False, qk_norm=False,
                 ct_zscore=False, ct_bn_adapt=False, fusion_control='crossattention'):
        super().__init__()
        if (fusion_control != 'crossattention'
                and (alignment_topk is not None or not pa_l2_normalize or all_patch_queries
                     or qk_norm or ct_zscore or ct_bn_adapt)):
            raise ValueError('Fusion controls require Top25, patch L2, fused Cox and original CT preprocessing/BN')
        if ct_zscore and ct_bn_adapt:
            raise ValueError('These controls change one CT factor at a time')
        self.pa_feature_dim = 1280
        self.alignment_topk = alignment_topk
        self.pa_l2_normalize = pa_l2_normalize
        self.all_patch_queries = all_patch_queries
        self.ct_zscore = ct_zscore
        self.fusion_control = fusion_control
        self.ct_backbone = CT_Model(pretrained_path=ct_pretrained_path, dropout=dropout).ct
        self.ct_backbone.freeze_bn_stats = not ct_bn_adapt
        self.pa_branch = Pa_Model(feature_dim=1280, abmil_dropout=dropout).mil
        self.fusion = PathQueryCTFusion(dropout=dropout, qk_norm=qk_norm, fusion_control=fusion_control)
        if fusion_control == 'pathology_only':
            self.ct_backbone = None

    def forward(self, ct, pa):
        if (pa.ndim != 3 or pa.size(-1) != 1280 or pa.size(1) < 25
                or (not self.all_patch_queries and pa.size(1) != 25)):
            raise ValueError('Expected [B,N,1280]; N=25 or full queries with exact Top25 first')
        if self.ct_zscore:
            axes = (1, 2, 3, 4)
            ct = (ct - ct.mean(axes, keepdim=True)) / ct.std(
                axes, unbiased=False, keepdim=True).clamp_min(1e-6)
        if self.fusion_control == 'pathology_only':
            tokens = pa.new_empty(pa.size(0), 0, 512)
            risk_ct = pa.new_zeros(pa.size(0))
        else:
            spatial = self.ct_backbone.extract_spatial_features(ct)
            tokens = spatial.flatten(2).transpose(1, 2)
            # One backbone traversal supplies both spatial KV and the diagnostic CT risk.
            global_ct = self.ct_backbone.avgpool(spatial).flatten(1)
            risk_ct = self.ct_backbone.risk_forward(global_ct)
        if self.pa_l2_normalize:
            pa = F.normalize(pa, p=2, dim=-1, eps=1e-12)
        if self.all_patch_queries:
            # One shared compression pass; only the original Top25 enter ABMIL pooling.
            patches_pa = self.pa_branch.compression(pa)
            top25 = patches_pa[:, :25]
            patient_pa, _ = self.pa_branch.pool_attention(top25, self.pa_branch.attention(top25))
            patient_pa = self.pa_branch.post_compression(patient_pa)
            risk_pa = self.pa_branch.risk_head(patient_pa).squeeze(-1)
        else:
            risk_pa, patient_pa, _, patches_pa = self.pa_branch(pa, return_patch_features=True)
        fused = self.fusion(patches_pa, tokens, patient_pa, alignment_topk=self.alignment_topk)
        risk, guided_ct, attention, pool_weights = fused[:4]
        output = risk, risk_ct, risk_pa, pool_weights, guided_ct, patient_pa, attention
        return (*output, fused[4]) if self.alignment_topk is not None else output
