"""Experiment-local pathology Q / CT layer4 KV teacher."""

import torch.nn as nn
import torch.nn.functional as F

from model.build import CT_Model, Pa_Model
from model.fusion import ConcatFusion
from pamt_alignment import pamt_alignment_loss


class PathQueryCTFusion(nn.Module):
    def __init__(self, dropout=.25):
        super().__init__()
        self.pa_projection = nn.Sequential(nn.Linear(512, 256), nn.ReLU(), nn.Dropout(dropout))
        self.ct_projection = nn.Sequential(nn.Linear(512, 256), nn.ReLU(), nn.Dropout(dropout))
        self.coattn = nn.MultiheadAttention(256, num_heads=1, batch_first=True)
        self.query_pool = nn.Sequential(
            nn.Linear(256, 128), nn.Tanh(), nn.Dropout(dropout), nn.Linear(128, 1),
        )
        self.head = ConcatFusion(256, 256, mmhid=256, dropout_rate=dropout, head_hidden_dim=128)

    def forward(self, pa_patches, ct_tokens, pa_patient, alignment_topk=None):
        query = self.pa_projection(pa_patches)
        ct = self.ct_projection(ct_tokens)
        retrieved_ct, attention = self.coattn(query, ct, ct)
        pool_weights = F.softmax(self.query_pool(retrieved_ct), dim=1)
        guided_ct = (pool_weights * retrieved_ct).sum(dim=1)
        output = self.head(guided_ct, pa_patient), guided_ct, attention, pool_weights
        if alignment_topk is not None:
            return (*output, pamt_alignment_loss(query, ct, topk=alignment_topk))
        return output


class PathQueryCTModel(nn.Module):
    def __init__(self, ct_pretrained_path=None, dropout=.25, alignment_topk=None,
                 pa_l2_normalize=True):
        super().__init__()
        self.pa_feature_dim = 1280
        self.alignment_topk = alignment_topk
        self.pa_l2_normalize = pa_l2_normalize
        self.ct_backbone = CT_Model(pretrained_path=ct_pretrained_path, dropout=dropout).ct
        self.pa_branch = Pa_Model(feature_dim=1280, abmil_dropout=dropout).mil
        self.fusion = PathQueryCTFusion(dropout=dropout)

    def forward(self, ct, pa):
        if pa.ndim != 3 or pa.shape[1:] != (25, 1280):
            raise ValueError('Experiment 15 requires pathology [B,25,1280]')
        spatial = self.ct_backbone.extract_spatial_features(ct)
        tokens = spatial.flatten(2).transpose(1, 2)
        # One backbone traversal supplies both spatial KV and the diagnostic CT risk.
        global_ct = self.ct_backbone.avgpool(spatial).flatten(1)
        risk_ct = self.ct_backbone.risk_forward(global_ct)
        if self.pa_l2_normalize:
            pa = F.normalize(pa, p=2, dim=-1, eps=1e-12)
        risk_pa, patient_pa, _, patches_pa = self.pa_branch(pa, return_patch_features=True)
        fused = self.fusion(patches_pa, tokens, patient_pa, alignment_topk=self.alignment_topk)
        risk, guided_ct, attention, pool_weights = fused[:4]
        output = risk, risk_ct, risk_pa, pool_weights, guided_ct, patient_pa, attention
        return (*output, fused[4]) if self.alignment_topk is not None else output
