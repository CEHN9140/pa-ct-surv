"""Joint residual teacher; no downstream unimodal checkpoint initialization."""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from loss import cox_loss
from model.build import CT_Model, Pa_Model


class ResidualTeacher(nn.Module):
    def __init__(self,ct_pretrained_path=None,dropout=.25):
        super().__init__()
        self.pa_branch=Pa_Model(feature_dim=1280,abmil_dropout=dropout).mil
        self.ct_backbone=CT_Model(pretrained_path=ct_pretrained_path,dropout=dropout).ct
        self.ct_backbone.fc=nn.Identity()
        self.ct_delta=nn.Sequential(nn.Linear(512,128),nn.ReLU(),nn.Linear(128,1))
        nn.init.zeros_(self.ct_delta[-1].weight)
        nn.init.zeros_(self.ct_delta[-1].bias)

    def forward(self,ct,pa):
        if pa.ndim!=3 or pa.shape[1:]!=(25,1280) or ct.size(0)!=pa.size(0):
            raise ValueError('Expected paired CT and pathology [B,25,1280]')
        risk_pa=self.pa_branch(F.normalize(pa,p=2,dim=-1,eps=1e-12))[0]
        delta=self.ct_delta(self.ct_backbone.extract_features(ct)).squeeze(-1)
        return risk_pa+delta,delta,risk_pa


def residual_objective(risk,delta,pathology,time,event,lambda_pa=.3,lambda_r=.01):
    if not np.isfinite([lambda_pa,lambda_r]).all() or min(lambda_pa,lambda_r)<0:
        raise ValueError('Loss weights must be finite and nonnegative')
    fused=cox_loss(risk,time,event)
    auxiliary=cox_loss(pathology,time,event)
    correction=delta.square().mean()
    return dict(total=fused+lambda_pa*auxiliary+lambda_r*correction,
                fused_cox=fused,pathology_cox=auxiliary,correction_l2=correction)


def pair_changes(time,event,reference,prediction):
    """Same comparable pairs/tie tolerance as Harrell C-index in scikit-survival."""
    t,e,p,r=[np.asarray(x) for x in (time,event,reference,prediction)]
    if (any(x.ndim!=1 or not np.isfinite(x).all() for x in (t,e,p,r))
            or len({len(x) for x in (t,e,p,r)})!=1 or not np.isin(e,[0,1]).all()):
        raise ValueError('Need aligned finite risks/outcomes and binary events')
    e=e.astype(bool)
    i,j=np.where(e[:,None]&((t[:,None]<t[None,:])|((t[:,None]==t[None,:])&~e[None,:])))
    if not len(i):raise ValueError('No comparable patient pairs')
    scores=[]
    for value in (p,r):
        difference=value[i]-value[j]
        scores.append(np.where(np.abs(difference)<=1e-8,.5,(difference>0).astype(float)))
    before,after=scores
    return dict(comparable_pairs=len(i),corrected_pairs=int(((before==0)&(after==1)).sum()),
        broken_pairs=int(((before==1)&(after==0)).sum()),improved_pairs=int((after>before).sum()),
        worsened_pairs=int((after<before).sum()),unchanged_pairs=int((after==before).sum()),
        reference_cindex=float(before.mean()),teacher_cindex=float(after.mean()),
        delta_cindex=float((after-before).mean()),
        note='Corrections/breaks exclude ties; improved/worsened include half-credit tie transitions.')
