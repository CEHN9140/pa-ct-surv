"""Experiment-local CT auxiliary Shape head; original CT risk path preserved."""

import numpy as np
import pandas as pd
import torch
from torch import nn

from loss import cox_loss
from model.build import CT_Model


class CTWithShape(CT_Model):
    def __init__(self,n_shape=1,pretrained_path=None,dropout=.25):
        super().__init__(pretrained_path=pretrained_path,dropout=dropout)
        if n_shape not in [1,3]:raise ValueError('Use sphere1 or allShape3 targets')
        with torch.random.fork_rng(devices=[]):
            full=nn.Linear(512,3)
            nn.init.normal_(full.weight,0,.01);nn.init.zeros_(full.bias)
            self.shape_head=full if n_shape==3 else nn.Linear(512,1)
            if n_shape==1:
                with torch.no_grad():
                    self.shape_head.weight.copy_(full.weight[2:3]);self.shape_head.bias.copy_(full.bias[2:3])
        self.register_buffer('shape_mean',torch.zeros(n_shape))
        self.register_buffer('shape_std',torch.ones(n_shape))

    def set_shape_scaler(self,targets):
        values=np.asarray(targets,dtype=np.float64)
        if values.ndim!=2 or values.shape[1]!=len(self.shape_mean) or not np.isfinite(values).all():
            raise ValueError('Invalid fitting-only Shape targets')
        mean,std=values.mean(0),values.std(0)
        if not np.isfinite(std).all() or (std<=1e-8).any():raise ValueError('Constant/nonfinite Shape fitting target')
        self.shape_mean.copy_(torch.as_tensor(mean,dtype=self.shape_mean.dtype,device=self.shape_mean.device))
        self.shape_std.copy_(torch.as_tensor(std,dtype=self.shape_std.dtype,device=self.shape_std.device))

    def forward(self,ct):
        features=self.ct.extract_features(ct)
        return self.ct.risk_forward(features),self.shape_head(features)


def auxiliary_objective(model,risk,shape,truth,time,event,weight):
    if not np.isfinite(weight) or weight<0:raise ValueError('Auxiliary weight must be finite and nonnegative')
    survival=cox_loss(risk,time,event)
    mse=nn.functional.mse_loss(shape,(truth-model.shape_mean)/model.shape_std)
    return dict(total=survival if weight==0 else survival+weight*mse,cox=survival,shape_mse=mse)


def shape_optimizer(model,lr,weight_decay,backbone_lr):
    head_ids={id(p) for module in [model.ct.fc,model.shape_head] for p in module.parameters()}
    backbone,head=[],[]
    for parameter in model.parameters():
        if parameter.requires_grad:(head if id(parameter) in head_ids else backbone).append(parameter)
    return torch.optim.Adam([dict(params=backbone,lr=backbone_lr),dict(params=head,lr=lr)],weight_decay=weight_decay)


def select_auxiliary(records):
    frame=pd.DataFrame(records);choices=[]
    for weight,rows in frame.groupby('weight'):
        if set(rows.inner_fold)!={0,1,2} or rows.inner_fold.duplicated().any():
            raise ValueError('Each auxiliary candidate needs the same three inner held folds')
        if not np.isfinite(rows[['held_cindex','selected_epoch','weight']]).all().all() or (rows.selected_epoch<1).any():
            raise ValueError('Invalid inner auxiliary selection record')
        choices.append(dict(weight=float(weight),mean_held_cindex=float(rows.held_cindex.mean()),
                            epochs=int(np.median(rows.selected_epoch))))
    if not choices:raise ValueError('No auxiliary candidates')
    best=max(choices,key=lambda r:(r['mean_held_cindex'],-r['weight']))
    return dict(**best,candidates=choices,selection='mean3 inner HELD C-index; exact tie prefers smaller lambda; '
                'epochs=floor median chosen-candidate STOP epochs; no outer selection')
