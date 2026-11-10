"""Experiment09 control: fixed patch mean and a single scalar-risk linear layer."""
from torch import nn


class MeanPoolLinear(nn.Module):
    def __init__(self,in_dim=1280):
        super().__init__()
        self.risk_head=nn.Linear(in_dim,1)

    def forward(self,features):
        if features.dim()==2:features=features.unsqueeze(0)
        pooled=features.mean(dim=1)
        return self.risk_head(pooled).squeeze(-1),pooled,None
