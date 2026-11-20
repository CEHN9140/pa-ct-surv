"""Global raw-UNI mean plus a low-capacity DFS-supervised attention branch."""
import torch
from torch import nn


class MeanGlobalLocal(nn.Module):
    def __init__(self,in_dim=1024,dropout=.25):
        super().__init__()
        self.compression=nn.Sequential(nn.Linear(in_dim,128),nn.ReLU(),nn.Dropout(dropout))
        self.attention=nn.Sequential(nn.Linear(128,32),nn.Tanh(),nn.Dropout(dropout),nn.Linear(32,1))
        self.post_compression=nn.Sequential(nn.Linear(128,64),nn.ReLU(),nn.Dropout(dropout))
        self.risk_head=nn.Sequential(nn.Linear(in_dim+64,64),nn.ReLU(),nn.Dropout(dropout),nn.Linear(64,1))

    def forward(self,x):
        if x.ndim==2:x=x.unsqueeze(0)
        global_features=x.mean(dim=1)
        h=self.compression(x)
        weights=torch.softmax(self.attention(h),dim=1)
        local_features=self.post_compression((weights*h).sum(dim=1))
        fused=torch.cat([global_features,local_features],dim=1)
        return self.risk_head(fused).squeeze(-1),fused,weights
