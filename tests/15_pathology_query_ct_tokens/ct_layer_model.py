"""Experiment-local MedicalNet layer3/layer4 global representations."""

import torch
from torch import nn

from model.resnet_cox import ResNetCox


class CTLayerBackbone(ResNetCox):
    def __init__(self,feature_source='layer4',pretrained_path=None,dropout=.25):
        if feature_source not in {'layer3','layer4','layer34'}:raise ValueError('Unknown CT feature source')
        super().__init__(pretrained_path=pretrained_path,dropout=dropout,model_depth=18,
                         freeze_bn_stats=True,l2_normalize=False)
        self.feature_source=feature_source
        if feature_source!='layer4':
            # Preserve A's post-construction RNG for identical shuffle/dropout streams.
            with torch.random.fork_rng(devices=[]):
                self.fc=nn.Sequential(nn.Linear(256 if feature_source=='layer3' else 768,256),
                                      nn.ReLU(inplace=True),nn.Dropout(dropout),nn.Linear(256,1))
                for module in self.fc.modules():
                    if isinstance(module,nn.Linear):
                        nn.init.normal_(module.weight,0,.01);nn.init.zeros_(module.bias)
            if feature_source=='layer3':self.layer4=nn.Identity()

    def extract_features(self,x):
        if self.feature_source=='layer4':return super().extract_features(x)
        x=self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x=self.layer1(x);x=self.layer2(x);stage3=self.layer3(x)
        pooled3=self.avgpool(stage3).flatten(1)
        if self.feature_source=='layer3':return pooled3
        pooled4=self.avgpool(self.layer4(stage3)).flatten(1)
        return torch.cat([pooled3,pooled4],dim=1)


class CTLayerModel(nn.Module):
    def __init__(self,feature_source='layer4',pretrained_path=None,dropout=.25):
        super().__init__()
        self.ct=CTLayerBackbone(feature_source,pretrained_path,dropout)

    def forward(self,x):
        return self.ct(x)
