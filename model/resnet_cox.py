import functools
import os
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3x3x3(in_planes, out_planes, stride=1, dilation=1):
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        dilation=dilation,
        bias=False,
    )


def downsample_basic_block(x, planes, stride):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.zeros(
        out.size(0),
        planes - out.size(1),
        out.size(2),
        out.size(3),
        out.size(4),
        dtype=out.dtype,
        device=out.device,
    )
    return torch.cat([out, zero_pads], dim=1)


class BasicBlock3D(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, dilation=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3x3(in_planes, planes, stride=stride, dilation=dilation)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes, dilation=dilation)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResNetCox(nn.Module):
    """MedicalNet-style 3D ResNet with Cox risk head for survival analysis."""

    _DEPTH_CONFIGS = {
        10: {"layers": [1, 1, 1, 1], "shortcut": "B"},
        18: {"layers": [2, 2, 2, 2], "shortcut": "A"},
    }

    def __init__(
        self,
        in_channels=1,
        pretrained_path=None,
        dropout=0.5,
        model_depth=18,
        freeze_bn_stats=False,
    ):
        super().__init__()
        if model_depth not in self._DEPTH_CONFIGS:
            supported = ", ".join(str(depth) for depth in self._DEPTH_CONFIGS)
            raise ValueError(
                f"Unsupported model_depth={model_depth}; supported depths: {supported}"
            )

        config = self._DEPTH_CONFIGS[model_depth]
        self.model_depth = model_depth
        self.shortcut_type = config["shortcut"]
        self.freeze_bn_stats = freeze_bn_stats
        self.inplanes = 64

        self.conv1 = nn.Conv3d(
            in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        layer_blocks = config["layers"]
        self.layer1 = self._make_layer(
            BasicBlock3D, 64, blocks=layer_blocks[0], stride=1
        )
        self.layer2 = self._make_layer(
            BasicBlock3D, 128, blocks=layer_blocks[1], stride=2, dilation=1
        )
        self.layer3 = self._make_layer(
            BasicBlock3D, 256, blocks=layer_blocks[2], stride=1, dilation=2
        )
        self.layer4 = self._make_layer(
            BasicBlock3D, 512, blocks=layer_blocks[3], stride=1, dilation=4
        )

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(512 * BasicBlock3D.expansion, 1)

        # init weights (only for layers not covered by pretrained)
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        for m in self.modules():
            if isinstance(m, BasicBlock3D):
                nn.init.constant_(m.bn2.weight, 0)

        if pretrained_path is not None:
            if not os.path.isfile(pretrained_path):
                raise FileNotFoundError(
                    f"MedicalNet pretrained weights not found: {pretrained_path}"
                )
            self._load_medicalnet(pretrained_path)

    def train(self, mode=True):
        super().train(mode)
        if mode and self.freeze_bn_stats:
            for module in self.modules():
                if isinstance(module, nn.BatchNorm3d):
                    module.eval()
        return self

    def _load_medicalnet(self, path):
        state = torch.load(path, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        new_state = OrderedDict()
        for k, v in state.items():
            new_key = k.replace("module.", "") if k.startswith("module.") else k
            new_state[new_key] = v
        missing, unexpected = self.load_state_dict(new_state, strict=False)
        allowed_missing = {"fc.weight", "fc.bias"}
        incompatible_missing = [
            key for key in missing if key not in allowed_missing
        ]
        if incompatible_missing:
            raise RuntimeError(
                "MedicalNet weights are incompatible with "
                f"ResNet{self.model_depth}; missing backbone keys: "
                f"{incompatible_missing}"
            )
        print(f"[ResNetCox] Loaded pretrained from {path}")
        if missing:
            print(f"  Missing keys (ok): {missing}")
        if unexpected:
            raise RuntimeError(
                "Unexpected keys in MedicalNet checkpoint: "
                f"{unexpected}"
            )

    @property
    def feature_dim(self):
        return self.fc.in_features

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if self.shortcut_type == "A":
                downsample = functools.partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride,
                )
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False,
                    ),
                    nn.BatchNorm3d(planes * block.expansion),
                )
        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride=stride,
                dilation=dilation,
                downsample=downsample,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))
        return nn.Sequential(*layers)

    def extract_features(self, x):
        """返回 fc 之前的特征向量（不含 dropout）。"""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    def risk_forward(self, fea):
        """Forward pass from pre-extracted features to risk prediction."""
        fea = self.dropout(fea)
        return self.fc(fea).squeeze(-1)

    def forward(self, x):
        fea = self.extract_features(x)
        return self.risk_forward(fea)
