import torch
import torch.nn as nn

from model.abmil_cox import ABMIL, ABMIL_RandomSample, ABMIL_TopK
from model.fusion import (
    BilinearFusion,
    BottleneckFusion,
    CompactBilinearFusion,
    ConcatFusion,
    ContrastiveFusion,
    CrossAttnFusion,
    DCCAFusion,
    FeatureProductFusion,
    FeatureSumFusion,
    FiLMPatchFusion,
    GatedFusion,
    GraphFusion,
    LowRankFusion,
    OptimalTransportFusion,
    QueryFusion,
    SharedPrivateFusion,
    TokenCoAttentionFusion,
)
from model.gabmil_cox import GABMIL, GABMIL_TopK
from model.meanpool_cox import MeanPool
from model.resnet_cox import ResNetCox
from model.transmil_cox import TransMILCox


class Pa_Model(nn.Module):
    """Factory for unimodal pathology MIL survival models."""

    def __init__(
        self,
        model_name="abmil",
        feature_dim=1024,
        k=None,
        abmil_dropout=0.0,
        attention_branches=1,
        time_bins=None,
    ):
        super().__init__()
        self.model_name = model_name
        uses_k = model_name.endswith("-topk") or model_name == "abmil_randsample"
        if uses_k and (k is None or k <= 0):
            raise ValueError(f"{model_name} requires a positive k")
        if not uses_k and k is not None:
            raise ValueError(f"{model_name} does not use k")
        if attention_branches != 1 and model_name not in {
            "abmil", "abmil-topk", "abmil_randsample"
        }:
            raise ValueError("Multiple attention branches are only supported by ABMIL")
        n_bins = None
        if time_bins is not None:
            bins = torch.as_tensor(time_bins, dtype=torch.float32).detach().clone()
            if bins.ndim != 1 or bins.numel() == 0 or not torch.isfinite(bins).all():
                raise ValueError("time_bins must contain finite cut points")
            if not torch.all(bins[1:] > bins[:-1]):
                raise ValueError("time_bins must be strictly increasing")
            self.register_buffer("time_bins", bins)
            n_bins = len(bins) + 1
        kwargs = dict(in_dim=feature_dim, dropout=abmil_dropout, n_bins=n_bins)
        if model_name in {"abmil", "abmil-topk", "abmil_randsample"}:
            kwargs["attention_branches"] = attention_branches
            if model_name == "abmil":
                self.mil = ABMIL(**kwargs)
            elif model_name == "abmil-topk":
                self.mil = ABMIL_TopK(k=k, **kwargs)
            else:
                self.mil = ABMIL_RandomSample(k=k, **kwargs)
        elif model_name == "gabmil":
            self.mil = GABMIL(**kwargs)
        elif model_name == "gabmil-topk":
            self.mil = GABMIL_TopK(k=k, **kwargs)
        elif model_name == "meanpool":
            self.mil = MeanPool(**kwargs)
        elif model_name == "transmil":
            self.mil = TransMILCox(**kwargs)
        else:
            raise ValueError(f"Unknown model_name: {model_name}")

    def forward(self, pa):
        return self.mil(pa)


class CT_Model(nn.Module):
    """Factory for unimodal CT survival models."""

    def __init__(
        self,
        model_name="resnet18",
        pretrained_path=None,
        freeze_backbone=False,
        dropout=0.5,
        model_depth=None,
    ):
        super().__init__()
        self.model_name = model_name
        self.freeze_backbone = freeze_backbone
        depth_map = {"resnet10": 10, "resnet18": 18}
        if model_name not in depth_map:
            raise ValueError(f"Unknown CT model: {model_name}")
        model_depth = depth_map[model_name] if model_depth is None else model_depth
        if model_depth not in depth_map.values():
            raise ValueError(f"Unknown CT model depth: {model_depth}")
        self.ct = ResNetCox(
            in_channels=1,
            pretrained_path=pretrained_path,
            dropout=dropout,
            model_depth=model_depth,
            freeze_bn_stats=True,
        )
        if freeze_backbone:
            for name, param in self.ct.named_parameters():
                if not name.startswith("fc."):
                    param.requires_grad = False

    def train(self, mode=True):
        super().train(mode)
        if mode and self.freeze_backbone:
            self.ct.eval()
            self.ct.fc.train()
        return self

    def forward(self, x):
        return self.ct(x)


class Pa_CT_Model(nn.Module):
    """Factory for bimodal PA+CT fusion survival models."""

    def __init__(
        self,
        pa_model_name="abmil",
        ct_model_name="resnet18",
        feature_dim=512,
        mmhid=512,
        ct_pretrained_path=None,
        fusion_dropout=0.0,
        pa_topk=None,
        fusion_type="concat",
        norm="none",
    ):
        super().__init__()
        if not 0.0 <= fusion_dropout < 1.0:
            raise ValueError("fusion_dropout must be in [0, 1)")
        if norm not in {"none", "layernorm"}:
            raise ValueError("norm must be 'none' or 'layernorm'")

        # CT branch
        depth_map = {"resnet10": 10, "resnet18": 18}
        self.ct_backbone = ResNetCox(
            in_channels=1,
            pretrained_path=ct_pretrained_path,
            dropout=0.0,
            model_depth=depth_map.get(ct_model_name, 18),
            freeze_bn_stats=True,
        )
        self.ct_projector = (
            nn.Identity()
            if self.ct_backbone.feature_dim == feature_dim
            else nn.Sequential(
                nn.Linear(self.ct_backbone.feature_dim, feature_dim),
                nn.ReLU(inplace=True),
            )
        )
        self.ct_norm = nn.LayerNorm(feature_dim) if norm == "layernorm" else nn.Identity()

        # All pathology encoders return a 256-dimensional patient feature.
        self.pa_branch = Pa_Model(
            model_name=pa_model_name, k=pa_topk
        ).mil
        pa_output_dim = 256

        self.pa_projector = (
            nn.Identity()
            if pa_output_dim == feature_dim
            else nn.Linear(pa_output_dim, feature_dim)
        )
        self.pa_norm = nn.LayerNorm(feature_dim) if norm == "layernorm" else nn.Identity()

        # Fusion
        self.fusion_type = fusion_type
        self.fusion_requires_patches = False
        self.fusion_requires_spatial = False
        if fusion_type == "concat":
            self.fusion = ConcatFusion(
                dim1=feature_dim,
                dim2=feature_dim,
                mmhid=mmhid,
                dropout_rate=fusion_dropout,
            )
        elif fusion_type == "bilinear":
            self.fusion = BilinearFusion(
                dim1=feature_dim,
                dim2=feature_dim,
                dropout_rate=fusion_dropout,
                in_dim=feature_dim,
            )
        elif fusion_type == "gated":
            self.fusion = GatedFusion(
                dim1=feature_dim,
                dim2=feature_dim,
                mmhid=mmhid,
                dropout_rate=fusion_dropout,
            )
        elif fusion_type == "crossattn":
            self.fusion = CrossAttnFusion(
                dim=feature_dim,
                mmhid=mmhid,
                dropout_rate=fusion_dropout,
            )
        elif fusion_type == "weighted":
            # Keep a uniform fusion-module interface for the training loop.
            self.fusion = nn.Identity()
            self.risk_weight = nn.Parameter(torch.tensor(0.0))
        elif fusion_type == "feature_sum":
            self.fusion = FeatureSumFusion(feature_dim, feature_dim, dropout_rate=fusion_dropout)
        elif fusion_type == "feature_product":
            self.fusion = FeatureProductFusion(feature_dim, feature_dim, dropout_rate=fusion_dropout)
        elif fusion_type == "lowrank":
            self.fusion = LowRankFusion(feature_dim, feature_dim, dropout_rate=fusion_dropout)
        elif fusion_type == "compact_bilinear":
            self.fusion = CompactBilinearFusion(feature_dim, feature_dim, dropout_rate=fusion_dropout)
        elif fusion_type == "film_patch":
            self.fusion = FiLMPatchFusion(feature_dim, 1024, dropout_rate=fusion_dropout)
            self.fusion_requires_patches = True
        elif fusion_type == "token_coattn":
            self.fusion = TokenCoAttentionFusion(self.ct_backbone.feature_dim, 1024,
                                                 dropout_rate=fusion_dropout)
            self.fusion_requires_patches = self.fusion_requires_spatial = True
        elif fusion_type == "bottleneck":
            self.fusion = BottleneckFusion(self.ct_backbone.feature_dim, 1024,
                                           dropout_rate=fusion_dropout)
            self.fusion_requires_patches = self.fusion_requires_spatial = True
        elif fusion_type == "query_fusion":
            self.fusion = QueryFusion(self.ct_backbone.feature_dim, 1024,
                                      dropout_rate=fusion_dropout)
            self.fusion_requires_patches = self.fusion_requires_spatial = True
        elif fusion_type == "graph_fusion":
            self.fusion = GraphFusion(self.ct_backbone.feature_dim, 1024,
                                      dropout_rate=fusion_dropout)
            self.fusion_requires_patches = self.fusion_requires_spatial = True
        elif fusion_type == "ot_fusion":
            self.fusion = OptimalTransportFusion(self.ct_backbone.feature_dim, 1024,
                                                 dropout_rate=fusion_dropout)
            self.fusion_requires_patches = self.fusion_requires_spatial = True
        elif fusion_type == "shared_private":
            self.fusion = SharedPrivateFusion(feature_dim, feature_dim, dropout_rate=fusion_dropout)
        elif fusion_type == "contrastive":
            self.fusion = ContrastiveFusion(feature_dim, feature_dim, dropout_rate=fusion_dropout)
        elif fusion_type == "dcca":
            self.fusion = DCCAFusion(feature_dim, feature_dim, dropout_rate=fusion_dropout)
        else:
            raise ValueError(f"Unknown fusion_type: {fusion_type}")

    def forward(self, ct, pa):
        spatial = None
        if self.fusion_requires_spatial:
            spatial = self.ct_backbone.extract_spatial_features(ct)
            raw_ct_fea = self.ct_backbone.avgpool(spatial).flatten(1)
        else:
            raw_ct_fea = self.ct_backbone.extract_features(ct)
        risk_ct = self.ct_backbone.risk_forward(raw_ct_fea)
        ct_fea = self.ct_projector(raw_ct_fea)
        ct_fea_for_fusion = self.ct_norm(ct_fea)
        risk_pa, pa_fea, pa_att = self.pa_branch(pa)
        pa_fea = self.pa_projector(pa_fea)
        pa_fea_for_fusion = self.pa_norm(pa_fea)
        auxiliary = None
        if self.fusion_requires_patches:
            if self.fusion_requires_spatial:
                risk_fused, auxiliary = self.fusion(
                    ct_fea_for_fusion, pa_fea_for_fusion,
                    patches=pa.unsqueeze(0) if pa.dim() == 2 else pa,
                    spatial=spatial,
                )
            else:
                risk_fused, auxiliary = self.fusion(
                    ct_fea_for_fusion, pa_fea_for_fusion,
                    patches=pa.unsqueeze(0) if pa.dim() == 2 else pa,
                )
        elif self.fusion_type in {
            "feature_sum", "feature_product", "lowrank", "compact_bilinear",
            "shared_private", "contrastive", "dcca",
        }:
            risk_fused, auxiliary = self.fusion(ct_fea_for_fusion, pa_fea_for_fusion)
        elif self.fusion_type == "weighted":
            alpha = torch.sigmoid(self.risk_weight)
            risk_fused = alpha * risk_ct + (1 - alpha) * risk_pa
        else:
            risk_fused = self.fusion(ct_fea_for_fusion, pa_fea_for_fusion)
        return risk_fused, risk_ct, risk_pa, auxiliary, ct_fea, pa_fea, pa_att

    def fusion_auxiliary_loss(self, payloads):
        if hasattr(self.fusion, "auxiliary_loss"):
            return self.fusion.auxiliary_loss(payloads)
        return next(self.parameters()).new_zeros(())
