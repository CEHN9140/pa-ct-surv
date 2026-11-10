import torch
import torch.nn as nn

from model.abmil_cox import ABMIL, ABMIL_RandomSample, ABMIL_TopK
from model.fusion import BilinearFusion, ConcatFusion, CrossAttentionFusion, GatedFusion
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
        l2_normalize=False,
    ):
        super().__init__()
        self.model_name = model_name
        self.feature_dim = feature_dim
        self.l2_normalize = l2_normalize
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
        if self.l2_normalize:
            pa = nn.functional.normalize(pa, p=2, dim=-1, eps=1e-12)
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
        l2_normalize=False,
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
            l2_normalize=l2_normalize,
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
        feature_dim=256,
        mmhid=256,
        ct_pretrained_path=None,
        fusion_dropout=0.0,
        pa_topk=None,
        fusion_type="concat",
        dropout=None,
        fusion_head_dim=128,
        pa_feature_dim=1024,
        pa_l2_normalize=False,
        fusion_l2_normalize=False,
        shared_ct_feature=False,
    ):
        super().__init__()
        self.pa_feature_dim = pa_feature_dim
        self.pa_l2_normalize = pa_l2_normalize
        self.fusion_l2_normalize = fusion_l2_normalize
        self.shared_ct_feature = shared_ct_feature
        if fusion_type not in {"concat", "weighted", "gated", "bilinear", "crossattention"}:
            raise ValueError(f"Unknown fusion_type: {fusion_type}")
        if fusion_type == "crossattention" and pa_model_name != "abmil":
            raise ValueError("crossattention currently requires pa_model_name='abmil'")
        # Legacy checkpoints only specified fusion_dropout; their branches used 0.
        branch_dropout = 0.0 if dropout is None else dropout
        if not 0.0 <= branch_dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if dropout is not None:
            fusion_dropout = dropout
        if not 0.0 <= fusion_dropout < 1.0:
            raise ValueError("fusion_dropout must be in [0, 1)")

        # CT branch
        self.ct_backbone = CT_Model(
            model_name=ct_model_name,
            pretrained_path=ct_pretrained_path,
            dropout=branch_dropout,
        ).ct
        self.ct_projector = (
            nn.Identity()
            if (256 if shared_ct_feature else self.ct_backbone.feature_dim) == feature_dim
            else nn.Sequential(
                nn.Linear(256 if shared_ct_feature else self.ct_backbone.feature_dim, feature_dim),
                nn.ReLU(inplace=True),
            )
        )

        # All pathology encoders return a 256-dimensional patient feature.
        self.pa_branch = Pa_Model(
            model_name=pa_model_name, k=pa_topk, abmil_dropout=branch_dropout,
            feature_dim=pa_feature_dim,
        ).mil
        pa_output_dim = 256

        self.pa_projector = (
            nn.Identity()
            if pa_output_dim == feature_dim
            else nn.Linear(pa_output_dim, feature_dim)
        )

        # Fusion
        self.fusion_type = fusion_type
        if fusion_type == "concat":
            self.fusion = ConcatFusion(
                dim1=feature_dim,
                dim2=feature_dim,
                mmhid=mmhid,
                dropout_rate=fusion_dropout,
                head_hidden_dim=fusion_head_dim,
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
                head_hidden_dim=fusion_head_dim,
            )
        elif fusion_type == "weighted":
            # Keep a uniform fusion-module interface for the training loop.
            self.fusion = nn.Identity()
            self.risk_weight = nn.Parameter(torch.tensor(0.0))
        elif fusion_type == "crossattention":
            self.fusion = CrossAttentionFusion(
                dim1=feature_dim, dim2=self.pa_branch.compression[0].out_features,
                mmhid=mmhid, dropout_rate=fusion_dropout,
                head_hidden_dim=fusion_head_dim, l2_normalize=fusion_l2_normalize,
            )

    def forward(self, ct, pa):
        raw_ct_fea = self.ct_backbone.extract_features(ct)
        if self.shared_ct_feature:
            ct_hidden = self.ct_backbone.fc[:-1](raw_ct_fea)
            risk_ct = self.ct_backbone.fc[-1](ct_hidden).squeeze(-1)
            ct_fea = self.ct_projector(ct_hidden)
        else:
            risk_ct = self.ct_backbone.risk_forward(raw_ct_fea)
            ct_fea = self.ct_projector(raw_ct_fea)
        if self.pa_l2_normalize:
            pa = nn.functional.normalize(pa, p=2, dim=-1, eps=1e-12)
        if self.fusion_type == "crossattention":
            risk_pa, pa_fea, pa_att, pa_patches = self.pa_branch(pa, return_patch_features=True)
            risk_fused, pa_fea, pa_att = self.fusion(ct_fea, pa_patches)
            if self.fusion_l2_normalize:
                ct_fea = nn.functional.normalize(ct_fea, p=2, dim=-1, eps=1e-12)
            # Here PA feature/attention are CT-conditioned, not auxiliary ABMIL output.
            return risk_fused, risk_ct, risk_pa, None, ct_fea, pa_fea, pa_att
        risk_pa, pa_fea, pa_att = self.pa_branch(pa)
        pa_fea = self.pa_projector(pa_fea)
        if self.fusion_l2_normalize:
            ct_fea = nn.functional.normalize(ct_fea, p=2, dim=-1, eps=1e-12)
            pa_fea = nn.functional.normalize(pa_fea, p=2, dim=-1, eps=1e-12)
        if self.fusion_type == "weighted":
            alpha = torch.sigmoid(self.risk_weight)
            risk_fused = alpha * risk_ct + (1 - alpha) * risk_pa
        else:
            risk_fused = self.fusion(ct_fea, pa_fea)
        # Keep feature positions stable for existing distillation and visualization.
        return risk_fused, risk_ct, risk_pa, None, ct_fea, pa_fea, pa_att
