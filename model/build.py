import torch
import torch.nn as nn
import torch.nn.functional as F

from model.abmil_cox import ABMIL, ProjABMIL
from model.fusion import BilinearFusion, ConcatFusion, CrossAttnFusion, GatedFusion
from model.gabmil_cox import GatedABMIL
from model.meanpool_cox import MeanPool
from model.resnet_cox import ResNetCox
from model.transmil_cox import TransMIL_cox


class Pa_Model(nn.Module):
    """Factory for unimodal pathology MIL survival models."""

    def __init__(self, model_name="abmil", feature_dim=1024, k=None, n_bins=None,
                 proj_dim=256, proj_type="linear"):
        super().__init__()
        self.model_name = model_name
        self.n_bins = n_bins

        if model_name.endswith("-topk"):
            base_name = model_name[:-5]
            if k is None or k <= 0:
                raise ValueError(f"{model_name} requires a positive k")
        else:
            base_name = model_name
            k = None

        if base_name == "abmil":
            self.mil = ABMIL(in_dim=feature_dim, k=k, n_bins=n_bins)
        elif base_name == "abmil-proj":
            self.mil = ProjABMIL(
                in_dim=feature_dim, proj_dim=proj_dim,
                proj_type=proj_type, k=k, n_bins=n_bins,
            )
        elif base_name == "gabmil":
            self.mil = GatedABMIL(in_dim=feature_dim, k=k, n_bins=n_bins)
        elif base_name == "meanpool":
            self.mil = MeanPool(in_dim=feature_dim, n_bins=n_bins)
        elif base_name == "transmil":
            self.mil = TransMIL_cox(n_classes=2)
        else:
            raise ValueError(f"Unknown model_name: {model_name}")

    def forward(self, pa):
        if self.model_name == "transmil":
            return self.mil(data=pa)
        return self.mil(pa)


class CT_Model(nn.Module):
    """Factory for unimodal CT survival models."""

    def __init__(
        self, model_name="resnet18", pretrained_path=None, freeze_backbone=False
    ):
        super().__init__()
        self.model_name = model_name
        self.freeze_backbone = freeze_backbone
        depth_map = {"resnet10": 10, "resnet18": 18}
        if model_name not in depth_map:
            raise ValueError(f"Unknown CT model: {model_name}")
        self.ct = ResNetCox(
            in_channels=1,
            pretrained_path=pretrained_path,
            dropout=0.5,
            model_depth=depth_map[model_name],
        )
        if freeze_backbone:
            for name, param in self.ct.named_parameters():
                if not name.startswith("fc.") and not name.startswith("dropout."):
                    param.requires_grad = False

    def train(self, mode=True):
        super().train(mode)
        if mode and self.freeze_backbone:
            self.ct.eval()
            self.ct.dropout.train()
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
        fusion_dropout=0.3,
        head_dropout=0.5,
        pa_topk=None,
        fusion_type="concat",
    ):
        super().__init__()

        # CT branch
        depth_map = {"resnet10": 10, "resnet18": 18}
        self.ct_backbone = ResNetCox(
            in_channels=1,
            pretrained_path=ct_pretrained_path,
            dropout=0.5,
            model_depth=depth_map.get(ct_model_name, 18),
        )
        self.ct_projector = (
            nn.Identity()
            if self.ct_backbone.feature_dim == feature_dim
            else nn.Sequential(
                nn.Linear(self.ct_backbone.feature_dim, feature_dim),
                nn.LayerNorm(feature_dim),
                nn.ReLU(inplace=True),
            )
        )

        # PA branch. A -topk suffix explicitly enables top-k pooling.
        if pa_model_name.endswith("-topk"):
            pa_base_name = pa_model_name[:-5]
            if pa_topk is None or pa_topk <= 0:
                raise ValueError(f"{pa_model_name} requires a positive pa_topk")
        else:
            pa_base_name = pa_model_name
            pa_topk = None

        if pa_base_name == "abmil":
            self.pa_branch = ABMIL(in_dim=1024, k=pa_topk)
            pa_output_dim = 500
        elif pa_base_name == "gabmil":
            self.pa_branch = GatedABMIL(in_dim=1024, k=pa_topk)
            pa_output_dim = 500
        else:
            raise ValueError(f"Unknown pa_model_name: {pa_model_name}")

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
            )
        elif fusion_type == "bilinear":
            self.fusion = BilinearFusion(
                dim1=128,
                dim2=128,
                mmhid=mmhid,
                dropout_rate=fusion_dropout,
                in_dim=feature_dim,
            )
            self.fused_head = nn.Linear(mmhid, 1)
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
            self.risk_weight = nn.Parameter(torch.tensor(0.0))
        else:
            raise ValueError(f"Unknown fusion_type: {fusion_type}")

    def forward(self, ct, pa):
        raw_ct_fea = self.ct_backbone.extract_features(ct)
        risk_ct = self.ct_backbone.risk_forward(raw_ct_fea)
        ct_fea = self.ct_projector(raw_ct_fea)
        # ct_fea = F.normalize(ct_fea, p=2, dim=1)  # COMMENTED: test fusion without L2 norm
        risk_pa, pa_fea, pa_att = self.pa_branch(pa)
        pa_fea = self.pa_projector(pa_fea)
        # pa_fea = F.normalize(pa_fea, p=2, dim=1)  # COMMENTED: test fusion without L2 norm
        if self.fusion_type == "weighted":
            alpha = torch.sigmoid(self.risk_weight)
            risk_fused = alpha * risk_ct + (1 - alpha) * risk_pa
        else:
            risk_fused = self.fusion(ct_fea, pa_fea)
            if self.fusion_type in ("bilinear",):
                risk_fused = self.fused_head(risk_fused).squeeze(-1)
        return risk_fused, risk_ct, risk_pa, None, ct_fea, pa_fea, pa_att
