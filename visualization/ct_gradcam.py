"""
3D CT Grad-CAM for survival model interpretation.

Supports single-case and val-set batch sampling.

Usage:
    python visualization/ct_gradcam.py \
        --data_dir /home/gly001/cqj/pa_ct_surv/data \
        --checkpoint_root experiments/final_core_v14/seed42/ct_student_xxx/checkpoints \
        --model_type ct_student \
        --fold 0 --ct_id 1000044195 --roi_size 64 \
        --pretrained_path /home/gly001/cqj/pa_ct_surv/model/ct_pretrain/resnet_18_23dataset.pth \
        --output_dir results/visualization/ct

Batch sampling:
    python visualization/ct_gradcam.py \
        --checkpoint_root ... --model_type ct_student --fold 0 \
        --sample_n 10 --sample_split val --cv_seed 42 --sample_seed 42
"""

import argparse
import random
import sys
from pathlib import Path

# Ensure project root is on sys.path
_project_root = Path(__file__).resolve().parents[1]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import numpy as np
import torch
import torch.nn.functional as F

from visualization.vis_utils import (
    ensure_dir,
    find_case_row,
    find_fold_checkpoint,
    get_split_ids,
    load_ct_mask,
    load_ct_volume,
    load_path_feature,
)


def parse_args():
    parser = argparse.ArgumentParser(description="CT Grad-CAM visualization")
    parser.add_argument("--data_dir", default="/home/gly001/cqj/pa_ct_surv/data")
    parser.add_argument("--checkpoint_root", required=True)
    parser.add_argument("--results_root", default=None)
    parser.add_argument(
        "--model_type",
        required=True,
        choices=["ct_baseline", "ct_student", "pact_teacher"],
    )
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument(
        "--ct_id", default=None, help="Single CT image ID (optional if --sample_n > 0)"
    )
    parser.add_argument(
        "--ct_ids",
        nargs="+",
        default=None,
        help="Explicit list of CT image IDs for aligned multi-model visualization.",
    )
    parser.add_argument("--roi_size", type=int, default=64)
    parser.add_argument(
        "--sample_n", type=int, default=0, help="Number of CT cases to sample"
    )
    parser.add_argument("--sample_split", default="val", choices=["train", "val"])
    parser.add_argument("--sample_seed", type=int, default=42)
    parser.add_argument("--cv_seed", type=int, default=42, help="CV split seed")
    parser.add_argument(
        "--pretrained_path",
        default="/home/gly001/cqj/pa_ct_surv/model/ct_pretrain/resnet_18_23dataset.pth",
    )
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--pa_model", default="abmil-cox-topk")
    parser.add_argument(
        "--implementation",
        default="legacy",
        choices=["original", "legacy"],
        help="Pathology implementation used inside a PACT checkpoint.",
    )
    parser.add_argument("--fusion_type", default="concat")
    parser.add_argument("--k", type=int, default=512)
    parser.add_argument(
        "--target_layer",
        default="layer3.1.conv2",
        help=(
            "Backbone layer for Grad-CAM. Use layer3.1.conv2 by default for "
            "ROI64 because layer4 only produces a 2x2x2 CAM."
        ),
    )
    parser.add_argument(
        "--slice_mode",
        default="maxcam",
        choices=["maxcam", "center", "fixed"],
        help=(
            "Slice selection mode. maxcam picks each model's highest-CAM slice; "
            "center uses the volume center; fixed uses --axial_slice/--coronal_slice/--sagittal_slice."
        ),
    )
    parser.add_argument("--axial_slice", type=int, default=None)
    parser.add_argument("--coronal_slice", type=int, default=None)
    parser.add_argument("--sagittal_slice", type=int, default=None)
    return parser.parse_args()


# ============================================================
# Model loading
# ============================================================


def load_ct_model_for_gradcam(ckpt_path, pretrained_path, dropout, device):
    from model.build import CT_Model

    model = CT_Model(
        model_name="resnet", pretrained_path=pretrained_path, dropout=dropout
    )
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, model.ct


def load_pact_model_for_gradcam(
    ckpt_path,
    pa_model_name,
    fusion_type,
    k,
    dropout,
    device,
    implementation="legacy",
):
    from model.build import Pa_CT_Model

    model = Pa_CT_Model(
        pa_model_name=pa_model_name,
        ct_pretrained_path=None,
        pa_topk=k,
        fusion_type=fusion_type,
        ct_dropout=dropout,
        pa_implementation=implementation,
    )
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, model.ct_backbone


# ============================================================
# Grad-CAM helpers
# ============================================================


def find_target_layer(backbone, target_layer_name=None):
    if target_layer_name:
        modules = dict(backbone.named_modules())
        if target_layer_name not in modules:
            available = [name for name, module in backbone.named_modules() if isinstance(module, torch.nn.Conv3d)]
            raise ValueError(
                f"Target layer '{target_layer_name}' not found. "
                f"Available Conv3d layers: {available}"
            )
        target_module = modules[target_layer_name]
        if not isinstance(target_module, torch.nn.Conv3d):
            raise ValueError(f"Target layer '{target_layer_name}' is not Conv3d")
        print(f"[Grad-CAM] Target layer: {target_layer_name}")
        return target_module, target_layer_name

    target_name, target_module = None, None
    for name, module in backbone.named_modules():
        if isinstance(module, torch.nn.Conv3d):
            target_name, target_module = name, module
    if target_module is None:
        raise RuntimeError("No Conv3d found")
    print(f"[Grad-CAM] Target layer: {target_name}")
    return target_module, target_name


def compute_signed_cams(activations, gradients, target_size):
    acts = activations[0].detach()
    grads = gradients[0].detach()
    weights = grads.mean(dim=(2, 3, 4), keepdim=True)
    signed_cam = (weights * acts).sum(dim=1, keepdim=True)
    signed_cam = F.interpolate(
        signed_cam,
        size=target_size,
        mode="trilinear",
        align_corners=False,
    )
    # Preserve both directions with one shared scale so their magnitudes remain
    # comparable. Standard Grad-CAM is the positive_cam output only.
    scale = signed_cam.abs().amax()
    if scale > 1e-8:
        signed_cam = signed_cam / scale
    positive_cam = torch.clamp(signed_cam, min=0)
    negative_cam = torch.clamp(-signed_cam, min=0)
    return (
        positive_cam.squeeze().cpu().numpy(),
        negative_cam.squeeze().cpu().numpy(),
    )


def collect_slices(positive_cam, negative_cam, ct_vol, args, mask_volume=None):
    positive_np = np.asarray(positive_cam)
    negative_np = np.asarray(negative_cam)
    magnitude_np = positive_np + negative_np
    ct_np = ct_vol.detach().squeeze().cpu().numpy()
    mask_np = (
        np.zeros_like(magnitude_np, dtype=np.uint8)
        if mask_volume is None
        else np.asarray(mask_volume, dtype=np.uint8)
    )
    if mask_np.shape != magnitude_np.shape:
        raise ValueError(
            f"CT mask shape {mask_np.shape} does not match CAM shape "
            f"{magnitude_np.shape}"
        )
    if args.slice_mode == "maxcam":
        if magnitude_np.max() <= 1e-8:
            z, y, x = (size // 2 for size in magnitude_np.shape)
        else:
            z = int(np.argmax(magnitude_np.sum(axis=(1, 2))))
            y = int(np.argmax(magnitude_np.sum(axis=(0, 2))))
            x = int(np.argmax(magnitude_np.sum(axis=(0, 1))))
    elif args.slice_mode == "center":
        z = magnitude_np.shape[0] // 2
        y = magnitude_np.shape[1] // 2
        x = magnitude_np.shape[2] // 2
    else:
        missing = [
            name
            for name, value in (
                ("--axial_slice", args.axial_slice),
                ("--coronal_slice", args.coronal_slice),
                ("--sagittal_slice", args.sagittal_slice),
            )
            if value is None
        ]
        if missing:
            raise ValueError(f"--slice_mode fixed requires {', '.join(missing)}")
        z, y, x = args.axial_slice, args.coronal_slice, args.sagittal_slice
        z = int(np.clip(z, 0, magnitude_np.shape[0] - 1))
        y = int(np.clip(y, 0, magnitude_np.shape[1] - 1))
        x = int(np.clip(x, 0, magnitude_np.shape[2] - 1))
    return {
        "axial": {
            "positive_cam": positive_np[z, :, :],
            "negative_cam": negative_np[z, :, :],
            "ct_slice": ct_np[z, :, :],
            "tumor_mask": mask_np[z, :, :],
        },
        "coronal": {
            "positive_cam": positive_np[:, y, :],
            "negative_cam": negative_np[:, y, :],
            "ct_slice": ct_np[:, y, :],
            "tumor_mask": mask_np[:, y, :],
        },
        "sagittal": {
            "positive_cam": positive_np[:, :, x],
            "negative_cam": negative_np[:, :, x],
            "ct_slice": ct_np[:, :, x],
            "tumor_mask": mask_np[:, :, x],
        },
    }, {"axial": z, "coronal": y, "sagittal": x}


# ============================================================
# Core single-case pipeline
# ============================================================


def gradcam_single_case(ct_id, model, backbone, args, output_dir, device):
    """Run Grad-CAM for one CT case and save PNG views."""
    row = find_case_row(args.data_dir, ct_id=ct_id, roi_size=args.roi_size)
    ct_path = str(row["ct_path"])
    pt_path = str(row.get("pa_path", ""))
    label = int(row["event"])
    time = float(row["time"])

    target_layer, target_name = find_target_layer(backbone, args.target_layer)
    activations, gradients = [], []

    def fw_hook(m, i, o):
        activations.append(o)

    def bw_hook(m, gi, go):
        gradients.append(go[0])

    fw_handle = target_layer.register_forward_hook(fw_hook)
    bw_handle = target_layer.register_full_backward_hook(bw_hook)

    try:
        ct_vol = load_ct_volume(ct_path).unsqueeze(0).to(device)
        tumor_mask = load_ct_mask(ct_path)
        if tumor_mask.shape != tuple(ct_vol.shape[-3:]):
            raise ValueError(
                f"CT mask shape {tumor_mask.shape} does not match CT shape "
                f"{tuple(ct_vol.shape[-3:])}"
            )
        ct_vol.requires_grad_(True)

        model.zero_grad()
        if args.model_type in ("ct_baseline", "ct_student"):
            risk = model(ct_vol)
        else:
            pa_feat = (
                load_path_feature(pt_path).to(device)
                if (pt_path and Path(pt_path).exists())
                else torch.zeros(1, 1024, device=device)
            )
            risk_fused, risk_ct, _, _, _, _, _ = model(ct_vol, pa_feat)
            risk = risk_fused
        risk.backward()

        target_size = tuple(ct_vol.shape[-3:])
        positive_cam, negative_cam = compute_signed_cams(
            activations, gradients, target_size
        )
        slices, indices = collect_slices(
            positive_cam,
            negative_cam,
            ct_vol,
            args,
            mask_volume=tumor_mask,
        )
    finally:
        fw_handle.remove()
        bw_handle.remove()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for view, data in slices.items():
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(data["ct_slice"], cmap="gray")
        axes[0].set_title(f"CT {view}")

        axes[1].imshow(data["ct_slice"], cmap="gray")
        # Standard Grad-CAM: JET heatmap overlay
        cam = data["positive_cam"]  # Grad-CAM map (0-1)
        cam_uint8 = (cam * 255).astype(np.uint8)
        import cv2
        heatmap = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        alpha = 0.5
        ct_display = (np.stack([data["ct_slice"]] * 3, axis=-1) * 255).astype(np.uint8)
        overlay = (ct_display * (1 - alpha) + heatmap * alpha).astype(np.uint8)
        axes[1].imshow(overlay)
        axes[1].set_title(f"Grad-CAM\nrisk={risk.item():.3f}")

        for ax in axes:
            if np.any(data["tumor_mask"]):
                ax.contour(
                    data["tumor_mask"],
                    levels=[0.5],
                    colors=["yellow"],
                    linewidths=1.2,
                )
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(
            output_dir / f"{ct_id}_{view}_gradcam.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)

    print(
        f"  [OK] {ct_id} risk={risk.item():.3f} slices={indices} "
        f"cam_max={positive_cam.max():.4f}"
    )


# ============================================================
# Main
# ============================================================


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.output_dir is None:
        base = Path(args.results_root or args.checkpoint_root)
        args.output_dir = str(base.parent / "results" / "visualization" / "ct")
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    ckpt_path = find_fold_checkpoint(args.checkpoint_root, args.fold)
    if args.model_type in ("ct_baseline", "ct_student"):
        model, backbone = load_ct_model_for_gradcam(
            ckpt_path, args.pretrained_path, args.dropout, device
        )
    else:
        model, backbone = load_pact_model_for_gradcam(
            ckpt_path,
            args.pa_model,
            args.fusion_type,
            args.k,
            args.dropout,
            device,
            implementation=args.implementation,
        )

    if args.ct_id:
        gradcam_single_case(args.ct_id, model, backbone, args, output_dir, device)
    elif args.ct_ids:
        print(f"Using {len(args.ct_ids)} explicit CT IDs")
        for ctid in args.ct_ids:
            try:
                gradcam_single_case(ctid, model, backbone, args, output_dir, device)
            except Exception as e:
                print(f"  [Error] {ctid}: {e}")
    elif args.sample_n > 0:
        pairs = get_split_ids(
            args.data_dir, args.fold, args.roi_size, args.sample_split, args.cv_seed
        )
        ct_ids = [ctid for _, ctid in pairs]
        rng = random.Random(args.sample_seed)
        sampled = rng.sample(ct_ids, min(args.sample_n, len(ct_ids)))
        print(
            f"Sampled {len(sampled)} CTs from fold {args.fold} {args.sample_split} set (cv_seed={args.cv_seed})"
        )
        for ctid in sampled:
            try:
                gradcam_single_case(ctid, model, backbone, args, output_dir, device)
            except Exception as e:
                print(f"  [Error] {ctid}: {e}")
    else:
        print("Specify --ct_id or --sample_n > 0")

    print(f"\n[OK] Results saved to {output_dir}")


if __name__ == "__main__":
    main()
