"""3D CT Grad-CAM visualization for CT, PACT, and CT-student models."""

import argparse
import random
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parents[1]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from visualization.vis_utils import (
    ensure_dir,
    find_case_row,
    get_split_ids,
    load_ct_mask,
    load_ct_volume,
    load_visualization_model,
    save_json,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_type", required=True, choices=["ct", "pact", "ct_student"])
    parser.add_argument("--config", required=True, help="Path to run_config.yaml")
    parser.add_argument("--checkpoint_root", required=True)
    parser.add_argument("--fold", type=int, default=0, choices=range(5))
    parser.add_argument("--output_root", default="vis/pact_v5")
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--roi_size", type=int, default=None)
    parser.add_argument("--ct_id", default=None)
    parser.add_argument("--ct_ids", nargs="+", default=None)
    parser.add_argument("--sample_n", type=int, default=0)
    parser.add_argument("--sample_split", choices=["train", "val"], default="val")
    parser.add_argument("--sample_seed", type=int, default=42)
    parser.add_argument("--target_layer", default="layer3.1.conv2")
    parser.add_argument(
        "--slice_mode", choices=["tumor", "maxcam", "center", "fixed"], default="tumor"
    )
    parser.add_argument("--axial_slice", type=int, default=None)
    parser.add_argument("--allow_mask_fallback", action="store_true")
    return parser.parse_args()


def find_target_layer(backbone, target_layer_name):
    modules = dict(backbone.named_modules())
    actual_name = target_layer_name
    if target_layer_name == "layer3.1.conv2" and target_layer_name not in modules:
        # ResNet10 has one block in layer3, so use its final convolution.
        if "layer3.0.conv2" in modules:
            actual_name = "layer3.0.conv2"
    if actual_name not in modules:
        available = [
            name for name, module in backbone.named_modules()
            if isinstance(module, torch.nn.Conv3d)
        ]
        raise ValueError(
            f"Target layer '{target_layer_name}' not found. Conv3d layers: {available}"
        )
    target = modules[actual_name]
    if not isinstance(target, torch.nn.Conv3d):
        raise ValueError(f"Target layer '{actual_name}' is not Conv3d")
    return target, actual_name


def compute_positive_gradcam(backbone, risk, activations, gradients, target_size):
    if not activations or not gradients:
        raise RuntimeError("Grad-CAM hooks did not capture activations and gradients")
    activation = activations[0]
    gradient = gradients[0]
    weights = gradient.mean(dim=(2, 3, 4), keepdim=True)
    cam = (weights * activation).sum(dim=1, keepdim=True)
    cam = F.relu(cam)
    cam = F.interpolate(cam, size=target_size, mode="trilinear", align_corners=False)
    cam = cam.detach().squeeze().cpu().numpy().astype(np.float32)
    cam_max = float(cam.max())
    if cam_max > 1e-8:
        cam /= cam_max
    return cam, cam_max


def _axial_index(mask, cam, args, mask_fallback_reason=None):
    if args.slice_mode == "tumor":
        area = mask.reshape(mask.shape[0], -1).sum(axis=1)
        if area.max() <= 0:
            if mask_fallback_reason is not None:
                return int(cam.shape[0] // 2)
            raise ValueError("Tumor mask is empty; use --allow_mask_fallback to use center")
        return int(np.argmax(area))
    if args.slice_mode == "maxcam":
        return int(np.argmax(cam.reshape(cam.shape[0], -1).sum(axis=1)))
    if args.slice_mode == "center":
        return int(cam.shape[0] // 2)
    if args.axial_slice is None:
        raise ValueError("--slice_mode fixed requires --axial_slice")
    return int(np.clip(args.axial_slice, 0, cam.shape[0] - 1))


def _risk_from_model(model_type, model, ct_volume):
    if model_type in {"ct", "ct_student"}:
        return model(ct_volume).reshape(-1)[0]
    backbone = model.ct_backbone
    features = backbone.extract_features(ct_volume)
    return backbone.risk_forward(features).reshape(-1)[0]


def _output_dir(args, metadata, ct_id):
    case_dir = Path(str(ct_id))
    if args.model_type == "pact":
        case_dir = case_dir / "ct"
    return (
        Path(args.output_root)
        / args.model_type
        / metadata["run_tag"]
        / f"seed{metadata['seed']}"
        / f"fold_{args.fold}"
        / case_dir
    )


def gradcam_single_case(ct_id, args, model, metadata, device):
    roi_size = args.roi_size or metadata["roi_size"]
    data_dir = args.data_dir or metadata["data_dir"]
    row = find_case_row(data_dir, ct_id=ct_id, roi_size=roi_size)
    ct_id = str(row["ct_id"])
    ct_path = str(row["ct_path"])
    ct_volume = load_ct_volume(ct_path).unsqueeze(0).to(device)
    ct_volume.requires_grad_(True)
    if tuple(ct_volume.shape[-3:]) != (roi_size, roi_size, roi_size):
        raise ValueError(f"CT shape {tuple(ct_volume.shape)} does not match ROI {roi_size}")

    mask_fallback_reason = None
    try:
        tumor_mask = load_ct_mask(ct_path)
        if tumor_mask.shape != tuple(ct_volume.shape[-3:]):
            raise ValueError(
                f"Mask shape {tumor_mask.shape} != CT shape {tuple(ct_volume.shape[-3:])}"
            )
        if not np.any(tumor_mask):
            raise ValueError("Tumor mask is empty")
    except Exception as exc:
        if not args.allow_mask_fallback:
            raise
        tumor_mask = np.zeros(tuple(ct_volume.shape[-3:]), dtype=np.uint8)
        mask_fallback_reason = str(exc)

    backbone = model.ct if args.model_type != "pact" else model.ct_backbone
    target_layer, effective_target_layer = find_target_layer(backbone, args.target_layer)
    activations, gradients = [], []

    def forward_hook(_module, _inputs, output):
        activations.append(output)
        output.register_hook(lambda grad: gradients.append(grad))

    handle = target_layer.register_forward_hook(forward_hook)
    try:
        model.zero_grad(set_to_none=True)
        risk = _risk_from_model(args.model_type, model, ct_volume)
        risk.backward()
        cam, raw_cam_max = compute_positive_gradcam(
            backbone, risk, activations, gradients, tuple(ct_volume.shape[-3:])
        )
    finally:
        handle.remove()

    axial = _axial_index(tumor_mask, cam, args, mask_fallback_reason)
    ct_slice = ct_volume.detach().squeeze().cpu().numpy()[axial]
    cam_slice = cam[axial]
    mask_slice = tumor_mask[axial]
    output_dir = _output_dir(args, metadata, ct_id)
    ensure_dir(output_dir)

    base = np.clip(ct_slice, 0.0, 1.0).astype(np.float32)
    base_rgb = np.repeat((base * 255).astype(np.uint8)[..., None], 3, axis=2)
    heatmap = cv2.cvtColor(
        cv2.applyColorMap((cam_slice * 255).astype(np.uint8), cv2.COLORMAP_JET),
        cv2.COLOR_BGR2RGB,
    )
    alpha = (0.75 * np.clip(cam_slice, 0.0, 1.0))[..., None]
    overlay = (
        base_rgb.astype(np.float32) * (1.0 - alpha)
        + heatmap.astype(np.float32) * alpha
    ).astype(np.uint8)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for axis, image, title in (
        (axes[0], base, "CT axial"),
        (axes[1], overlay, f"Grad-CAM | risk={float(risk.detach()):.3f}"),
    ):
        axis.imshow(image, cmap="gray" if image.ndim == 2 else None, vmin=0, vmax=1)
        if np.any(mask_slice):
            axis.contour(mask_slice, levels=[0.5], colors="yellow", linewidths=1.2)
        axis.set_title(title)
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(output_dir / "axial_gradcam.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    save_json(
        {
            "model_type": args.model_type,
            "run_tag": metadata["run_tag"], "seed": metadata["seed"], "fold": args.fold,
            "ct_id": ct_id, "risk": float(risk.detach()),
            "target_layer": effective_target_layer, "requested_target_layer": args.target_layer,
            "slice_mode": args.slice_mode,
            "axial_slice": axial, "cam_max_raw": raw_cam_max,
            "cam_max_normalized": float(cam.max()),
            "cam_is_all_zero": bool(not np.any(cam)),
            "ct_path": ct_path, "mask_fallback_reason": mask_fallback_reason,
            "config_path": metadata["config_path"], "checkpoint_path": metadata["checkpoint_path"],
            "split": int(row["split"]) if "split" in row.index else None,
        },
        output_dir / "metadata.json",
    )
    print(f"[OK] {ct_id}: risk={float(risk.detach()):.4f}, axial={axial} -> {output_dir}")


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, metadata = load_visualization_model(
        args.model_type, args.config, args.checkpoint_root, args.fold, device
    )
    data_dir = args.data_dir or metadata["data_dir"]
    roi_size = args.roi_size or metadata["roi_size"]
    if args.ct_id:
        ids = [args.ct_id]
    elif args.ct_ids:
        ids = args.ct_ids
    elif args.sample_n > 0:
        pairs = get_split_ids(data_dir, args.fold, roi_size, args.sample_split)
        rng = random.Random(args.sample_seed)
        ids = [ctid for _, ctid in rng.sample(pairs, min(args.sample_n, len(pairs)))]
    else:
        raise ValueError("Specify --ct_id, --ct_ids, or --sample_n > 0")

    print(f"Using {len(ids)} CT case(s), model={args.model_type}, fold={args.fold}")
    failed = []
    succeeded = 0
    for ct_id in ids:
        try:
            gradcam_single_case(ct_id, args, model, metadata, device)
            succeeded += 1
        except Exception as exc:
            print(f"[Error] {ct_id}: {exc}")
            failed.append(str(ct_id))
    summary_dir = (
        Path(args.output_root)
        / args.model_type
        / metadata["run_tag"]
        / f"seed{metadata['seed']}"
        / f"fold_{args.fold}"
    )
    ensure_dir(summary_dir)
    save_json(
        {"total": len(ids), "succeeded": succeeded, "failed": failed},
        summary_dir / "failed_cases.json",
    )
    print(f"Completed: {succeeded} succeeded, {len(failed)} failed")
    if failed:
        print(f"Failed cases: {', '.join(failed)}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
