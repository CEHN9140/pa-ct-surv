"""
Pathology WSI attention heatmap visualization.

Supports single-case and val-set sampling modes.

Single case:
    python visualization/path_wsi_heatmap.py \
        --data_dir ... --h5_dir ... --checkpoint_root ... \
        --model_type pact --fold 0 --slide_id 1801006

Sample from validation set:
    python visualization/path_wsi_heatmap.py \
        --data_dir ... --h5_dir ... --checkpoint_root ... \
        --model_type pact --fold 0 \
        --sample_n 10 --sample_split val --sample_seed 42
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

from visualization.vis_utils import (
    draw_heatmap_overlay,
    ensure_dir,
    find_case_row,
    find_file_by_slide_id,
    find_fold_checkpoint,
    find_h5_by_slide_id,
    get_split_ids,
    load_h5_coords,
    load_pa_model,
    load_pact_model,
    load_path_feature,
    save_top_patches,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Pathology attention heatmap")
    parser.add_argument("--data_dir", default="/home/gly001/cqj/pa_ct_surv/data")
    parser.add_argument("--h5_dir", required=True, help="Directory with CLAM .h5 files")
    parser.add_argument("--checkpoint_root", required=True)
    parser.add_argument("--results_root", default=None)
    parser.add_argument(
        "--model_type",
        required=True,
        choices=["pa", "pact"],
    )
    parser.add_argument("--pa_model", default="abmil-cox-topk")
    parser.add_argument(
        "--implementation",
        default="original",
        choices=["original", "legacy"],
        help="PA implementation used by the checkpoint.",
    )
    parser.add_argument("--fusion", default="concat", choices=["concat", "bilinear"])
    parser.add_argument("--k", type=int, default=512)
    parser.add_argument("--fold", type=int, default=0)

    # Single case mode
    parser.add_argument("--slide_id", default=None, help="Single case: slide ID")
    parser.add_argument(
        "--slide_ids",
        nargs="+",
        default=None,
        help="Explicit list of slide IDs for aligned multi-model visualization.",
    )

    # Sampling mode
    parser.add_argument(
        "--sample_n",
        type=int,
        default=0,
        help="Number of cases to sample (0 = single case mode)",
    )
    parser.add_argument(
        "--sample_split",
        default="val",
        choices=["train", "val"],
        help="Which split to sample from",
    )
    parser.add_argument("--sample_seed", type=int, default=42)
    parser.add_argument(
        "--cv_seed",
        type=int,
        default=42,
        help="CV split seed (should match training seed)",
    )

    parser.add_argument("--roi_size", type=int, default=64)
    parser.add_argument("--output_dir", default=None)

    # WSI overlay
    parser.add_argument(
        "--wsi_dir",
        default=None,
        help="Directory with whole-slide images (.svs/.tif/.ndpi)",
    )
    parser.add_argument(
        "--wsi_exts", nargs="+", default=[".svs", ".tif", ".tiff", ".ndpi", ".mrxs"]
    )
    parser.add_argument("--patch_size", type=int, default=256)
    parser.add_argument("--downsample", type=int, default=32)
    parser.add_argument("--alpha", type=float, default=0.30)
    parser.add_argument(
        "--smooth_sigma",
        type=float,
        default=0.0,
        help="Gaussian smoothing sigma for attention mask. 0 keeps patch-block heatmap.",
    )
    parser.add_argument(
        "--normalize_mode",
        default="positive",
        choices=["positive", "all"],
        help=(
            "positive normalizes only nonzero attention values, better for top-k MIL; "
            "all normalizes across every patch."
        ),
    )
    parser.add_argument(
        "--clip_percentiles",
        nargs=2,
        type=float,
        default=[1.0, 99.0],
        metavar=("LOW", "HIGH"),
        help="Percentiles used for attention color normalization.",
    )
    parser.add_argument(
        "--alpha_mode",
        default="coverage",
        choices=["coverage", "attention"],
        help=(
            "coverage: fixed-alpha JET overlay on all tissue patches, closer to pa_ct heatmaps; "
            "attention: transparency scales with attention."
        ),
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=20,
        help="Number of top patches to save.",
    )
    return parser.parse_args()


def extract_attention_from_model(model, features, device):
    model.eval()
    with torch.no_grad():
        if isinstance(features, np.ndarray):
            features = torch.as_tensor(features, dtype=torch.float32)
        features = features.to(device)
        if features.dim() == 2:
            features = features.unsqueeze(0)
        output = model(features)
        if isinstance(output, tuple):
            risk, fea = output[0], output[1]
            if len(output) >= 3:
                att = output[2].squeeze().cpu().numpy()
                return (
                    float(risk.squeeze().cpu().item()),
                    fea.squeeze().cpu().numpy(),
                    att,
                )
            raise RuntimeError("Model did not return attention")
        raise RuntimeError("Model returned scalar")


def _sample_slide_ids(args):
    """Sample slide IDs from fold split using vis_utils.get_split_ids."""
    pairs = get_split_ids(
        args.data_dir,
        args.fold,
        args.roi_size,
        args.sample_split,
        args.cv_seed,
    )
    slide_ids = [sid for sid, _ctid in pairs]
    rng = random.Random(args.sample_seed)
    return rng.sample(slide_ids, min(args.sample_n, len(slide_ids)))


def process_single_case(slide_id, args, model, output_dir, device):
    """Generate heatmap for a single slide."""
    row = find_case_row(args.data_dir, slide_id=slide_id, roi_size=args.roi_size)
    slide_id = str(row["slide_id"])
    pt_path = str(row["pt_path"])
    label = int(row["label"])
    time = float(row["time"])
    print(f"\nSlide: {slide_id} | label={label} | time={time:.1f}mo")

    features = load_path_feature(pt_path)
    print(f"Features: {features.shape}")

    h5_path = find_h5_by_slide_id(args.h5_dir, slide_id)
    coords = load_h5_coords(h5_path)
    print(f"Coords: {coords.shape}")

    if features.shape[0] != coords.shape[0]:
        print(
            f"[Skip] feature count mismatch: {features.shape[0]} vs {coords.shape[0]}"
        )
        return

    if args.model_type == "pact":
        risk, pa_fea, att = extract_attention_from_model(
            model.pa_branch, features, device
        )
    else:
        target = model.mil if hasattr(model, "mil") else model
        risk, pa_fea, att = extract_attention_from_model(target, features, device)

    print(f"Risk: {risk:.4f} | Att range: [{att.min():.4f}, {att.max():.4f}]")

    # ── WSI overlay heatmap ──
    if args.wsi_dir:
        wsi_path = find_file_by_slide_id(slide_id, args.wsi_dir, args.wsi_exts)
        if wsi_path is None:
            print(f"  [Skip overlay] No WSI found for {slide_id} in {args.wsi_dir}")
        else:
            print(f"  WSI: {wsi_path}")
            try:
                heatmap_path = output_dir / f"{slide_id}_heatmap.png"
                wsi_thumb_path = output_dir / f"{slide_id}_wsi.png"

                draw_heatmap_overlay(
                    att,
                    h5_path,
                    wsi_path,
                    patch_size=args.patch_size,
                    downsample=args.downsample,
                    alpha=args.alpha,
                    alpha_mode=args.alpha_mode,
                    smooth_sigma=args.smooth_sigma,
                    normalize_mode=args.normalize_mode,
                    clip_percentiles=tuple(args.clip_percentiles),
                    save_path=heatmap_path,
                    save_wsi_path=wsi_thumb_path,
                )
                saved_msg = f"    -> {heatmap_path.name}, {wsi_thumb_path.name}"
                top_dir = output_dir / "top_patches" / slide_id
                save_top_patches(
                    wsi_path,
                    h5_path,
                    att,
                    top_dir,
                    top_k=args.top_k,
                    patch_size=args.patch_size,
                )
                saved_msg += f", top patches in {top_dir}"
                print(saved_msg)
            except Exception as exc:
                print(f"  [Overlay failed] {slide_id}: {exc}")


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.output_dir is None:
        base = Path(args.results_root or args.checkpoint_root)
        args.output_dir = str(base.parent / "results" / "visualization" / "pathology")
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    ckpt_path = find_fold_checkpoint(args.checkpoint_root, args.fold)
    if args.model_type == "pa":
        model = load_pa_model(
            ckpt_path,
            args.pa_model,
            args.k,
            device,
            implementation=args.implementation,
        )
    else:
        model = load_pact_model(
            ckpt_path,
            args.pa_model,
            args.fusion,
            args.k,
            0.5,
            device,
            implementation=args.implementation,
        )

    if args.slide_id:
        process_single_case(args.slide_id, args, model, output_dir, device)
    elif args.slide_ids:
        print(f"Using {len(args.slide_ids)} explicit slide IDs")
        for sid in args.slide_ids:
            try:
                process_single_case(sid, args, model, output_dir, device)
            except Exception as e:
                print(f"  [Error] {sid}: {e}")
    elif args.sample_n > 0:
        slide_ids = _sample_slide_ids(args)
        print(
            f"Sampled {len(slide_ids)} slides from fold {args.fold} {args.sample_split} set (cv_seed={args.cv_seed}):"
        )
        for sid in slide_ids:
            try:
                process_single_case(sid, args, model, output_dir, device)
            except Exception as e:
                print(f"  [Error] {sid}: {e}")
    else:
        print("Please specify either --slide_id or --sample_n > 0")

    print(f"\n[OK] Outputs saved to {output_dir}")


if __name__ == "__main__":
    main()
