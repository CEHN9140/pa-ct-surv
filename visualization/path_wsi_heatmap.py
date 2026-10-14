"""ABMIL attention heatmaps for pathology and PACT checkpoints."""

import argparse
import random
import sys
from pathlib import Path

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
    get_split_ids,
    load_visualization_case_file,
    load_h5_coords,
    load_path_feature,
    load_visualization_model,
    save_json,
    save_top_patches,
    validate_attention_weights,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_type", required=True, choices=["pathology", "pact"])
    parser.add_argument("--config", required=True, help="Path to run_config.yaml")
    parser.add_argument("--checkpoint_root", required=True)
    parser.add_argument("--fold", type=int, default=None, choices=range(5))
    parser.add_argument("--output_root", default="vis/pact_v5")
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--roi_size", type=int, default=None)
    parser.add_argument("--sample_n", type=int, default=0)
    parser.add_argument("--sample_split", choices=["train", "val"], default="val")
    parser.add_argument("--sample_seed", type=int, default=42)
    parser.add_argument("--case_file", default=None)
    parser.add_argument("--slide_id", default=None)
    parser.add_argument("--slide_ids", nargs="+", default=None)
    parser.add_argument("--wsi_dir", required=True)
    parser.add_argument(
        "--wsi_exts", nargs="+", default=[".svs", ".tif", ".tiff", ".ndpi", ".mrxs"]
    )
    parser.add_argument("--patch_size", type=int, default=256)
    parser.add_argument("--downsample", type=float, default=32)
    parser.add_argument("--alpha", type=float, default=0.60)
    parser.add_argument("--alpha_mode", choices=["attention", "coverage"], default="attention")
    parser.add_argument("--smooth_sigma", type=float, default=0.0)
    parser.add_argument("--normalize_mode", choices=["positive", "all"], default="positive")
    parser.add_argument("--clip_percentiles", nargs=2, type=float, default=[1.0, 99.0])
    parser.add_argument("--top_k", type=int, default=20)
    return parser.parse_args()


def _attention_output(model_type, model, features, device):
    if features.ndim != 2 or features.shape[1] != 1024:
        raise ValueError(f"Expected UNI features [N, 1024], got {tuple(features.shape)}")
    target = model.pa_branch if model_type == "pact" else model.mil
    with torch.no_grad():
        risk, _pooled, attention = target(features.unsqueeze(0).to(device))
    if attention.ndim == 3:
        if attention.shape[0] != 1 or attention.shape[2] != 1:
            raise ValueError(
                f"Only one attention branch is supported; got {tuple(attention.shape)}"
            )
        attention = attention[0, :, 0]
    attention = validate_attention_weights(attention, features.shape[0])
    return float(risk.reshape(-1)[0].cpu()), attention


def _case_output_dir(args, metadata, slide_id, ct_id):
    case_dir = Path(str(ct_id)) if args.model_type == "pact" else Path(str(slide_id))
    if args.model_type == "pact":
        case_dir = case_dir / "pathology"
    return (
        Path(args.output_root)
        / args.model_type
        / metadata["run_tag"]
        / f"seed{metadata['seed']}"
        / f"fold_{args.fold}"
        / case_dir
    )


def _split_note(row, fold):
    split = int(row["split"]) if "split" in row.index else None
    return {"split": split, "is_current_fold_validation": None if split is None else split == int(fold)}


def process_single_case(slide_id, args, model, metadata, device):
    roi_size = args.roi_size or metadata["roi_size"]
    data_dir = args.data_dir or metadata["data_dir"]
    row = find_case_row(data_dir, slide_id=slide_id, roi_size=roi_size)
    slide_id = str(row["pa_id"])
    features = load_path_feature(str(row["pa_path"]))
    h5_path = str(row["h5_path"])
    coords = load_h5_coords(h5_path)
    if features.shape[0] != coords.shape[0]:
        raise ValueError(
            f"Patch feature count ({features.shape[0]}) != H5 coordinate count ({coords.shape[0]})"
        )
    risk, attention = _attention_output(args.model_type, model, features, device)
    wsi_path = find_file_by_slide_id(slide_id, args.wsi_dir, args.wsi_exts)
    if wsi_path is None:
        raise FileNotFoundError(f"No WSI found for slide_id={slide_id} in {args.wsi_dir}")

    output_dir = _case_output_dir(args, metadata, slide_id, row["ct_id"])
    ensure_dir(output_dir / "top_patches")
    draw_heatmap_overlay(
        attention, h5_path, wsi_path,
        patch_size=args.patch_size, downsample=args.downsample,
        alpha=args.alpha, alpha_mode=args.alpha_mode,
        smooth_sigma=args.smooth_sigma, normalize_mode=args.normalize_mode,
        clip_percentiles=tuple(args.clip_percentiles),
        save_path=output_dir / "attention_heatmap.png",
        save_wsi_path=output_dir / "wsi.png",
    )
    save_top_patches(
        wsi_path, h5_path, attention, output_dir / "top_patches",
        top_k=args.top_k, patch_size=args.patch_size,
    )
    save_json(
        {
            "model_type": args.model_type,
            "run_tag": metadata["run_tag"], "seed": metadata["seed"], "fold": args.fold,
            "slide_id": slide_id, "ct_id": str(row["ct_id"]), "risk": risk,
            "attention_sum": float(attention.sum()), "num_patches": int(len(attention)),
            "wsi_path": str(wsi_path), "h5_path": h5_path,
            "config_path": metadata["config_path"], "checkpoint_path": metadata["checkpoint_path"],
            **_split_note(row, args.fold),
        },
        output_dir / "metadata.json",
    )
    print(f"[OK] {slide_id}: risk={risk:.4f} -> {output_dir}")


def _run_fold(args, fold, device):
    args.fold = fold
    model, metadata = load_visualization_model(
        args.model_type, args.config, args.checkpoint_root, fold, device
    )
    data_dir = args.data_dir or metadata["data_dir"]
    roi_size = args.roi_size or metadata["roi_size"]
    if args.case_file:
        ids = [pa_id for pa_id, _ in load_visualization_case_file(args.case_file, fold)]
    elif args.slide_id:
        ids = [args.slide_id]
    elif args.slide_ids:
        ids = args.slide_ids
    elif args.sample_n > 0:
        pairs = get_split_ids(data_dir, fold, roi_size, args.sample_split)
        rng = random.Random(args.sample_seed)
        ids = [sid for sid, _ in rng.sample(pairs, min(args.sample_n, len(pairs)))]
    else:
        raise ValueError("Specify --case_file, --slide_id, --slide_ids, or --sample_n > 0")

    print(f"Using {len(ids)} slide(s), model={args.model_type}, fold={fold}")
    failed = []
    succeeded = 0
    for slide_id in ids:
        try:
            process_single_case(slide_id, args, model, metadata, device)
            succeeded += 1
        except Exception as exc:
            print(f"[Error] {slide_id}: {exc}")
            failed.append(str(slide_id))
    summary_dir = (
        Path(args.output_root) / args.model_type / metadata["run_tag"]
        / f"seed{metadata['seed']}" / f"fold_{fold}"
    )
    ensure_dir(summary_dir)
    save_json({"total": len(ids), "succeeded": succeeded, "failed": failed}, summary_dir / "failed_cases.json")
    print(f"Completed fold_{fold}: {succeeded} succeeded, {len(failed)} failed")
    return failed


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.fold is None and not args.case_file:
        raise ValueError("Omit --fold only when using --case_file")
    folds = [args.fold] if args.fold is not None else range(5)
    failed = []
    for fold in folds:
        failed.extend(_run_fold(args, fold, device))
    if failed:
        print(f"Failed cases: {', '.join(failed)}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
