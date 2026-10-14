"""
Shared utilities for visualization scripts.

Usage:
    from visualization.vis_utils import (
        ensure_dir, load_label_csv, load_ct_volume, load_path_feature,
        load_ct_model, load_pa_model, load_pact_model,
        compute_km_groups, collect_fold_cindex, set_plot_style, save_figure,
        draw_heatmap_overlay, save_top_patches, find_file_by_slide_id,
    )
"""

import json
import os
from pathlib import Path

import cv2
import h5py
import numpy as np
import openslide
import pandas as pd
import torch
import yaml
from final_utils import cv_fold_indices, locked_split_indices
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test
from PIL import Image

# ============================================================
# 1.1 Path & config tools
# ============================================================


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def find_fold_checkpoint(checkpoint_root, fold):
    ckpt = Path(checkpoint_root) / f"fold_{fold}" / "best_model.pth"
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    return str(ckpt)


def find_case_row(data_dir, case_id=None, slide_id=None, ct_id=None, roi_size=64):
    label_file = Path(data_dir) / f"all_label_roi{roi_size}.csv"
    if not label_file.exists():
        raise FileNotFoundError(f"Label file not found: {label_file}")
    df = pd.read_csv(label_file)
    if case_id:
        parts = str(case_id).split("|")
        if len(parts) == 2:
            slide_id = parts[0]
            ct_id = parts[1]
    if slide_id:
        df = df[df["pa_id"].astype(str) == str(slide_id)]
    if ct_id:
        df = df[df["ct_id"].astype(str) == str(ct_id)]
    if len(df) == 0:
        raise ValueError(
            f"No row found for pa_id={slide_id}, ct_id={ct_id} in {label_file}"
        )
    return df.iloc[0]


# ============================================================
# 1.2 Data loading tools
# ============================================================


def load_label_csv(data_dir, roi_size=64):
    label_file = Path(data_dir) / f"all_label_roi{roi_size}.csv"
    if not label_file.exists():
        raise FileNotFoundError(f"Label file not found: {label_file}")
    return pd.read_csv(label_file)


def load_ct_volume(ct_image_path):
    ct = np.load(ct_image_path).astype(np.float32)
    if ct.ndim == 3:
        ct = ct[np.newaxis, ...]
    return torch.as_tensor(ct, dtype=torch.float32)


def load_ct_mask(ct_image_path):
    image_path = Path(ct_image_path)
    mask_path = image_path.parent.parent / "mask" / image_path.name
    if not mask_path.exists():
        raise FileNotFoundError(f"CT mask not found: {mask_path}")

    mask = np.load(mask_path)
    if mask.ndim == 4 and mask.shape[0] == 1:
        mask = mask[0]
    if mask.ndim != 3:
        raise ValueError(f"Invalid CT mask shape {mask.shape}: {mask_path}")
    if not np.isin(mask, (0, 1)).all():
        raise ValueError(f"CT mask must be binary: {mask_path}")
    return mask.astype(np.uint8)


def load_path_feature(pt_path):
    return torch.load(pt_path, map_location="cpu", weights_only=True).float()


def load_h5_coords(h5_path):
    with h5py.File(h5_path, "r") as f:
        keys = list(f.keys())
        print(f"[h5] Keys in {h5_path}: {keys}")
        coords = f["coords"][:]
        if "features" in keys:
            features = f["features"][:]
            print(f"  coords: {coords.shape}, features: {features.shape}")
            if coords.shape[0] != features.shape[0]:
                raise ValueError(
                    f"coords ({coords.shape[0]}) != features ({features.shape[0]}) in {h5_path}"
                )
    return coords


def validate_attention_weights(att_weights, n_patches):
    """Validate one ABMIL attention branch without silently truncating it."""
    att = _to_numpy_1d(att_weights)
    if att.shape[0] != int(n_patches):
        raise ValueError(
            f"Attention count ({att.shape[0]}) != patch count ({n_patches})"
        )
    if not np.isfinite(att).all():
        raise ValueError("Attention contains NaN or Inf")
    if (att < -1e-6).any():
        raise ValueError("Attention contains negative weights")
    total = float(att.sum())
    if not np.isclose(total, 1.0, atol=1e-4):
        raise ValueError(f"Attention weights must sum to 1; got {total:.8f}")
    return att.astype(np.float32)


def find_h5_by_slide_id(h5_dir, slide_id):
    h5_dir = Path(h5_dir)
    exact = h5_dir / f"{slide_id}.h5"
    if exact.exists():
        return str(exact)
    candidates = list(h5_dir.glob(f"*{slide_id}*.h5"))
    if len(candidates) == 0:
        raise FileNotFoundError(
            f"No .h5 file found for slide_id={slide_id} in {h5_dir}"
        )
    if len(candidates) > 1:
        print(
            f"[Warning] Multiple h5 matches for {slide_id}: {[c.name for c in candidates]}"
        )
    return str(candidates[0])


# ============================================================
# 1.3 Model loading tools
# ============================================================


def _load_checkpoint_strict(ckpt_path, model, device):
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    if not isinstance(state, dict):
        raise TypeError(f"Checkpoint must contain a state dict: {ckpt_path}")
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            f"Checkpoint/configuration mismatch for {ckpt_path}: {exc}"
        ) from exc
    return model


def resolve_config_path(config=None, checkpoint_root=None):
    """Resolve the run_config.yaml associated with a visualization run."""
    if config:
        path = Path(config)
        if not path.is_file():
            raise FileNotFoundError(f"Run config not found: {path}")
        return path
    if checkpoint_root:
        root = Path(checkpoint_root)
        candidates = [root / "run_config.yaml", root.parent / "run_config.yaml"]
        for candidate in candidates:
            if candidate.is_file():
                return candidate
    raise ValueError("Provide --config pointing to run_config.yaml")


def _config_data_dir(config):
    data_dir = config.get("data_dir")
    if not data_dir:
        raise ValueError("run_config.yaml is missing data_dir")
    return Path(data_dir)


def _config_roi_size(config):
    if "ct_roi_size" in config:
        return int(config["ct_roi_size"])
    teacher = config.get("teacher") or {}
    if "ct_roi_size" in teacher:
        return int(teacher["ct_roi_size"])
    raise ValueError("run_config.yaml is missing ct_roi_size")


def _run_tag(checkpoint_root, config_path):
    root = Path(checkpoint_root) if checkpoint_root else config_path.parent
    return root.name if root.name.startswith("seed") is False else root.parent.name


def load_visualization_model(model_type, config_path, checkpoint_root, fold, device):
    """Build and strictly load one of the four supported visualization models."""
    config_path = resolve_config_path(config_path, checkpoint_root)
    config = load_yaml(config_path) or {}
    checkpoint_root = Path(
        checkpoint_root or config.get("checkpoint_root", "")
    )
    if not checkpoint_root:
        raise ValueError("checkpoint_root is required")
    ckpt_path = checkpoint_root / f"fold_{int(fold)}" / "best_model.pth"

    from model.build import CT_Model, Pa_CT_Model, Pa_Model

    if model_type == "pathology":
        pa_model = config.get("pa_model", "abmil")
        if pa_model not in {"abmil", "abmil-topk", "abmil_randsample"}:
            raise ValueError(
                f"Pathology attention visualization does not support pa_model={pa_model}"
            )
        model = Pa_Model(
            model_name=pa_model,
            feature_dim=1024,
            k=config.get("k"),
            abmil_dropout=float(config.get("dropout", 0.0)),
            attention_branches=int(config.get("attention_branches", 1)),
        )
    elif model_type == "pact":
        teacher = config.get("teacher") or {}
        pa_model = config.get("pa_model", teacher.get("pa_model", "abmil"))
        ct_model = config.get("ct_model", teacher.get("ct_model", "resnet18"))
        model = Pa_CT_Model(
            pa_model_name=pa_model,
            ct_model_name=ct_model,
            ct_pretrained_path=None,
            pa_topk=config.get("k", teacher.get("k")),
            fusion_type=config.get("fusion_type", teacher.get("fusion_type", "concat")),
            fusion_dropout=float(config.get("fusion_dropout", 0.0)),
            norm=config.get("norm", "none"),
        )
    elif model_type == "ct":
        model = CT_Model(
            model_name=config.get("ct_model", "resnet18"),
            pretrained_path=None,
            freeze_backbone=bool(config.get("freeze_backbone", False)),
            dropout=float(config.get("dropout", 0.0)),
        )
    elif model_type == "ct_student":
        model = CT_Model(
            model_name=config.get("student_model", "resnet18"),
            pretrained_path=None,
            freeze_backbone=False,
            dropout=float(config.get("student_dropout", 0.0)),
        )
    else:
        raise ValueError(f"Unsupported visualization model_type={model_type}")

    model = _load_checkpoint_strict(ckpt_path, model, device)
    model.to(device)
    model.eval()
    metadata = {
        "model_type": model_type,
        "config_path": str(config_path),
        "checkpoint_path": str(ckpt_path),
        "checkpoint_root": str(checkpoint_root),
        "run_tag": _run_tag(checkpoint_root, config_path),
        "seed": int(config.get("seed", 42)),
        "data_dir": str(_config_data_dir(config)),
        "roi_size": _config_roi_size(config),
        "config": config,
    }
    return model, metadata


def load_ct_model(ckpt_path, pretrained_path=None, dropout=0.0, device=None):
    """Backward-compatible strict CT loader using the current CT_Model API."""
    from model.build import CT_Model

    device = device or torch.device("cpu")
    model = CT_Model(model_name="resnet18", pretrained_path=None, dropout=dropout)
    _load_checkpoint_strict(ckpt_path, model, device)
    return model.to(device).eval()


def load_pa_model(ckpt_path, pa_model_name, k, device, implementation=None):
    from model.build import Pa_Model

    model = Pa_Model(model_name=pa_model_name, feature_dim=1024, k=k)
    _load_checkpoint_strict(ckpt_path, model, device)
    return model.to(device).eval()


def load_pact_model(ckpt_path, pa_model_name, fusion_type, k, ct_dropout, device, implementation=None):
    from model.build import Pa_CT_Model

    model = Pa_CT_Model(
        pa_model_name=pa_model_name,
        ct_model_name="resnet18",
        ct_pretrained_path=None,
        pa_topk=k,
        fusion_type=fusion_type,
        fusion_dropout=0.0,
        norm="none",
    )
    _load_checkpoint_strict(ckpt_path, model, device)
    return model.to(device).eval()


# ============================================================
# 1.4 Survival analysis tools
# ============================================================


def compute_km_groups(train_df, val_df, event_col="dfs.status", time_col="dfs.month"):
    cutoff = float(train_df["risk_score"].median())
    high = val_df[val_df["risk_score"] >= cutoff]
    low = val_df[val_df["risk_score"] < cutoff]
    try:
        lr_result = logrank_test(
            high[time_col], low[time_col], high[event_col], low[event_col]
        )
        logrank_p = float(lr_result.p_value)
    except Exception:
        logrank_p = np.nan
    try:
        cph = CoxPHFitter()
        df_hr = val_df[[time_col, event_col, "risk_group_binary"]].copy()
        cph.fit(df_hr, duration_col=time_col, event_col=event_col)
        row = cph.summary.loc["risk_group_binary"]
        hr = float(np.exp(row["coef"]))
        hr_lo = float(np.exp(row["coef lower 95%"]))
        hr_hi = float(np.exp(row["coef upper 95%"]))
    except Exception:
        hr, hr_lo, hr_hi = np.nan, np.nan, np.nan
    return cutoff, high, low, logrank_p, hr, hr_lo, hr_hi


def collect_fold_cindex(result_root, seeds=(42, 123, 2024)):
    rows = []
    if "{seed}" in str(result_root):
        for seed in seeds:
            results_dir = Path(str(result_root).format(seed=seed)) / "results"
            metrics_path = results_dir / "cv_fold_metrics.csv"
            if metrics_path.exists():
                df = pd.read_csv(metrics_path)
                for _, r in df.iterrows():
                    rows.append(
                        {
                            "seed": seed,
                            "fold": int(r["fold"]),
                            "cindex": float(r["cindex"]),
                        }
                    )
    else:
        for seed in seeds:
            seed_dir = Path(result_root) / f"seed{seed}"
            if not seed_dir.exists():
                continue
            metrics_path = seed_dir / "results" / "cv_fold_metrics.csv"
            if metrics_path.exists():
                df = pd.read_csv(metrics_path)
                for _, r in df.iterrows():
                    rows.append(
                        {
                            "seed": seed,
                            "fold": int(r["fold"]),
                            "cindex": float(r["cindex"]),
                        }
                    )
                continue
            for subdir in seed_dir.iterdir():
                if not subdir.is_dir():
                    continue
                nested = subdir / "results" / "cv_fold_metrics.csv"
                if nested.exists():
                    df = pd.read_csv(nested)
                    for _, r in df.iterrows():
                        rows.append(
                            {
                                "seed": seed,
                                "fold": int(r["fold"]),
                                "cindex": float(r["cindex"]),
                            }
                        )
                    break
    return pd.DataFrame(rows)


def get_split_ids(data_dir, fold, roi_size, sample_split="val", cv_seed=42):
    label_file = Path(data_dir) / f"all_label_roi{roi_size}.csv"
    if not label_file.exists():
        raise FileNotFoundError(label_file)
    df = pd.read_csv(label_file)
    if "split" not in df.columns:
        raise ValueError("Dataset CSV missing split column; run preprocess_data.py")
    split_values = pd.to_numeric(df["split"], errors="coerce")
    if sample_split == "val":
        split_df = df.loc[split_values == int(fold)]
    elif sample_split == "train":
        split_df = df.loc[(split_values >= 0) & (split_values != int(fold))]
    else:
        raise ValueError(f"Unsupported sample_split={sample_split}")
    return list(zip(split_df["pa_id"].values, split_df["ct_id"].values))


def load_visualization_case_file(case_file, fold):
    """Load fold-matched cases from a tab-separated vis file."""
    rows = []
    with Path(case_file).open() as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split("\t")
            if fields[0] == "fold":
                continue
            if len(fields) != 3:
                raise ValueError(f"Invalid case file line {line_number}: {line}")
            row_fold, pa_id, ct_id = fields
            if int(row_fold) == int(fold):
                rows.append((pa_id, ct_id))
    if not rows:
        raise ValueError(f"No cases found for fold={fold} in {case_file}")
    return rows


# ============================================================
# 1.5 Plotting tools
# ============================================================


def set_plot_style():
    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
        }
    )
    return plt


def save_figure(fig, path):
    fig.savefig(path, dpi=300, bbox_inches="tight")
    fig.clear()
    import matplotlib.pyplot as plt

    plt.close(fig)


def normalize_array(x, method="minmax"):
    x = np.asarray(x, dtype=np.float64)
    if method == "minmax":
        x_min, x_max = x.min(), x.max()
        if x_max - x_min < 1e-8:
            return np.zeros_like(x)
        return (x - x_min) / (x_max - x_min)
    raise ValueError(f"Unknown normalization method: {method}")


# ============================================================
# 1.6 WSI overlay heatmap tools (migrated from pa_ct/path_vis_utils.py)
# ============================================================


def _to_numpy_1d(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float32).reshape(-1)


def normalize_attention(att_weights, clip_percentiles=(1, 99), mode="positive"):
    att = np.nan_to_num(_to_numpy_1d(att_weights), nan=0.0, posinf=0.0, neginf=0.0)
    if att.size == 0:
        return att

    if mode == "positive":
        ref = att[att > 0]
        if ref.size == 0:
            return np.zeros_like(att, dtype=np.float32)
    elif mode == "all":
        ref = att
    else:
        raise ValueError(f"Unsupported attention normalize mode: {mode}")

    lo, hi = np.percentile(ref, clip_percentiles)
    if hi <= lo:
        lo, hi = float(ref.min()), float(ref.max())
    if hi <= lo:
        return np.zeros_like(att, dtype=np.float32)

    norm = ((np.clip(att, lo, hi) - lo) / (hi - lo + 1e-8)).astype(np.float32)
    if mode == "positive":
        norm[att <= 0] = 0.0
    return norm


def load_patch_coords(h5_path, coord_key="coords"):
    with h5py.File(h5_path, "r") as f:
        if coord_key not in f:
            raise KeyError(f"Cannot find '{coord_key}' in {h5_path}")
        coords = np.asarray(f[coord_key])
    if coords.ndim != 2 or coords.shape[1] < 2:
        raise ValueError(f"Invalid coords shape in {h5_path}: {coords.shape}")
    return coords[:, :2].astype(np.int64)


def find_file_by_slide_id(slide_id, root_dir, exts):
    root = Path(root_dir)
    sid = Path(str(slide_id)).stem
    names = [str(slide_id), sid]
    for name in names:
        for ext in exts:
            path = root / f"{name}{ext}"
            if path.exists():
                return path
    hits = []
    for name in names:
        for ext in exts:
            hits.extend(root.rglob(f"{name}*{ext}"))
    return sorted(hits)[0] if hits else None


def draw_heatmap_overlay(
    att_weights,
    h5_path,
    wsi_path,
    patch_size=256,
    downsample=32,
    alpha=0.60,
    alpha_mode="coverage",
    smooth_sigma=0.0,
    normalize_mode="positive",
    clip_percentiles=(1, 99),
    save_path=None,
    save_wsi_path=None,
    coord_key="coords",
):
    """Generate WSI overlay heatmap from attention weights.

    Args:
        att_weights: 1D attention array [N]
        h5_path: CLAM-style h5 file with patch coords
        wsi_path: openslide-compatible WSI file
        patch_size: size of each patch in pixels
        downsample: factor to shrink heatmap resolution
        alpha: blend factor (0=WSI only, 1=heatmap only)
        alpha_mode: "coverage" blends all tissue-covered patches with fixed alpha;
            "attention" scales transparency by attention intensity.
        smooth_sigma: Gaussian smoothing sigma. 0 disables smoothing and preserves
            patch-level blocks.
        normalize_mode: "positive" normalizes only nonzero attention values; "all"
            normalizes across all patches.
        clip_percentiles: percentile range used before min-max normalization.
        save_path: if provided, save overlay image here
        save_wsi_path: if provided, save WSI thumbnail here
    Returns:
        overlay image (BGR format)
    """
    coords = load_patch_coords(h5_path, coord_key)
    att = normalize_attention(
        att_weights,
        clip_percentiles=clip_percentiles,
        mode=normalize_mode,
    )
    if len(coords) != len(att):
        raise ValueError(
            f"coords ({len(coords)}) != attention ({len(att)}); refusing truncation"
        )
    if len(coords) == 0:
        raise ValueError("No patch coordinates or attention weights to draw")

    min_x, min_y = coords.min(axis=0)
    max_x, max_y = coords.max(axis=0) + patch_size
    content_w = int(max_x - min_x)
    content_h = int(max_y - min_y)
    slide = openslide.OpenSlide(str(wsi_path))
    level = slide.get_best_level_for_downsample(float(downsample))
    level_downsample = float(slide.level_downsamples[level])
    target_w = max(1, int(np.ceil(content_w / level_downsample)))
    target_h = max(1, int(np.ceil(content_h / level_downsample)))

    mask = np.zeros((target_h, target_w), dtype=np.float32)
    count = np.zeros((target_h, target_w), dtype=np.float32)
    patch_ds = max(1, int(np.ceil(patch_size / level_downsample)))

    for (x, y), value in zip(coords, att):
        x0 = int((x - min_x) / level_downsample)
        y0 = int((y - min_y) / level_downsample)
        x1 = min(target_w, x0 + patch_ds)
        y1 = min(target_h, y0 + patch_ds)
        if x0 < target_w and y0 < target_h and x1 > 0 and y1 > 0:
            ys, ye = max(0, y0), y1
            xs, xe = max(0, x0), x1
            mask[ys:ye, xs:xe] += float(value)
            count[ys:ye, xs:xe] += 1.0
    mask = np.divide(mask, np.maximum(count, 1.0))
    if smooth_sigma and smooth_sigma > 0:
        mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=float(smooth_sigma))
    mask = np.clip(mask, 0.0, 1.0)
    coverage = (count > 0).astype(np.float32)
    heatmap_color = cv2.applyColorMap((mask * 255).astype(np.uint8), cv2.COLORMAP_JET)

    try:
        region = slide.read_region(
            (int(min_x), int(min_y)), level, (target_w, target_h)
        ).convert("RGB")
    finally:
        slide.close()
    thumbnail = cv2.cvtColor(np.asarray(region), cv2.COLOR_RGB2BGR)

    if save_wsi_path is not None:
        p = Path(save_wsi_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(p), thumbnail)

    if alpha_mode == "attention":
        alpha_mask = mask
    elif alpha_mode == "coverage":
        alpha_mask = coverage
    else:
        raise ValueError(f"Unsupported alpha_mode: {alpha_mode}")

    overlay = (
        thumbnail * (1 - alpha_mask[..., None] * alpha)
        + heatmap_color * (alpha_mask[..., None] * alpha)
    ).astype(np.uint8)
    if save_path is not None:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(p), overlay)
    return overlay


def save_top_patches(
    wsi_path,
    h5_path,
    att_weights,
    save_dir=None,
    top_k=50,
    patch_size=256,
    coord_key="coords",
):
    if top_k <= 0:
        return []
    if save_dir is None:
        save_dir = Path("top_patches")
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    coords = load_patch_coords(h5_path, coord_key)
    att = _to_numpy_1d(att_weights)
    if len(coords) != len(att):
        raise ValueError(
            f"coords ({len(coords)}) != attention ({len(att)}); refusing truncation"
        )
    if len(coords) == 0:
        return []

    top_idx = np.argsort(att)[-min(top_k, len(att)) :][::-1]
    saved = []
    slide = openslide.OpenSlide(str(wsi_path))
    try:
        for rank, idx in enumerate(top_idx, 1):
            x, y = coords[idx]
            patch = slide.read_region(
                (int(x), int(y)), 0, (patch_size, patch_size)
            ).convert("RGB")
            out = save_dir / f"top{rank:03d}_idx{idx}_att{att[idx]:.4f}.png"
            cv2.imwrite(str(out), cv2.cvtColor(np.asarray(patch), cv2.COLOR_RGB2BGR))
            saved.append(str(out))
    finally:
        slide.close()
    return saved
