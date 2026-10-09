"""Zero-retraining attention-temperature intervention for pathology ABMIL.

This diagnostic loads frozen per-fold ABMIL checkpoints and changes only the
attention softmax temperature at inference time. The locked test split is
never loaded; metrics are computed on each fold's train and validation rows.
"""

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from sksurv.metrics import concordance_index_censored
from torch.utils.data import DataLoader, Subset

# Permit both ``python -m diagnostics.attention_temperature`` and direct
# execution via ``python diagnostics/attention_temperature.py``.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset import Path_Dataset
from final_utils import cv_fold_indices, locked_split_indices
from model.build import Pa_Model
from path_train import load_pca_transform


DEFAULT_TEMPERATURES = (0.5, 1.0, 1.5, 2.0, 4.0)


def attention_statistics(weights):
    weights = weights.detach().float().reshape(-1)
    weights = weights / weights.sum().clamp_min(torch.finfo(weights.dtype).eps)
    n = int(weights.numel())
    entropy = -(weights * weights.clamp_min(1e-12).log()).sum()
    entropy_norm = entropy / np.log(n) if n > 1 else 0.0
    effective = 1.0 / weights.square().sum().item()
    return {
        "num_patches": n,
        "max_attention": float(weights.max().item()),
        "attention_entropy": float(entropy.item()),
        "attention_entropy_norm": float(entropy_norm),
        "effective_patch_num": float(effective),
        "effective_patch_ratio": float(effective / n),
    }


@torch.no_grad()
def predict_with_temperature(model, features, temperature):
    """Return risk and attention after changing only the ABMIL temperature."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if features.dim() == 2:
        features = features.unsqueeze(0)
    mil = model.mil
    hidden = mil.projector(features)
    logits = mil.attention(hidden)
    weights = F.softmax(logits.transpose(1, 2) / temperature, dim=2)
    pooled = torch.bmm(weights, hidden).reshape(hidden.size(0), -1)
    risk = mil.risk_head(pooled).squeeze(-1)
    return risk, weights.transpose(1, 2)


def _load_state_dict(model, checkpoint):
    try:
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state)


@torch.no_grad()
def evaluate_split(model, loader, device, temperatures, split):
    """Evaluate all temperatures while reading each WSI only once."""
    risks = {temperature: [] for temperature in temperatures}
    stats = {temperature: [] for temperature in temperatures}
    times, events, case_ids = [], [], []
    mil = model.mil
    for features, event, time, case_id in loader:
        features = features.to(device, non_blocking=True)
        if features.dim() == 2:
            features = features.unsqueeze(0)
        hidden = mil.projector(features)
        logits = mil.attention(hidden)
        times.extend(time.numpy().reshape(-1).tolist())
        events.extend(event.numpy().reshape(-1).astype(int).tolist())
        case_ids.extend(case_id)
        batch_case_ids = list(case_id)
        for temperature in temperatures:
            weights = F.softmax(logits.transpose(1, 2) / temperature, dim=2)
            pooled = torch.bmm(weights, hidden).reshape(hidden.size(0), -1)
            risk = mil.risk_head(pooled).squeeze(-1)
            risks[temperature].extend(risk.cpu().numpy().reshape(-1).tolist())
            weights = weights.transpose(1, 2)
            for batch_index in range(features.size(0)):
                for branch in range(weights.size(2)):
                    row = attention_statistics(weights[batch_index, :, branch])
                    row.update({"case_id": batch_case_ids[batch_index], "branch": branch})
                    stats[temperature].append(row)

    evaluated = {}
    for temperature in temperatures:
        prediction = pd.DataFrame(
            {
                "case_id": case_ids,
                "dfs.month": times,
                "dfs.status": events,
                "risk_score": risks[temperature],
            }
        )
        prediction["split"] = split
        attention = pd.DataFrame(stats[temperature])
        cindex = float(
            concordance_index_censored(
                prediction["dfs.status"].to_numpy(bool),
                prediction["dfs.month"].to_numpy(float),
                prediction["risk_score"].to_numpy(float),
            )[0]
        )
        evaluated[temperature] = (cindex, prediction, attention)
    return evaluated


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", default="/home/gly001/cqj/pa_ct_surv/data/seed_42")
    parser.add_argument("--roi_size", type=int, default=64)
    parser.add_argument(
        "--pca_dim",
        type=int,
        default=None,
        help="PCA dimension used by the checkpoint; fit separately per training fold.",
    )
    parser.add_argument("--abmil_hidden_dim", type=int, default=512)
    parser.add_argument("--abmil_attention_dim", type=int, default=128)
    parser.add_argument(
        "--checkpoint_root",
        default=(
            "/home/gly001/cqj/pa_ct_surv/checkpoints/pact_v5/pathology/"
            "roi64_abmil_dropout0.0_epochs30_coxbs64_lr1e-4_wd5e-4_seed42"
        ),
    )
    parser.add_argument(
        "--checkpoint_name",
        choices=("best_model.pth", "last_model.pth"),
        default="best_model.pth",
        help="Checkpoint file to intervene on.",
    )
    parser.add_argument(
        "--entropy_match_checkpoint_root",
        default=None,
        help="Optional Best-checkpoint root used to choose T_match from the train-set entropy.",
    )
    parser.add_argument(
        "--results_root",
        default=(
            "/home/gly001/cqj/pa_ct_surv/results/pact_v5/diagnostics/pathology/"
            "attention_temperature"
        ),
    )
    parser.add_argument("--temperatures", nargs="+", type=float, default=DEFAULT_TEMPERATURES)
    parser.add_argument("--fold", type=int, default=None, help="Run one CV fold; default runs all five folds.")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    temperatures = tuple(args.temperatures)
    if any(t <= 0 for t in temperatures):
        raise ValueError("all temperatures must be positive")
    if args.pca_dim is not None and not 0 < args.pca_dim < 1024:
        raise ValueError("pca_dim must be between 1 and 1023")

    dataset = Path_Dataset(args.data_dir, roi_size=args.roi_size)
    train_indices, _ = locked_split_indices(dataset.samples)
    fold_splits = [cv_fold_indices(dataset.samples, fold) for fold in range(5)]
    checkpoint_root = Path(args.checkpoint_root)
    missing = [
        fold
        for fold in range(5)
        if not (checkpoint_root / f"fold_{fold}" / args.checkpoint_name).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"Missing {args.checkpoint_name} for fold(s): "
            + ", ".join(map(str, missing))
            + f" under {checkpoint_root}"
        )
    match_root = Path(args.entropy_match_checkpoint_root) if args.entropy_match_checkpoint_root else None
    if match_root is not None:
        missing_match = [
            fold
            for fold in range(5)
            if not (match_root / f"fold_{fold}" / "best_model.pth").is_file()
        ]
        if missing_match:
            raise FileNotFoundError(
                "Missing best_model.pth for entropy matching fold(s): "
                + ", ".join(map(str, missing_match))
                + f" under {match_root}"
            )

    results_root = Path(args.results_root)
    results_root.mkdir(parents=True, exist_ok=True)
    metric_rows, summary_rows = [], []
    folds_to_run = range(5) if args.fold is None else [args.fold]
    if any(fold < 0 or fold >= 5 for fold in folds_to_run):
        raise ValueError("--fold must be in [0, 4]")
    for fold in folds_to_run:
        train_idx, val_idx = fold_splits[fold]
        if args.pca_dim is not None:
            late_pca_path = checkpoint_root / f"fold_{fold}" / "pca_transform.pt"
            if not late_pca_path.is_file():
                raise FileNotFoundError(f"PCA transform not found: {late_pca_path}")
            late_pca_transform = load_pca_transform(late_pca_path)
            dataset.set_pca_transform(late_pca_transform)
        else:
            late_pca_transform = None
            dataset.set_pca_transform(None)
        model = Pa_Model(
            model_name="abmil",
            feature_dim=args.pca_dim if args.pca_dim is not None else 1024,
            abmil_hidden_dim=args.abmil_hidden_dim,
            abmil_attention_dim=args.abmil_attention_dim,
        ).to(device)
        _load_state_dict(model, checkpoint_root / f"fold_{fold}" / args.checkpoint_name)
        model.eval()
        train_loader = DataLoader(Subset(dataset, train_idx), batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True)
        val_loader = DataLoader(Subset(dataset, val_idx), batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True)

        baseline_checked = False
        baseline_risks = {}
        train_results = evaluate_split(model, train_loader, device, temperatures, "train")
        val_results = evaluate_split(model, val_loader, device, temperatures, "val")
        # Verify the intervention is algebraically identical to ABMIL at T=1.
        sample = next(iter(val_loader))[0].to(device)
        native = model(sample)[0]
        intervention = predict_with_temperature(model, sample, 1.0)[0]
        if not torch.allclose(native, intervention, atol=1e-6, rtol=1e-5):
            raise RuntimeError(f"T=1 consistency check failed in fold {fold}")
        baseline_checked = True
        matched_temperature = None
        if match_root is not None and args.checkpoint_name == "last_model.pth":
            late_entropy = {
                temperature: float(
                    train_results[temperature][2]["attention_entropy_norm"].mean()
                )
                for temperature in temperatures
            }
            _load_state_dict(model, match_root / f"fold_{fold}" / "best_model.pth")
            if args.pca_dim is not None:
                best_pca_path = match_root / f"fold_{fold}" / "pca_transform.pt"
                if not best_pca_path.is_file():
                    raise FileNotFoundError(f"PCA transform not found: {best_pca_path}")
                dataset.set_pca_transform(load_pca_transform(best_pca_path))
            best_train = evaluate_split(model, train_loader, device, (1.0,), "train")[1.0]
            best_entropy = float(best_train[2]["attention_entropy_norm"].mean())
            matched_temperature = min(
                late_entropy,
                key=lambda temperature: abs(late_entropy[temperature] - best_entropy),
            )
            _load_state_dict(model, checkpoint_root / f"fold_{fold}" / args.checkpoint_name)
            dataset.set_pca_transform(late_pca_transform)
            match_c, match_pred, match_attention = train_results[matched_temperature]
            match_val_c, match_val_pred, match_val_attention = val_results[matched_temperature]
            for split, pred, attention, cindex in (
                ("train", match_pred, match_attention, match_c),
                ("val", match_val_pred, match_val_attention, match_val_c),
            ):
                pred_dir = results_root / f"fold_{fold}" / "T_match"
                pred_dir.mkdir(parents=True, exist_ok=True)
                pred.to_csv(pred_dir / f"{split}_predictions.csv", index=False)
                attention.to_csv(pred_dir / f"{split}_attention_stats.csv", index=False)
                metric_rows.append({
                    "fold": fold,
                    "temperature": matched_temperature,
                    "temperature_label": "T_match",
                    "split": split,
                    "cindex": cindex,
                    "risk_spearman_vs_T1": np.nan,
                    "attention_entropy_norm_mean": float(attention["attention_entropy_norm"].mean()),
                    "effective_patch_ratio_mean": float(attention["effective_patch_ratio"].mean()),
                    "max_attention_mean": float(attention["max_attention"].mean()),
                    "best_train_entropy_target": best_entropy,
                })
            summary_rows.append({
                "fold": fold,
                "temperature": matched_temperature,
                "temperature_label": "T_match",
                "train_cindex": match_c,
                "val_cindex": match_val_c,
                "gap": match_c - match_val_c,
                "t1_consistency_checked": baseline_checked,
                "best_train_entropy_target": best_entropy,
            })
        for temperature in temperatures:
            train_c, train_pred, train_attention = train_results[temperature]
            val_c, val_pred, val_attention = val_results[temperature]
            for split, pred, attention, cindex in (
                ("train", train_pred, train_attention, train_c),
                ("val", val_pred, val_attention, val_c),
            ):
                pred_dir = results_root / f"fold_{fold}" / f"T{temperature:g}"
                pred_dir.mkdir(parents=True, exist_ok=True)
                pred.to_csv(pred_dir / f"{split}_predictions.csv", index=False)
                attention.to_csv(pred_dir / f"{split}_attention_stats.csv", index=False)
                if temperature == 1.0:
                    baseline_risks[split] = pred.set_index("case_id")["risk_score"]
                risk_corr = np.nan
                if split in baseline_risks:
                    joined = pd.concat([baseline_risks[split], pred.set_index("case_id")["risk_score"]], axis=1).dropna()
                    if len(joined) > 1:
                        risk_corr = float(spearmanr(joined.iloc[:, 0], joined.iloc[:, 1]).statistic)
                metric_rows.append({
                    "fold": fold, "temperature": temperature, "split": split,
                    "temperature_label": f"T={temperature:g}",
                    "cindex": cindex, "risk_spearman_vs_T1": risk_corr,
                    "attention_entropy_norm_mean": float(attention["attention_entropy_norm"].mean()),
                    "effective_patch_ratio_mean": float(attention["effective_patch_ratio"].mean()),
                    "max_attention_mean": float(attention["max_attention"].mean()),
                })
            summary_rows.append({"fold": fold, "temperature": temperature, "temperature_label": f"T={temperature:g}", "train_cindex": train_c, "val_cindex": val_c, "gap": train_c - val_c, "t1_consistency_checked": baseline_checked})

    metrics = pd.DataFrame(metric_rows)
    summary = pd.DataFrame(summary_rows)
    metrics.to_csv(results_root / "fold_split_metrics.csv", index=False)
    summary.to_csv(results_root / "fold_summary.csv", index=False)
    aggregate = summary.groupby("temperature", as_index=False).agg(
        train_cindex_mean=("train_cindex", "mean"), val_cindex_mean=("val_cindex", "mean"),
        gap_mean=("gap", "mean"), val_cindex_std=("val_cindex", "std"),
    )
    attention_aggregate = metrics[metrics["split"] == "val"].groupby("temperature", as_index=False).agg(
        attention_entropy_norm_mean=("attention_entropy_norm_mean", "mean"),
        effective_patch_ratio_mean=("effective_patch_ratio_mean", "mean"),
        max_attention_mean=("max_attention_mean", "mean"),
    )
    aggregate.merge(attention_aggregate, on="temperature").to_csv(results_root / "temperature_summary.csv", index=False)
    print(aggregate.merge(attention_aggregate, on="temperature").to_string(index=False))


if __name__ == "__main__":
    main()
