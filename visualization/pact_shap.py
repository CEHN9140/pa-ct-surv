"""Exact two-modality Shapley analysis for trained PACT survival models.

The two players are pathology and CT. For each CV fold, the reference value
for a missing modality is the mean projected feature from that fold's training
patients. The corresponding validation patients are explained out of fold.
"""

import argparse
import os
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(_PROJECT_ROOT / ".matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from dataset import Pa_CT_Dataset
from final_utils import cv_fold_indices, locked_split_indices
from model.build import Pa_CT_Model


def fuse_risk(
    model,
    ct_feature,
    pa_feature,
    ct_risk=None,
    pa_risk=None,
):
    """Evaluate the trained PACT risk head from modality representations."""
    if model.fusion_type == "weighted":
        if ct_risk is None or pa_risk is None:
            raise ValueError("Weighted fusion requires CT and PA branch risks")
        alpha = torch.sigmoid(model.risk_weight)
        return (alpha * ct_risk + (1 - alpha) * pa_risk).reshape(-1)

    risk = model.fusion(ct_feature, pa_feature)
    if model.fusion_type == "bilinear":
        risk = model.fused_head(risk).squeeze(-1)
    return risk.reshape(-1)


def load_model(config, checkpoint, device):
    pa_model_name = config["pa_model"]
    pa_topk = None
    if pa_model_name.endswith("-topk"):
        if config.get("k") is None:
            raise ValueError("Current PACT top-k model requires a numeric k")
        pa_topk = int(config["k"])

    model = Pa_CT_Model(
        pa_model_name=pa_model_name,
        ct_model_name=config.get("ct_model", "resnet18"),
        pa_topk=pa_topk,
        fusion_type=config.get("fusion_type", "concat"),
        # The PACT checkpoint already contains the complete CT branch.
        ct_pretrained_path=None,
    )
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model


def get_sample_features(model, dataset, index, device):
    ct, pa, event, time, case_id = dataset[int(index)]
    ct = ct.unsqueeze(0).to(device, non_blocking=True)
    pa = pa.to(device, non_blocking=True)
    raw_ct_feature = model.ct_backbone.extract_features(ct)
    ct_risk = model.ct_backbone.risk_forward(raw_ct_feature)
    ct_feature = model.ct_projector(raw_ct_feature)
    pa_risk, pa_feature, _ = model.pa_branch(pa)
    pa_feature = model.pa_projector(pa_feature)
    return (
        ct_feature,
        pa_feature,
        ct_risk,
        pa_risk,
        int(event.item()),
        float(time.item()),
        str(case_id),
    )


def compute_background(model, dataset, train_indices, device):
    ct_sum = None
    pa_sum = None
    ct_risk_sum = None
    pa_risk_sum = None
    with torch.inference_mode():
        for position, index in enumerate(train_indices, start=1):
            ct_feature, pa_feature, ct_risk, pa_risk, _, _, _ = get_sample_features(
                model, dataset, index, device
            )
            ct_sum = ct_feature.clone() if ct_sum is None else ct_sum + ct_feature
            pa_sum = pa_feature.clone() if pa_sum is None else pa_sum + pa_feature
            ct_risk_sum = (
                ct_risk.clone() if ct_risk_sum is None else ct_risk_sum + ct_risk
            )
            pa_risk_sum = (
                pa_risk.clone() if pa_risk_sum is None else pa_risk_sum + pa_risk
            )
            if position % 100 == 0:
                print(f"    background features: {position}/{len(train_indices)}")
    return (
        ct_sum / len(train_indices),
        pa_sum / len(train_indices),
        ct_risk_sum / len(train_indices),
        pa_risk_sum / len(train_indices),
    )


def explain_fold(
    model,
    dataset,
    train_indices,
    val_indices,
    fold,
    device,
):
    background_ct, background_pa, background_ct_risk, background_pa_risk = compute_background(
        model, dataset, train_indices, device
    )
    baseline_risk = float(
        fuse_risk(
            model,
            background_ct,
            background_pa,
            background_ct_risk,
            background_pa_risk,
        ).item()
    )

    rows = []
    with torch.inference_mode():
        for position, index in enumerate(val_indices, start=1):
            (
                ct_feature,
                pa_feature,
                ct_risk,
                pa_risk,
                event,
                time,
                case_id,
            ) = get_sample_features(
                model, dataset, index, device
            )
            full_risk = float(
                fuse_risk(model, ct_feature, pa_feature, ct_risk, pa_risk).item()
            )
            pa_only_risk = float(
                fuse_risk(
                    model,
                    background_ct,
                    pa_feature,
                    background_ct_risk,
                    pa_risk,
                ).item()
            )
            ct_only_risk = float(
                fuse_risk(
                    model,
                    ct_feature,
                    background_pa,
                    ct_risk,
                    background_pa_risk,
                ).item()
            )
            pa_shap = 0.5 * (
                (pa_only_risk - baseline_risk) + (full_risk - ct_only_risk)
            )
            ct_shap = 0.5 * (
                (ct_only_risk - baseline_risk) + (full_risk - pa_only_risk)
            )
            interaction = (
                full_risk - pa_only_risk - ct_only_risk + baseline_risk
            )
            if "|" in case_id:
                slide_id, ct_id = case_id.split("|", 1)
            else:
                slide_id, ct_id = case_id, ""
            rows.append(
                {
                    "fold": fold,
                    "dataset_index": int(index),
                    "case_id": case_id,
                    "pa_id": slide_id,
                    "ct_id": ct_id,
                    "time": time,
                    "event": event,
                    "baseline_risk": baseline_risk,
                    "pa_only_risk": pa_only_risk,
                    "ct_only_risk": ct_only_risk,
                    "full_risk": full_risk,
                    "pa_shap": pa_shap,
                    "ct_shap": ct_shap,
                    "interaction": interaction,
                }
            )
            if position % 25 == 0:
                print(f"    validation SHAP: {position}/{len(val_indices)}")
    return rows


def save_summary(df, output_dir):
    summary = pd.DataFrame(
        [
            {
                "n_patients": len(df),
                "n_events": int(df["event"].sum()),
                "mean_pa_shap": df["pa_shap"].mean(),
                "mean_abs_pa_shap": df["pa_shap"].abs().mean(),
                "pa_positive_fraction": (df["pa_shap"] > 0).mean(),
                "mean_ct_shap": df["ct_shap"].mean(),
                "mean_abs_ct_shap": df["ct_shap"].abs().mean(),
                "ct_positive_fraction": (df["ct_shap"] > 0).mean(),
                "mean_interaction": df["interaction"].mean(),
                "mean_abs_interaction": df["interaction"].abs().mean(),
            }
        ]
    )
    summary.to_csv(output_dir / "oof_modality_shap_summary.csv", index=False)


def save_plots(df, output_dir):
    values = [df["pa_shap"].to_numpy(), df["ct_shap"].to_numpy()]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    parts = ax.violinplot(values, positions=[1, 2], showmedians=True)
    for body, color in zip(parts["bodies"], ["#C44E52", "#4C72B0"]):
        body.set_facecolor(color)
        body.set_alpha(0.55)
    rng = np.random.default_rng(42)
    for position, (column, color) in enumerate(
        [("pa_shap", "#C44E52"), ("ct_shap", "#4C72B0")],
        start=1,
    ):
        jitter = rng.normal(position, 0.035, size=len(df))
        ax.scatter(
            jitter,
            df[column],
            s=10,
            alpha=0.45,
            color=color,
            edgecolors="none",
        )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks([1, 2], ["Pathology", "CT"])
    ax.set_ylabel("SHAP contribution to Cox log-risk")
    ax.set_title("Out-of-fold modality contributions")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "oof_modality_shap_distribution.png", dpi=300)
    plt.close(fig)

    selected = df.assign(
        total_abs=df["pa_shap"].abs() + df["ct_shap"].abs()
    ).nlargest(min(30, len(df)), "total_abs")
    selected = selected.sort_values("full_risk")
    x = np.arange(len(selected))
    fig, ax = plt.subplots(figsize=(max(9, len(selected) * 0.34), 5))
    ax.bar(x, selected["pa_shap"], label="Pathology", color="#C44E52")
    ax.bar(
        x,
        selected["ct_shap"],
        bottom=selected["pa_shap"],
        label="CT",
        color="#4C72B0",
    )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x, selected["ct_id"].astype(str), rotation=75, ha="right")
    ax.set_ylabel("SHAP contribution to Cox log-risk")
    ax.set_title("Patients with the largest absolute modality contributions")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "oof_patient_shap_contributions.png", dpi=300)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Exact PA/CT modality Shapley analysis on PACT CV validation folds"
    )
    parser.add_argument("--config", required=True, help="PACT run_config.yaml")
    parser.add_argument(
        "--checkpoint_root",
        required=True,
        help="Directory containing fold_0 ... fold_4",
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config_path = Path(args.config)
    checkpoint_root = Path(args.checkpoint_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with config_path.open("r") as handle:
        config = yaml.safe_load(handle)
    data_dir = config["data_dir"]
    roi_size = int(config["ct_roi_size"])
    device = torch.device(args.device)

    dataset = Pa_CT_Dataset(data_dir, roi_size=roi_size, augment=False)
    fold_splits = [cv_fold_indices(dataset.samples, fold) for fold in range(5)]
    all_rows = []

    print(
        f"PACT modality SHAP | patients={len(dataset)} | "
        f"ROI={roi_size} | device={device}"
    )
    for fold, (train_indices, val_indices) in enumerate(fold_splits):
        checkpoint = checkpoint_root / f"fold_{fold}" / "best_model.pth"
        if not checkpoint.exists():
            raise FileNotFoundError(f"Missing fold checkpoint: {checkpoint}")
        print(
            f"\nFold {fold}: train={len(train_indices)}, "
            f"val={len(val_indices)}"
        )
        model = load_model(config, checkpoint, device)
        all_rows.extend(
            explain_fold(
                model,
                dataset,
                train_indices,
                val_indices,
                fold,
                device,
            )
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    df = pd.DataFrame(all_rows).sort_values("dataset_index").reset_index(drop=True)
    development_indices, _ = locked_split_indices(dataset.samples)
    if len(df) != len(development_indices):
        raise RuntimeError(
            f"Expected {len(development_indices)} OOF rows, found {len(df)}"
        )
    if df["dataset_index"].duplicated().any():
        raise RuntimeError("OOF output contains duplicate patients")
    if set(df["dataset_index"]) != set(development_indices.tolist()):
        raise RuntimeError("OOF output does not exactly cover the development set")

    df.to_csv(output_dir / "oof_modality_shap.csv", index=False)
    save_summary(df, output_dir)
    save_plots(df, output_dir)
    with (output_dir / "analysis_config.yaml").open("w") as handle:
        yaml.safe_dump(
            {
                "source_config": str(config_path.resolve()),
                "checkpoint_root": str(checkpoint_root.resolve()),
                "roi_size": roi_size,
                "background": "fold-training mean projected feature",
                "explained_split": "out-of-fold validation",
                "risk_output": "Cox log-risk",
            },
            handle,
            sort_keys=True,
        )
    print(f"\n[OK] Saved modality SHAP analysis to {output_dir}")


if __name__ == "__main__":
    main()
