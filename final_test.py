import argparse
from pathlib import Path

import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader, Subset

from cox_utils import evaluate_survival
from dataset import CT_Dataset, Pa_CT_Dataset, Path_Dataset
from final_utils import bootstrap_cindex, locked_split_indices, seed_everything
from model.build import CT_Model, Pa_CT_Model, Pa_Model


def build_model(model_type, config):
    if model_type == "ct":
        return CT_Model(
            model_name=config["model_name"],
            pretrained_path=None,
            dropout=float(config["dropout"]),
            freeze_backbone=False,
            model_depth=int(config.get("model_depth", 18)),
        )
    if model_type == "path":
        return Pa_Model(
            num_classes=2,
            feature_dim=1024,
            model_name=config["model_name"],
            k=int(config["k"]),
            implementation=config["implementation"],
        )
    if model_type == "pact":
        return Pa_CT_Model(
            pa_model_name=config["pa_model"],
            ct_pretrained_path=None,
            pa_topk=int(config["k"]),
            fusion_type=config["fusion_type"],
            ct_dropout=float(config["ct_dropout"]),
            pa_implementation=config.get("implementation", "legacy"),
            ct_model_depth=int(config.get("ct_model_depth", 18)),
        )
    if model_type == "student":
        return CT_Model(
            model_name="resnet",
            pretrained_path=None,
            dropout=float(config["student_ct_dropout"]),
            model_depth=int(config.get("student_model_depth", 18)),
        )
    raise ValueError(f"Unknown model type: {model_type}")


def build_dataset(model_type, data_dir, roi_size):
    if model_type == "path":
        return Path_Dataset(data_dir, roi_size=roi_size)
    if model_type == "pact":
        return Pa_CT_Dataset(data_dir, roi_size=roi_size, augment=False)
    return CT_Dataset(data_dir, roi_size=roi_size, augment=False)


def select_split_dataset(dataset, split_name="test"):
    train_indices, test_indices = locked_split_indices(dataset.samples)
    indices = train_indices if split_name == "train" else test_indices
    if split_name not in {"train", "test"}:
        raise ValueError("split_name must be 'train' or 'test'")
    return Subset(dataset, indices)


def predict_risk(model_type):
    def predict(model, batch, device):
        if model_type in {"ct", "student"}:
            ct, event, time, case_id = batch
            risk = model(ct.to(device, non_blocking=True))
        elif model_type == "path":
            features, event, time, case_id = batch
            output = model(features.to(device, non_blocking=True))
            risk = output[0] if isinstance(output, tuple) else output
        else:
            ct, features, event, time, case_id = batch
            risk = model(
                ct.to(device, non_blocking=True),
                features.to(device, non_blocking=True),
            )[0]
        return risk, event, time, case_id

    return predict


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate one final survival model on the locked test set."
    )
    parser.add_argument(
        "--model_type", required=True, choices=["ct", "path", "pact", "student"]
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data_dir", default="/home/gly001/cqj/pa_ct_surv/data/seed_42")
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--bootstrap_samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)
    checkpoint = Path(args.checkpoint)
    config_path = Path(args.config)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if not config_path.is_file():
        raise FileNotFoundError(config_path)

    with open(config_path) as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError("Final config must contain a YAML mapping")
    if config.get("model_type") != args.model_type:
        raise ValueError(
            f"Config model_type={config.get('model_type')} does not match "
            f"--model_type={args.model_type}"
        )

    roi_size = int(config["roi_size"])
    dataset = build_dataset(args.model_type, args.data_dir, roi_size)
    test_subset = select_split_dataset(dataset, split_name="test")
    default_batch_size = 1 if args.model_type in {"path", "pact"} else int(
        config.get("batch_size", 16)
    )
    batch_size = args.batch_size or default_batch_size
    loader = DataLoader(
        test_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=int(config.get("num_workers", 8)),
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args.model_type, config).to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state)
    cindex, predictions = evaluate_survival(
        model, loader, predict_risk(args.model_type), device
    )

    point, ci_lower, ci_upper, valid_bootstrap = bootstrap_cindex(
        predictions["dfs.status"].to_numpy(),
        predictions["dfs.month"].to_numpy(),
        predictions["risk_score"].to_numpy(),
        n_bootstrap=args.bootstrap_samples,
        seed=args.seed,
    )
    if abs(point - cindex) > 1e-12:
        raise RuntimeError("Inconsistent C-index calculation")

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(results_dir / "test_predictions.csv", index=False)
    metrics = {
        "model_type": args.model_type,
        "cindex": cindex,
        "cindex_ci95_lower": ci_lower,
        "cindex_ci95_upper": ci_upper,
        "n_test": len(predictions),
        "n_events": int(predictions["dfs.status"].sum()),
        "bootstrap_samples": args.bootstrap_samples,
        "valid_bootstrap_samples": valid_bootstrap,
        "checkpoint": str(checkpoint.resolve()),
        "config": str(config_path.resolve()),
    }
    pd.DataFrame([metrics]).to_csv(results_dir / "test_metrics.csv", index=False)
    print(
        f"Test C-index: {cindex:.4f} "
        f"(95% CI {ci_lower:.4f}-{ci_upper:.4f}) | "
        f"n={len(predictions)}, events={metrics['n_events']}"
    )
    print(f"Results: {results_dir}")


if __name__ == "__main__":
    main()
