import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml


def seed_everything(seed):
    """Set training RNGs; runtime PYTHONHASHSEED only affects child processes."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def build_ct_optimizer(model, lr, weight_decay, ct_backbone_lr=None):
    """Use separate learning rates for the CT backbone and risk head."""
    head_ids = {id(p) for p in model.ct.fc.parameters()}
    backbone, head = [], []
    for parameter in model.parameters():
        if parameter.requires_grad:
            (head if id(parameter) in head_ids else backbone).append(parameter)
    groups = []
    if backbone:
        groups.append({"params": backbone, "lr": lr if ct_backbone_lr is None else ct_backbone_lr})
    if head:
        groups.append({"params": head, "lr": lr})
    return torch.optim.Adam(groups, weight_decay=weight_decay)


def save_run_config(args, model_type, tag, excluded=(), extra=None):
    """Resolve output directories and save only the effective run settings."""
    root = Path(__file__).resolve().parent
    for key, directory in (("checkpoint_root", "checkpoints"), ("results_root", "results")):
        path = getattr(args, key) or str(root / directory / "pact_v6" / model_type / tag)
        setattr(args, key, path)
        Path(path).mkdir(parents=True, exist_ok=True)
    config = {key: value for key, value in vars(args).items() if key not in excluded}
    config.update(extra or {})
    with (Path(args.results_root) / "run_config.yaml").open("w") as handle:
        yaml.safe_dump(config, handle, allow_unicode=True)
    print(f"Checkpoints: {args.checkpoint_root}")
    print(f"Results: {args.results_root}")


def save_cv_results(fold_results, results_root):
    """Store fold metrics and descriptive summaries together in metrics.csv."""
    frame = pd.DataFrame(fold_results)
    root = Path(results_root)
    root.mkdir(parents=True, exist_ok=True)
    # Standalone analysis can refresh survival metrics without losing saved losses.
    if (root / "metrics.csv").is_file():
        previous = pd.read_csv(root / "metrics.csv")
        previous["fold"] = pd.to_numeric(previous["fold"], errors="coerce")
        previous = previous.dropna(subset=["fold"]).set_index("fold")
        for key in ("train_loss", "val_loss"):
            if key not in frame and key in previous:
                frame[key] = frame["fold"].map(previous[key])
    leading = [key for key in ("fold", "train_loss", "train_cindex", "val_loss", "val_cindex")
               if key in frame]
    frame = frame[leading + [key for key in frame if key not in leading]]
    # HR/CI endpoints, p-values and model-specific cutoffs remain per-fold results.
    columns = [key for key in frame if key in
               {"train_loss", "train_cindex", "val_loss", "val_cindex", "gap"}
               or (key.startswith("auc_") and key.endswith("m"))]
    summary = pd.DataFrame([{"fold": stat, **{
        key: float(getattr(frame[key], stat)(skipna=False)) for key in columns
    }} for stat in ("mean", "std")])
    output = pd.concat([frame, summary], ignore_index=True)
    temporary = root / "metrics.csv.tmp"
    output.to_csv(temporary, index=False)
    temporary.replace(root / "metrics.csv")
    print(f"\n{'=' * 50}\n5-Fold CV Summary\n{'=' * 50}")
    for key, label in (("train_cindex", "Train C-index"), ("val_cindex", "Val C-index")):
        if key in columns:
            print(f"  {label}: {summary.iloc[0][key]:.4f} ± {summary.iloc[1][key]:.4f}")
