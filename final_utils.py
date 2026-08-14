import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sksurv.exceptions import NoComparablePairException
from sksurv.metrics import concordance_index_censored


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def locked_split_indices(samples):
    if "split" not in samples.columns:
        raise ValueError("Dataset CSV missing split column; run preprocess_data.py")
    raw_split = samples["split"]
    legacy_values = set(raw_split.astype(str))
    if legacy_values.issubset({"train", "test"}):
        train_mask = raw_split.astype(str).to_numpy() == "train"
        test_mask = raw_split.astype(str).to_numpy() == "test"
        if not train_mask.any() or not test_mask.any():
            raise ValueError("Dataset split must contain both train and test samples")
        return np.flatnonzero(train_mask), np.flatnonzero(test_mask)

    split = pd.to_numeric(raw_split, errors="coerce").to_numpy()
    valid = np.isfinite(split) & np.isin(split, [-1, 0, 1, 2, 3, 4])
    if not valid.all():
        raise ValueError("Dataset split values must be integers -1 or 0..4")
    train_mask = split >= 0
    test_mask = split == -1
    if not train_mask.any() or not test_mask.any():
        raise ValueError("Dataset split must contain both train and test samples")
    return np.flatnonzero(train_mask), np.flatnonzero(test_mask)


def cv_fold_indices(samples, fold, n_splits=5):
    """Return (train, validation) row indices for a preassigned CV fold."""
    if not isinstance(fold, (int, np.integer)) or not 0 <= int(fold) < n_splits:
        raise ValueError(f"fold must be an integer in [0, {n_splits - 1}]")
    train_indices, _ = locked_split_indices(samples)
    split = pd.to_numeric(samples["split"], errors="coerce").to_numpy()
    validation_mask = split == int(fold)
    fold_train_mask = (split >= 0) & (split != int(fold))
    validation_indices = np.flatnonzero(validation_mask)
    fold_train_indices = np.flatnonzero(fold_train_mask)
    if not len(validation_indices) or not len(fold_train_indices):
        raise ValueError(f"CV fold {fold} must have non-empty train and validation sets")
    if len(fold_train_indices) + len(validation_indices) != len(train_indices):
        raise ValueError("CV folds must cover exactly the locked train set")
    return fold_train_indices, validation_indices


def save_final_artifacts(
    model,
    args,
    checkpoint_root,
    results_root,
    history,
    model_type,
):
    checkpoint_root = Path(checkpoint_root)
    results_root = Path(results_root)
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    results_root.mkdir(parents=True, exist_ok=True)

    checkpoint_path = checkpoint_root / "final_model.pth"
    config_path = checkpoint_root / "final_config.yaml"
    history_path = results_root / "final_train_history.csv"

    torch.save(model.state_dict(), checkpoint_path)
    config = dict(vars(args))
    config["model_type"] = model_type
    config["checkpoint"] = str(checkpoint_path.resolve())
    with open(config_path, "w") as f:
        yaml.safe_dump(config, f, sort_keys=True, allow_unicode=True)
    pd.DataFrame(history).to_csv(history_path, index=False)
    return checkpoint_path, config_path, history_path


def bootstrap_cindex(event, time, risk, n_bootstrap=1000, seed=42):
    event = np.asarray(event, dtype=bool)
    time = np.asarray(time, dtype=float)
    risk = np.asarray(risk, dtype=float)
    if not (len(event) == len(time) == len(risk)) or len(event) == 0:
        raise ValueError("event, time, and risk must have the same non-zero length")
    if n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be positive")

    point = float(concordance_index_censored(event, time, risk)[0])
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(n_bootstrap):
        index = rng.integers(0, len(event), size=len(event))
        try:
            value = concordance_index_censored(
                event[index], time[index], risk[index]
            )[0]
        except (NoComparablePairException, ValueError):
            continue
        if np.isfinite(value):
            estimates.append(float(value))

    if not estimates:
        raise ValueError("No valid bootstrap resamples for C-index")
    lower, upper = np.percentile(estimates, [2.5, 97.5])
    return point, float(lower), float(upper), len(estimates)
