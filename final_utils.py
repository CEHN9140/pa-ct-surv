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
    split = samples["split"].astype(str).to_numpy()
    train_mask = split == "train"
    test_mask = split == "test"
    if (~(train_mask | test_mask)).any():
        raise ValueError("Dataset split values must be 'train' or 'test'")
    if not train_mask.any() or not test_mask.any():
        raise ValueError("Dataset split must contain both train and test samples")
    return np.flatnonzero(train_mask), np.flatnonzero(test_mask)


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
