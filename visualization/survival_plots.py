"""Analyze saved five-fold validation predictions without retraining models."""

import argparse
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.exceptions import ConvergenceWarning
from lifelines.plotting import add_at_risk_counts
from lifelines.statistics import logrank_test
from sksurv.metrics import concordance_index_censored, cumulative_dynamic_auc
from sksurv.util import Surv


def read_predictions(path):
    frame = pd.read_csv(path, dtype={"case_id": str})
    columns = ["case_id", "dfs.month", "dfs.status", "risk_score"]
    if not set(columns).issubset(frame.columns) or frame.empty:
        raise ValueError(f"Missing prediction columns or empty table: {path}")
    if frame[columns].isna().any().any() or frame.case_id.duplicated().any():
        raise ValueError(f"Missing values or duplicate patients: {path}")
    for column in columns[1:]:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    if not np.isfinite(frame[columns[1:]].to_numpy()).all():
        raise ValueError(f"Nonfinite predictions: {path}")
    if (frame["dfs.month"] < 0).any() or not frame["dfs.status"].isin([0, 1]).all():
        raise ValueError(f"Invalid follow-up time or event status: {path}")
    return frame


def experiment_runs(template, seeds):
    """A concrete result directory is processed once, independent of --seeds."""
    if "{seed}" in template and not seeds:
        raise ValueError("A {seed} path requires --seeds, e.g. --seeds 42 123 2024")
    runs = []
    for seed in (seeds if "{seed}" in template else [None]):
        root = Path(template.format(seed=seed)).resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"Results directory not found: {root}")
        config_path = root / "run_config.yaml"
        config = (yaml.safe_load(config_path.read_text()) or {}) if config_path.is_file() else {}
        configured_seed = config.get("seed")
        if seed is not None and configured_seed is not None and int(configured_seed) != seed:
            raise ValueError(f"Requested seed {seed} disagrees with {config_path}")
        runs.append((seed if seed is not None else configured_seed, root))
    if len({root for _, root in runs}) != len(runs):
        raise ValueError("Duplicate result directories requested")
    return runs


def fold_metrics(train, val, eval_times):
    cutoff = float(train.risk_score.median())
    val = val.copy()
    val["risk_group_binary"] = (val.risk_score >= cutoff).astype(int)
    high, low = val[val.risk_group_binary == 1], val[val.risk_group_binary == 0]
    metrics = {"cutoff_from_train_median": cutoff, "n_val": len(val),
               "n_event_val": int(val["dfs.status"].sum()),
               "n_high_risk": len(high), "n_low_risk": len(low)}
    for split, frame in (("train", train), ("val", val)):
        metrics[f"{split}_cindex"] = float(concordance_index_censored(
            frame["dfs.status"].astype(bool), frame["dfs.month"], frame.risk_score
        )[0])
    metrics["gap"] = metrics["train_cindex"] - metrics["val_cindex"]
    y_train = Surv.from_arrays(train["dfs.status"].astype(bool), train["dfs.month"])
    y_val = Surv.from_arrays(val["dfs.status"].astype(bool), val["dfs.month"])
    for time in eval_times:
        key = f"auc_{time:g}m"
        metrics[key] = np.nan
        metrics[f"{key}_status"] = "ok"
        has_case = ((val["dfs.month"] <= time) & (val["dfs.status"] == 1)).any()
        has_control = (val["dfs.month"] > time).any()
        if not has_case or not has_control:
            metrics[f"{key}_status"] = "no cases or controls at this time"
            continue
        try:
            # Evaluate separately: failure at one horizon must not erase another.
            auc, _ = cumulative_dynamic_auc(y_train, y_val, val.risk_score, [time])
            if not np.isfinite(auc[0]):
                raise ValueError("AUC is not finite")
            metrics[key] = float(auc[0])
        except ValueError as exc:
            metrics[f"{key}_status"] = str(exc)
    metrics.update(logrank_p=np.nan, logrank_stat=np.nan, HR_high_vs_low=np.nan,
                   HR_95CI_lower=np.nan, HR_95CI_upper=np.nan, cox_p=np.nan,
                   hr_status="empty risk group")
    if len(high) and len(low):
        test = logrank_test(high["dfs.month"], low["dfs.month"],
                            high["dfs.status"], low["dfs.status"])
        metrics.update(logrank_p=float(test.p_value), logrank_stat=float(test.test_statistic))
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", ConvergenceWarning)
                fitter = CoxPHFitter().fit(val[["dfs.month", "dfs.status", "risk_group_binary"]],
                                           duration_col="dfs.month", event_col="dfs.status")
            row = fitter.summary.loc["risk_group_binary"]
            values = np.exp(row[["coef", "coef lower 95%", "coef upper 95%"]].to_numpy())
            if not np.isfinite(values).all() or (values <= 0).any():
                raise ValueError("HR or its confidence interval is not finite and positive")
            metrics.update(HR_high_vs_low=float(values[0]), HR_95CI_lower=float(values[1]),
                           HR_95CI_upper=float(values[2]), cox_p=float(row["p"]), hr_status="ok")
        except Exception as exc:
            # A failed HR fit must not discard valid KM/AUC/C-index results.
            metrics["hr_status"] = f"{type(exc).__name__}: {exc}"
    return val, metrics


def plot_km(val, metrics, path):
    fig, ax = plt.subplots(figsize=(7, 7))
    fitters = []
    for group, name, color in ((1, "High risk", "crimson"), (0, "Low risk", "steelblue")):
        frame = val[val.risk_group_binary == group]
        if frame.empty:
            continue
        fitter = KaplanMeierFitter().fit(frame["dfs.month"], frame["dfs.status"],
                                         label=f"{name} (n={len(frame)})")
        fitter.plot_survival_function(ax=ax, ci_show=True, show_censors=True, color=color)
        fitters.append(fitter)
    ax.set(xlabel="DFS time (months)", ylabel="Disease-free survival probability", ylim=(0, 1.02),
           title=f"log-rank p = {metrics['logrank_p']:.3g}")
    ax.grid(alpha=0.2)
    if fitters:
        add_at_risk_counts(*fitters, ax=ax, rows_to_show=["At risk"])
    fig.subplots_adjust(bottom=0.3, top=0.86)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def analyze_fold(train, val, destination, eval_times=(36., 60.)):
    """Save and log one validation fold using the v5 metric block format."""
    grouped, metrics = fold_metrics(train, val, eval_times)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    grouped.to_csv(destination / "val_predictions.csv", index=False)
    plot_km(grouped, metrics, destination / "km_high_vs_low.png")
    logged = {"cutoff_from_train_median": metrics["cutoff_from_train_median"],
              "n_test": metrics["n_val"], "n_high_risk": metrics["n_high_risk"],
              "n_low_risk": metrics["n_low_risk"], "n_event_test": metrics["n_event_val"],
              "logrank_stat": metrics["logrank_stat"], "logrank_p": metrics["logrank_p"]}
    logged.update({f"auc_{time:g}m": metrics[f"auc_{time:g}m"] for time in eval_times})
    logged.update({key: metrics[key] for key in
                   ("HR_high_vs_low", "HR_95CI_lower", "HR_95CI_upper", "cox_p")})
    print("\nExtra survival metrics:")
    for key, value in logged.items():
        print(f"{key}: {value:.6f}" if isinstance(value, float) else f"{key}: {value}")
    return {key: value for key, value in metrics.items() if not key.endswith("_status")}


def analyze_survival(experiments, output_dir, seeds=None, eval_times=(36., 60.)):
    """Generate all fold metrics and figures from exported best-model predictions."""
    if not eval_times or not np.isfinite(eval_times).all() or min(eval_times) <= 0 or len(set(eval_times)) != len(eval_times):
        raise ValueError("Evaluation times must be distinct, positive finite month values")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    names = set()
    for specification in experiments:
        if "=" not in specification:
            raise ValueError("--experiments expects name=results_directory")
        name, template = specification.split("=", 1)
        if not name or Path(name).name != name or name in {".", ".."} or name in names:
            raise ValueError("Experiment names must be unique, nonempty directory names")
        names.add(name)
        runs = experiment_runs(template, seeds)
        if len(runs) > 1:
            raise ValueError("Use a distinct experiment name for each seed to avoid overwriting fold results")
        for seed, root in runs:
            frames = []
            for fold in range(5):
                paths = [root / f"fold_{fold}" / f"{split}_predictions.csv" for split in ("train", "val")]
                if not all(path.is_file() for path in paths):
                    raise FileNotFoundError(f"Missing fold {fold} predictions under {root}. "
                                            "For current models, run the training entry with --eval_only to export predictions.")
                train, val = [read_predictions(path) for path in paths]
                if set(train.case_id) & set(val.case_id):
                    raise ValueError(f"Train/validation patient overlap: {root}, fold={fold}")
                frames.append((train, val))
            # Do not silently produce an incomplete five-fold report.
            all_val_ids = [case for _, val in frames for case in val.case_id]
            cohort = set(all_val_ids)
            if len(cohort) != len(all_val_ids):
                raise ValueError(f"Repeated validation patients across folds: {root}")
            labels = pd.concat([part[["case_id", "dfs.month", "dfs.status"]]
                                for pair in frames for part in pair], ignore_index=True)
            if (labels.groupby("case_id")[["dfs.month", "dfs.status"]].nunique() > 1).any().any():
                raise ValueError(f"Inconsistent patient outcomes across prediction tables: {root}")
            rows = []
            for fold, (train, val) in enumerate(frames):
                if set(train.case_id) | set(val.case_id) != cohort:
                    raise ValueError(f"Inconsistent five-fold cohort: {root}, fold={fold}")
                metrics = analyze_fold(train, val, output_dir / name / f"fold_{fold}", eval_times)
                rows.append({"fold": fold, **metrics})
            from utils import save_cv_results

            save_cv_results(rows, output_dir / name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiments", nargs="+", required=True,
                        help="name=results_directory; optional {seed} placeholder with --seeds")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--eval_times", nargs="+", type=float, default=[36., 60.])
    args = parser.parse_args()
    analyze_survival(args.experiments, args.output_dir, args.seeds, args.eval_times)


if __name__ == "__main__":
    main()
