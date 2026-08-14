"""Survival analysis visualization."""

import argparse
import sys
from pathlib import Path

# Ensure project root is on sys.path so 'visualization' can be imported
_project_root = Path(__file__).resolve().parents[1]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

from visualization.vis_utils import (
    collect_fold_cindex,
    ensure_dir,
    save_figure,
    set_plot_style,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Survival analysis visualization")
    parser.add_argument(
        "--experiments",
        nargs="+",
        required=True,
        help="List of name=path pairs. Use {seed} placeholder for seed dir.",
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 2024])
    parser.add_argument(
        "--representative_seed", type=int, default=42, help="Seed for single-fold KM"
    )
    return parser.parse_args()


def parse_experiments(exp_specs):
    exps = {}
    for spec in exp_specs:
        if "=" not in spec:
            raise ValueError(f"Invalid spec: {spec}, expected name=path")
        name, path = spec.split("=", 1)
        exps[name.strip()] = path.strip()
    return exps


def _resolve_dir(template, seed):
    """Resolve template with {seed} to a Path."""
    if "{seed}" in template:
        return Path(template.format(seed=seed))
    return Path(template)


def load_survival_metrics_all_folds(exp_template, seeds):
    """Load survival metrics across all folds and seeds."""
    rows = []
    for seed in seeds:
        ckpt_root = _resolve_dir(exp_template, seed) / "checkpoints"
        if not ckpt_root.exists():
            continue
        for fold in range(5):
            metrics_path = (
                ckpt_root
                / f"fold_{fold}"
                / "best_results"
                / "survival_extra_metrics.csv"
            )
            if not metrics_path.exists():
                continue
            df = pd.read_csv(metrics_path)
            row = df.iloc[0].to_dict()
            row["seed"] = seed
            row["fold"] = fold
            rows.append(row)
    return pd.DataFrame(rows)


# ============================================================
# Plot 1: C-index boxplot
# ============================================================


def plot_cindex_boxplot(experiments, output_dir, seeds):
    set_plot_style()
    data, labels = [], []

    for name, template in experiments.items():
        df = collect_fold_cindex(template, seeds)
        if len(df) == 0:
            print(f"[Warn] No data for {name}")
            continue
        data.append(df["cindex"].values)
        labels.append(name.replace("_", " "))

    if not data:
        return

    fig, ax = plt.subplots(figsize=(max(6, len(data) * 1.5), 5))
    bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.5)
    for patch, color in zip(bp["boxes"], plt.cm.Set2(np.linspace(0, 1, len(data)))):
        patch.set_facecolor(color)
    ax.set_ylabel("C-index")
    ax.set_title(f"C-index Distribution ({len(seeds)} seeds × 5 folds)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    save_figure(fig, Path(output_dir) / "cindex_boxplot.png")

    summary = []
    for name, vals in zip(labels, data):
        summary.append(
            {
                "experiment": name,
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals, ddof=1)),
                "n": len(vals),
            }
        )
    pd.DataFrame(summary).to_csv(Path(output_dir) / "cindex_summary.csv", index=False)
    print(f"[OK] cindex_boxplot.png + cindex_summary.csv")


# ============================================================
# Plot 2: 2×2 KM panel
# ============================================================


def plot_km_panel(experiments, output_dir, representative_seed):
    set_plot_style()
    n = len(experiments)
    if n == 0:
        return
    cols = min(2, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
    if rows * cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, (name, template) in enumerate(experiments.items()):
        ckpt_root = _resolve_dir(template, representative_seed) / "checkpoints"
        pred_path = (
            ckpt_root / "fold_0" / "best_results" / "survival_results_with_group.csv"
        )
        if not pred_path.exists():
            axes[idx].text(0.5, 0.5, "No data", ha="center", va="center")
            continue

        df = pd.read_csv(pred_path)
        high = df[df["risk_group_binary"] == 1]
        low = df[df["risk_group_binary"] == 0]
        if len(high) == 0 or len(low) == 0:
            continue

        km_h = KaplanMeierFitter()
        km_l = KaplanMeierFitter()
        km_h.fit(high["dfs.month"], high["dfs.status"], label="High risk")
        km_l.fit(low["dfs.month"], low["dfs.status"], label="Low risk")

        try:
            pval = logrank_test(
                high["dfs.month"],
                low["dfs.month"],
                high["dfs.status"],
                low["dfs.status"],
            ).p_value
        except Exception:
            pval = np.nan

        km_h.plot_survival_function(ax=axes[idx], color="crimson", ci_show=False)
        km_l.plot_survival_function(ax=axes[idx], color="steelblue", ci_show=False)
        axes[idx].set_title(f"{name}\nlog-rank p = {pval:.2e}")
        axes[idx].set_xlabel("DFS months")
        axes[idx].set_ylabel("Survival Probability")
        axes[idx].grid(alpha=0.3)

    for j in range(idx + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        f"Kaplan-Meier Curves (Seed={representative_seed}, Fold=0)", fontsize=14, y=1.01
    )
    fig.tight_layout()
    save_figure(fig, Path(output_dir) / "km_panel.png")
    print(f"[OK] km_panel.png")


# ============================================================
# Plot 3: AUC bar + HR forest
# ============================================================


def plot_auc_hr_summary(experiments, output_dir, seeds):
    set_plot_style()
    rows = []
    for name, template in experiments.items():
        df = load_survival_metrics_all_folds(template, seeds)
        if len(df) == 0:
            print(f"[Warn] No metrics for {name}")
            continue
        rows.append(
            {
                "experiment": name,
                "auc_36m": float(df["auc_36m"].mean()),
                "auc_60m": float(df["auc_60m"].mean()),
                "HR": float(df["HR_high_vs_low"].mean()),
                "HR_lo": float(df["HR_95CI_lower"].mean()),
                "HR_hi": float(df["HR_95CI_upper"].mean()),
            }
        )
    if not rows:
        return
    df = pd.DataFrame(rows)

    # AUC bar
    fig, ax = plt.subplots(figsize=(max(5, len(df) * 1.2), 4))
    x = np.arange(len(df))
    w = 0.35
    ax.bar(x - w / 2, df["auc_36m"], w, label="3-year AUC", color="steelblue")
    ax.bar(x + w / 2, df["auc_60m"], w, label="5-year AUC", color="coral")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [n.replace("_", " ") for n in df["experiment"]], rotation=20, ha="right"
    )
    ax.set_ylabel("AUC")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    save_figure(fig, Path(output_dir) / "auc_barplot.png")

    # HR forest
    fig, ax = plt.subplots(figsize=(7, max(3, len(df) * 0.5)))
    for i, row in df.iterrows():
        y = len(df) - i - 1
        ax.errorbar(
            row["HR"],
            y,
            xerr=[[row["HR"] - row["HR_lo"]], [row["HR_hi"] - row["HR"]]],
            fmt="o",
            color="darkred",
            capsize=5,
        )
        ax.axvline(x=1, color="gray", linestyle="--", alpha=0.5)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels([n.replace("_", " ") for n in df["experiment"]])
    ax.set_xlabel("Hazard Ratio")
    ax.set_title(f"Cox HR (all folds, {len(seeds)} seeds)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    save_figure(fig, Path(output_dir) / "hr_forestplot.png")
    df.to_csv(Path(output_dir) / "survival_metrics_summary.csv", index=False)
    print(f"[OK] auc_barplot + hr_forestplot")


def main():
    args = parse_args()
    experiments = parse_experiments(args.experiments)
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    plot_cindex_boxplot(experiments, output_dir, args.seeds)
    plot_km_panel(experiments, output_dir, args.representative_seed)
    plot_auc_hr_summary(experiments, output_dir, args.seeds)
    print("\nDone.")


if __name__ == "__main__":
    main()
