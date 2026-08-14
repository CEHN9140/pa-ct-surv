from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test
from sksurv.metrics import concordance_index_censored, cumulative_dynamic_auc
from sksurv.util import Surv


def nll_loss(hazards, S, Y, c, alpha=0.4, eps=1e-7):
    """
    Negative log-likelihood loss for discrete-time survival model
    (MCAT / Nnet-survival style).

    hazards: [B, K]  — per-bin death probabilities h(t) = sigmoid(logits)
    S:       [B, K]  — survival function S(t) = cumprod(1 - h(t))
    Y:       [B]     — discrete time bin index (0, 1, ..., K-1), where larger = later
    c:       [B]     — censorship status (1 = censored, 0 = died)
    alpha:   float   — weight on uncensored loss
    """
    batch_size = len(Y)
    Y = Y.view(batch_size, 1).long()
    c = c.view(batch_size, 1).float()
    if S is None:
        S = torch.cumprod(1 - hazards, dim=1)
    S_padded = torch.cat([torch.ones_like(c), S], 1)  # S(-1)=1
    uncensored_loss = -(1 - c) * (
        torch.log(torch.gather(S_padded, 1, Y).clamp(min=eps))
        + torch.log(torch.gather(hazards, 1, Y).clamp(min=eps))
    )
    censored_loss = -c * torch.log(torch.gather(S_padded, 1, Y + 1).clamp(min=eps))
    neg_l = censored_loss + uncensored_loss
    loss = (1 - alpha) * neg_l + alpha * uncensored_loss
    return loss.mean()


def get_bin_edges(times, events, n_bins=4):
    """Compute bin edges from uncensored event-time quantiles."""
    uncensored = np.asarray(times)[np.asarray(events, dtype=bool)]
    if len(uncensored) < n_bins:
        uncensored = np.asarray(times)
    percentiles = np.linspace(0, 100, n_bins + 1)[1:]
    edges = np.percentile(uncensored, percentiles)
    edges[-1] = np.inf
    return edges


def discretize_time(times, events, bin_edges):
    """
    Map continuous (time, event) to discrete (Y, c).
    bin_edges: [e1, e2, ..., e_{K-1}, inf]
    Y = bin_index (0-based)
    c = 1 if censored else 0
    """
    times = np.asarray(times, dtype=np.float32)
    events = np.asarray(events, dtype=int)
    Y = np.searchsorted(bin_edges, times, side="right").clip(0, len(bin_edges) - 1)
    c = 1 - events
    return Y, c

def pairwise_ranking_loss(risk, time, event, eps=1e-8):
    """
    Smooth pairwise ranking loss (soft C-index loss) for survival analysis.

    For each patient i with an event (event_i=1), compare with every patient j
    who survived longer (Tj > Ti). risk_i should be > risk_j.

    Loss = mean(softplus(risk_j - risk_i)) over all comparable pairs.
    Equivalent to a differentiable surrogate of 1 - C-index.

    risk: [B] scalar risk scores
    time: [B] survival times
    event: [B] 1=death, 0=censored
    """
    risk = risk.view(-1)
    time = time.view(-1)
    event = event.view(-1)

    n = risk.size(0)
    if n < 2:
        return risk.new_tensor(0.0)

    # Build comparison matrix [n, n]
    time_i = time.unsqueeze(0)   # [1, n]
    time_j = time.unsqueeze(1)   # [n, 1]
    event_i = event.unsqueeze(0) # [1, n]

    # comparable[i,j]: patient i has event AND Ti < Tj
    comparable = event_i.bool() & (time_i < time_j)

    n_pairs = comparable.sum()
    if n_pairs == 0:
        return risk.new_tensor(0.0)

    risk_i = risk.unsqueeze(0)   # [1, n]
    risk_j = risk.unsqueeze(1)   # [n, 1]

    # softplus(risk_j - risk_i) = log(1 + exp(risk_j - risk_i))
    # Penalty when risk_j > risk_i (wrong ordering)
    losses = torch.nn.functional.softplus(risk_j - risk_i)

    return (losses * comparable.float()).sum() / n_pairs.float()

def cox_loss(risk, time, event, eps=1e-8):
    risk, time, event = risk.view(-1), time.view(-1), event.view(-1).bool()
    if event.sum() == 0:
        return risk.sum() * 0.0

    loss = risk.new_tensor(0.0)
    for t in torch.unique(time[event]):
        event_mask = (time == t) & event
        risk_set = time >= t
        d_t = event_mask.float().sum()
        loss -= risk[event_mask].sum() - d_t * torch.logsumexp(risk[risk_set], dim=0)
    return loss / (event.float().sum() + eps)


def step_cox_batch(risks, times, events, optimizer, model=None, grad_clip=None):
    loss = cox_loss(torch.cat(risks), torch.cat(times), torch.cat(events))
    loss.backward()
    if model is not None and grad_clip is not None and grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return float(loss.detach().cpu())


def evaluate_survival(model, loader, predict_fn, device):
    model.eval()
    risks, times, events, case_ids = [], [], [], []

    with torch.no_grad():
        for batch in loader:
            risk, event, time, case_id = predict_fn(model, batch, device)
            risks.extend(risk.detach().cpu().numpy().reshape(-1).tolist())
            times.extend(time.detach().cpu().numpy().reshape(-1).tolist())
            events.extend(event.detach().cpu().numpy().reshape(-1).astype(int).tolist())
            case_ids.extend(_as_case_id_list(case_id))

    risks = np.asarray(risks, dtype=np.float32)
    times = np.asarray(times, dtype=np.float32)
    events_arr = np.asarray(events, dtype=int)
    cindex, *_ = concordance_index_censored(events_arr.astype(bool), times, risks)
    return float(cindex), pd.DataFrame(
        {
            "case_id": case_ids,
            "dfs.month": times,
            "dfs.status": events_arr,
            "risk_score": risks,
        }
    )


def evaluate_survival_metrics(
    train_df,
    test_df,
    save_dir,
    time_col="dfs.month",
    event_col="dfs.status",
    risk_col="risk_score",
    eval_times=(36.0, 60.0),
    include_hr=True,
    km_title="Kaplan-Meier Curve: High-risk vs Low-risk",
):
    save_dir = Path(save_dir)
    train_df = clean_survival_df(train_df, time_col, event_col, risk_col)
    test_df = clean_survival_df(test_df, time_col, event_col, risk_col)

    auc_stats = time_dependent_auc(
        train_df, test_df, time_col, event_col, risk_col, eval_times
    )
    cutoff = float(train_df[risk_col].median())
    test_df["risk_group_binary"] = (test_df[risk_col] >= cutoff).astype(int)
    test_df["risk_group"] = np.where(
        test_df["risk_group_binary"] == 1, "High risk", "Low risk"
    )

    metrics = {
        "cutoff_from_train_median": cutoff,
        "n_test": int(len(test_df)),
        "n_high_risk": int((test_df["risk_group_binary"] == 1).sum()),
        "n_low_risk": int((test_df["risk_group_binary"] == 0).sum()),
        "n_event_test": int(test_df[event_col].sum()),
        **plot_km_and_logrank(test_df, save_dir, time_col, event_col, km_title),
        **auc_stats,
    }
    if include_hr:
        metrics.update(cox_hr(test_df, time_col, event_col))

    pd.DataFrame([metrics]).to_csv(save_dir / "survival_extra_metrics.csv", index=False)
    test_df.to_csv(save_dir / "survival_results_with_group.csv", index=False)

    print("\nExtra survival metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value:.6f}" if isinstance(value, float) else f"{key}: {value}")
    return metrics, test_df


def write_test_metrics(
    save_dir, train_cindex, test_cindex, extra_metrics, include_hr=True
):
    save_dir = Path(save_dir)
    with open(save_dir / "test_metrics.txt", "w") as f:
        f.write(f"Train C-index: {train_cindex:.6f}\n")
        f.write(f"Test C-index: {test_cindex:.6f}\n")
        f.write(f"3-year AUC: {extra_metrics.get('auc_36m', np.nan):.6f}\n")
        f.write(f"5-year AUC: {extra_metrics.get('auc_60m', np.nan):.6f}\n")
        f.write(f"Mean AUC: {extra_metrics.get('mean_auc', np.nan):.6f}\n")
        if include_hr:
            f.write(
                f"HR high vs low: {extra_metrics.get('HR_high_vs_low', np.nan):.6f}\n"
            )
        f.write(f"log-rank p: {extra_metrics.get('logrank_p', np.nan):.6g}\n")


def summarize_runs(metrics_df, out_dir):
    out_dir = Path(out_dir)
    metrics_df.to_csv(out_dir / "all_run_test_metrics.csv", index=False)
    numeric_cols = metrics_df.select_dtypes(include=[np.number]).columns
    summary = []
    for col in numeric_cols:
        if col == "run":
            continue
        summary.append(
            {
                "metric": col,
                "mean": float(metrics_df[col].mean()),
                "std": float(metrics_df[col].std(ddof=1)),
                "n": int(metrics_df[col].notna().sum()),
            }
        )
    pd.DataFrame(summary).to_csv(out_dir / "test_metrics_mean_std.csv", index=False)


def summarize_epoch_history(histories, out_dir):
    out_dir = Path(out_dir)
    all_hist = pd.concat(histories, ignore_index=True)
    all_hist.to_csv(out_dir / "all_run_epoch_history.csv", index=False)

    numeric_cols = [
        "loss",
        "train_cindex",
        "val_cindex",
        "best_val_cindex_so_far",
        "best_epoch_so_far",
    ]
    rows = []
    for epoch, group in all_hist.groupby("epoch"):
        row = {"epoch": int(epoch), "count": int(len(group))}
        for col in numeric_cols:
            if col not in group:
                continue
            row[f"{col}_mean"] = float(group[col].mean())
            row[f"{col}_std"] = (
                float(group[col].std(ddof=1)) if len(group) > 1 else np.nan
            )
        rows.append(row)
    pd.DataFrame(rows).sort_values("epoch").to_csv(
        out_dir / "epoch_mean_std.csv", index=False
    )


def summarize_run_files(
    checkpoint_root,
    results_root,
    n_runs,
    history_filename="epoch_history.csv",
    metrics_filename="run_metrics_with_best.csv",
):
    histories, run_metrics = [], []
    checkpoint_root = Path(checkpoint_root)
    results_root = Path(results_root)

    for run_id in range(1, n_runs + 1):
        hist_path = checkpoint_root / f"run_{run_id}" / history_filename
        metric_path = results_root / f"run_{run_id}" / "best_results" / metrics_filename
        if hist_path.exists():
            histories.append(pd.read_csv(hist_path))
        if metric_path.exists():
            run_metrics.append(pd.read_csv(metric_path))

    if histories:
        summarize_epoch_history(histories, results_root)

    if run_metrics:
        all_metrics = pd.concat(run_metrics, ignore_index=True)
        summarize_runs(all_metrics, results_root)


def save_config(args_or_dict, results_dir, device=None):
    results_dir = Path(results_dir)
    cfg = (
        vars(args_or_dict).copy()
        if hasattr(args_or_dict, "__dict__")
        else dict(args_or_dict)
    )
    if device is not None:
        cfg["device"] = str(device)
    pd.DataFrame([cfg]).to_csv(results_dir / "run_config.csv", index=False)
    print("\n========== Experiment config ==========")
    for key, value in cfg.items():
        print(f"{key}: {value}")


def clean_survival_df(
    df, time_col="dfs.month", event_col="dfs.status", risk_col="risk_score"
):
    out = df.copy()
    out[time_col] = out[time_col].astype(float)
    out[event_col] = out[event_col].astype(int)
    out[risk_col] = out[risk_col].astype(float)
    out = out.replace([np.inf, -np.inf], np.nan).dropna(
        subset=[time_col, event_col, risk_col]
    )
    return out[out[time_col] > 0].reset_index(drop=True)


def time_dependent_auc(train_df, test_df, time_col, event_col, risk_col, eval_times):
    y_train = Surv.from_arrays(
        train_df[event_col].astype(bool).values, train_df[time_col].values
    )
    y_test = Surv.from_arrays(
        test_df[event_col].astype(bool).values, test_df[time_col].values
    )
    test_risk = test_df[risk_col].values.astype(float)

    stats, valid_times = {}, []
    max_train_time = train_df[time_col].max()
    max_test_time = test_df[time_col].max()
    for t in eval_times:
        has_case = ((test_df[time_col] <= t) & (test_df[event_col] == 1)).any()
        has_control = (test_df[time_col] > t).any()
        if has_case and has_control and t < max_train_time and t < max_test_time:
            valid_times.append(float(t))
        else:
            stats[f"auc_{int(t)}m"] = np.nan
            print(f"[Warning] Skip {t:g}m AUC: case={has_case}, control={has_control}")

    if not valid_times:
        stats["mean_auc"] = np.nan
        return stats

    try:
        aucs, mean_auc = cumulative_dynamic_auc(
            y_train, y_test, test_risk, np.asarray(valid_times)
        )
    except Exception as exc:
        print(f"[Warning] Time-dependent AUC failed: {exc}")
        for t in valid_times:
            stats[f"auc_{int(t)}m"] = np.nan
        stats["mean_auc"] = np.nan
        return stats

    stats.update({f"auc_{int(t)}m": float(auc) for t, auc in zip(valid_times, aucs)})
    stats["mean_auc"] = float(mean_auc)
    return stats


def plot_km_and_logrank(test_df, save_dir, time_col, event_col, title):
    high = test_df[test_df["risk_group_binary"] == 1]
    low = test_df[test_df["risk_group_binary"] == 0]

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for name, group in (("High risk", high), ("Low risk", low)):
        if len(group):
            KaplanMeierFitter().fit(
                group[time_col], group[event_col], label=f"{name} (n={len(group)})"
            ).plot_survival_function(ax=ax, ci_show=True)

    logrank_p, logrank_stat = np.nan, np.nan
    if len(high) and len(low):
        result = logrank_test(
            high[time_col], low[time_col], high[event_col], low[event_col]
        )
        logrank_p = float(result.p_value)
        logrank_stat = float(result.test_statistic)
        ax.text(0.05, 0.05, f"log-rank p = {logrank_p:.4g}", transform=ax.transAxes)

    ax.set_title(title)
    ax.set_xlabel("DFS time (months)")
    ax.set_ylabel("Disease-free survival probability")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(Path(save_dir) / "km_high_vs_low.png", dpi=300)
    plt.close(fig)
    return {"logrank_stat": logrank_stat, "logrank_p": logrank_p}


def cox_hr(test_df, time_col="dfs.month", event_col="dfs.status"):
    try:
        df = test_df[[time_col, event_col, "risk_group_binary"]].copy()
        cph = CoxPHFitter()
        cph.fit(df, duration_col=time_col, event_col=event_col)
        row = cph.summary.loc["risk_group_binary"]
        return {
            "HR_high_vs_low": float(np.exp(row["coef"])),
            "HR_95CI_lower": float(np.exp(row["coef lower 95%"])),
            "HR_95CI_upper": float(np.exp(row["coef upper 95%"])),
            "cox_p": float(row["p"]),
        }
    except Exception as exc:
        print(f"[Warning] Cox HR failed: {exc}")
        return {
            "HR_high_vs_low": np.nan,
            "HR_95CI_lower": np.nan,
            "HR_95CI_upper": np.nan,
            "cox_p": np.nan,
        }


def _as_case_id_list(case_id):
    if isinstance(case_id, torch.Tensor):
        return case_id.detach().cpu().numpy().reshape(-1).tolist()
    if isinstance(case_id, (list, tuple)):
        return list(case_id)
    return [case_id]
