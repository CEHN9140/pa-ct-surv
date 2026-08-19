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


def evaluate_survival(
    model,
    train_loader,
    val_loader,
    device,
    save_dir=None,
    time_col="dfs.month",
    event_col="dfs.status",
    risk_col="risk_score",
    eval_times=(36.0, 60.0),
    include_hr=True,
    km_title="Kaplan-Meier Curve: High-risk vs Low-risk",
):
    predictions = []
    model.eval()
    with torch.no_grad():
        for loader in (train_loader, val_loader):
            risks, times, events, case_ids = [], [], [], []
            for batch in loader:
                feat, event, time, case_id = batch
                output = model(feat.to(device, non_blocking=True))
                risk = output[0] if isinstance(output, tuple) else output
                risks.extend(risk.detach().cpu().numpy().reshape(-1).tolist())
                times.extend(time.detach().cpu().numpy().reshape(-1).tolist())
                events.extend(
                    event.detach().cpu().numpy().reshape(-1).astype(int).tolist()
                )
                case_ids.extend(case_id)

            risks = np.asarray(risks, dtype=np.float32)
            times = np.asarray(times, dtype=np.float32)
            events = np.asarray(events, dtype=int)
            cindex, *_ = concordance_index_censored(
                events.astype(bool), times, risks
            )
            predictions.append(
                (
                    float(cindex),
                    pd.DataFrame(
                        {
                            "case_id": case_ids,
                            "dfs.month": times,
                            "dfs.status": events,
                            "risk_score": risks,
                        }
                    ),
                )
            )

    (train_cindex, train_df), (val_cindex, val_df) = predictions

    if save_dir is None:
        return train_cindex, val_cindex, train_df, val_df

    save_dir = Path(save_dir)
    auc_stats = time_dependent_auc(
        train_df, val_df, time_col, event_col, risk_col, eval_times
    )
    cutoff = float(train_df[risk_col].median())
    val_df["risk_group_binary"] = (val_df[risk_col] >= cutoff).astype(int)
    val_df["risk_group"] = np.where(
        val_df["risk_group_binary"] == 1, "High risk", "Low risk"
    )

    metrics = {
        "cutoff_from_train_median": cutoff,
        "n_test": int(len(val_df)),
        "n_high_risk": int((val_df["risk_group_binary"] == 1).sum()),
        "n_low_risk": int((val_df["risk_group_binary"] == 0).sum()),
        "n_event_test": int(val_df[event_col].sum()),
        **plot_km_and_logrank(val_df, save_dir, time_col, event_col, km_title),
        **auc_stats,
    }
    if include_hr:
        metrics.update(cox_hr(val_df, time_col, event_col))

    val_df.to_csv(save_dir / "val_predictions.csv", index=False)

    print("\nExtra survival metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value:.6f}" if isinstance(value, float) else f"{key}: {value}")
    return train_cindex, val_cindex, train_df, val_df, metrics


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
        return stats

    try:
        aucs, _ = cumulative_dynamic_auc(
            y_train, y_test, test_risk, np.asarray(valid_times)
        )
    except Exception as exc:
        print(f"[Warning] Time-dependent AUC failed: {exc}")
        for t in valid_times:
            stats[f"auc_{int(t)}m"] = np.nan
        return stats

    stats.update({f"auc_{int(t)}m": float(auc) for t, auc in zip(valid_times, aucs)})
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
