import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from sksurv.metrics import concordance_index_censored
from torch.utils.data import DataLoader, Subset

from cox_utils import (
    cox_loss,
    evaluate_survival,
)
from dataset import Path_Dataset
from final_utils import (
    cv_fold_indices,
    locked_split_indices,
    seed_everything,
)
from model.build import Pa_Model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def attention_statistics(weights):
    """Return attention concentration statistics for one WSI."""
    weights = weights.detach().float().reshape(-1)
    weights = weights / weights.sum().clamp_min(torch.finfo(weights.dtype).eps)
    num_patches = int(weights.numel())
    entropy = -(weights * weights.clamp_min(1e-12).log()).sum()
    entropy_norm = entropy / np.log(num_patches) if num_patches > 1 else 0.0
    effective_patch_num = 1.0 / (weights.square().sum().item())
    return {
        "num_patches": num_patches,
        "max_attention": float(weights.max().item()),
        "attention_entropy": float(entropy.item()),
        "attention_entropy_norm": float(entropy_norm),
        "effective_patch_num": float(effective_patch_num),
        "effective_patch_ratio": float(effective_patch_num / num_patches),
    }


def collect_attention_stats(model, loader, device, split, epoch):
    """Evaluate full WSIs and collect per-branch attention statistics."""
    rows = []
    model.eval()
    with torch.no_grad():
        for feat, _, _, case_id in loader:
            output = model(feat.to(device, non_blocking=True))
            if not isinstance(output, tuple) or len(output) < 3:
                return pd.DataFrame()
            branch_attentions = output[2]
            if not torch.is_tensor(branch_attentions):
                return pd.DataFrame()
            for index in range(feat.size(0)):
                case_weights = [
                    branch_attentions[index, :, branch].unsqueeze(-1)
                    for branch in range(branch_attentions.size(2))
                ]
                pairwise_cosines = []
                pairwise_correlations = []
                for left in range(len(case_weights)):
                    for right in range(left + 1, len(case_weights)):
                        left_weights = case_weights[left].reshape(-1)
                        right_weights = case_weights[right].reshape(-1)
                        pairwise_cosines.append(
                            float(
                                F.cosine_similarity(
                                    left_weights,
                                    right_weights,
                                    dim=0,
                                ).item()
                            )
                        )
                        pairwise_correlations.append(
                            float(
                                F.cosine_similarity(
                                    left_weights - left_weights.mean(),
                                    right_weights - right_weights.mean(),
                                    dim=0,
                                ).item()
                            )
                        )
                branch_cosine = (
                    float(np.mean(pairwise_cosines))
                    if pairwise_cosines
                    else np.nan
                )
                branch_correlation = (
                    float(np.mean(pairwise_correlations))
                    if pairwise_correlations
                    else np.nan
                )
                for branch, weights in enumerate(case_weights):
                    row = attention_statistics(weights)
                    row.update(
                        {
                            "epoch": epoch,
                            "split": split,
                            "case_id": case_id[index],
                            "branch": branch,
                            "branch_cosine_mean": branch_cosine,
                            "branch_correlation_mean": branch_correlation,
                        }
                    )
                    rows.append(row)
    return pd.DataFrame(rows)


def train_path(
    model,
    train_loader,
    train_stats_loader,
    val_loader,
    optimizer,
    args,
    device,
    fold,
    checkpoint_dir,
):
    best_cindex = -np.inf
    best_state = None
    cox_batch_size = getattr(args, "cox_batch_size", 64)
    wait = 0

    for epoch in range(1, args.num_epochs + 1):
        model.train()
        optimizer.zero_grad()
        losses, risks, times, events = [], [], [], []
        all_risks, all_times, all_events = [], [], []

        for batch in train_loader:
            feat, event, time, case_id = batch
            feat = feat.to(device, non_blocking=True)
            risk = model(feat)
            if isinstance(risk, tuple):
                risk = risk[0]
            all_risks.append(risk.detach().cpu())
            all_times.append(time.detach().cpu())
            all_events.append(event.detach().cpu())
            risks.append(risk)
            times.append(time.to(device))
            events.append(event.to(device))

            if len(risks) >= cox_batch_size:
                loss = cox_loss(
                    torch.cat(risks), torch.cat(times), torch.cat(events)
                )
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                losses.append(float(loss.detach().cpu()))
                risks, times, events = [], [], []

        if risks:
            loss = cox_loss(
                torch.cat(risks), torch.cat(times), torch.cat(events)
            )
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            losses.append(float(loss.detach().cpu()))
        avg_loss = float(np.mean(losses)) if losses else np.nan
        train_cindex, *_ = concordance_index_censored(
            torch.cat(all_events).numpy().astype(bool),
            torch.cat(all_times).numpy(),
            torch.cat(all_risks).numpy().reshape(-1),
        )
        train_cindex = float(train_cindex)

        model.eval()
        val_risks_np, val_times_np, val_events_np = [], [], []
        with torch.no_grad():
            for batch in val_loader:
                feat, event, time, _ = batch
                output = model(feat.to(device, non_blocking=True))
                risk = output[0] if isinstance(output, tuple) else output
                val_risks_np.extend(risk.detach().cpu().numpy().reshape(-1).tolist())
                val_times_np.extend(time.detach().cpu().numpy().reshape(-1).tolist())
                val_events_np.extend(
                    event.detach().cpu().numpy().reshape(-1).astype(int).tolist()
                )

        val_risks_arr = np.asarray(val_risks_np, dtype=np.float32)
        val_times_arr = np.asarray(val_times_np, dtype=np.float32)
        val_events_arr = np.asarray(val_events_np, dtype=int)

        val_loss = float(
            cox_loss(
                torch.as_tensor(val_risks_arr, device=device),
                torch.as_tensor(val_times_arr, device=device),
                torch.as_tensor(val_events_arr, device=device),
            )
            .detach()
            .cpu()
        )

        val_cindex, *_ = concordance_index_censored(
            val_events_arr.astype(bool), val_times_arr, val_risks_arr
        )
        val_cindex = float(val_cindex)

        if args.pa_model == "abmil" and args.patch_sample_size is None:
            attention_df = pd.concat(
                [
                    collect_attention_stats(
                        model, train_stats_loader, device, "train", epoch
                    ),
                    collect_attention_stats(
                        model, val_loader, device, "val", epoch
                    ),
                ],
                ignore_index=True,
            )
            if not attention_df.empty:
                branch_summary = attention_df.groupby(["split", "branch"])[
                    [
                        "max_attention",
                        "attention_entropy_norm",
                        "effective_patch_num",
                        "effective_patch_ratio",
                    ]
                ].mean()
                for (split, branch), row in branch_summary.iterrows():
                    print(
                        f"{split} branch={int(branch)} attention: "
                        f"max={row.max_attention:.4f}, "
                        f"entropy={row.attention_entropy_norm:.4f}, "
                        f"effective_patches={row.effective_patch_num:.1f}, "
                        f"effective_ratio={row.effective_patch_ratio:.4f}"
                    )
                diversity = attention_df.dropna(
                    subset=[
                        "branch_cosine_mean",
                        "branch_correlation_mean",
                    ]
                ).groupby("split")[
                    ["branch_cosine_mean", "branch_correlation_mean"]
                ].mean()
                for split, row in diversity.iterrows():
                    print(
                        f"{split} branch cosine similarity: "
                        f"{row.branch_cosine_mean:.4f} | "
                        f"Pearson correlation: "
                        f"{row.branch_correlation_mean:.4f}"
                    )

        print(
            f"Epoch {epoch}/{args.num_epochs} | "
            f"Train Loss: {avg_loss:.4f} | "
            f"Train C-index: {train_cindex:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val C-index: {val_cindex:.4f}"
        )

        if val_cindex > best_cindex:
            best_cindex = val_cindex
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            torch.save(model.state_dict(), checkpoint_dir / "best_model.pth")
            wait = 0
        else:
            wait += 1
        if args.patience > 0 and wait >= args.patience:
            print(f"Early stopping at epoch {epoch}")
            break

    model.load_state_dict(best_state)
    print(f"Fold {fold} final C-index: {best_cindex:.4f}")
    return model


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train pathology MIL survival model (5-fold CV)."
    )
    parser.add_argument(
        "--data_dir", default="/home/gly001/cqj/pa_ct_surv/data/seed_42"
    )
    parser.add_argument("--ct_roi_size", type=int, default=96)
    parser.add_argument(
        "--pa_model",
        default="abmil",
        choices=[
            "abmil",
            "abmil-topk",
            "gabmil",
            "gabmil-topk",
            "meanpool",
            "transmil",
        ],
    )
    parser.add_argument(
        "--k",
        type=int,
        default=None,
        help="Top-k count; required only for *-topk models.",
    )
    parser.add_argument("--checkpoint_root", default=None)
    parser.add_argument("--results_root", default=None)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.0,
        help="Dropout after the ABMIL projector ReLU; default 0 disables it.",
    )
    parser.add_argument(
        "--attention_branches",
        type=int,
        default=1,
        help="Number of independent ABMIL attention branches.",
    )
    parser.add_argument("--cox_batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for initialization and training randomness.",
    )
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument(
        "--patch_sample_size",
        type=int,
        default=None,
        help="Random training patch count for ABMIL; validation uses all patches.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)
    is_topk = args.pa_model.endswith("-topk")
    if is_topk and (args.k is None or args.k <= 0):
        raise ValueError("--k must be a positive integer for *-topk models")
    if not is_topk and args.k is not None:
        raise ValueError("--k is only valid for *-topk models")
    if args.patch_sample_size is not None and not args.pa_model.startswith("abmil"):
        raise ValueError("--patch_sample_size is only supported by ABMIL models")
    if not 0.0 <= args.dropout < 1.0:
        raise ValueError("--dropout must be in [0, 1)")
    if args.dropout > 0 and args.pa_model not in {"abmil", "abmil-topk"}:
        raise ValueError("--dropout is currently supported only by ABMIL models")
    if args.attention_branches <= 0:
        raise ValueError("--attention_branches must be positive")
    if args.attention_branches != 1 and args.pa_model not in {
        "abmil",
        "abmil-topk",
    }:
        raise ValueError(
            "--attention_branches is currently supported only by ABMIL models"
        )
    k_tag = f"k{args.k}" if is_topk else "all"
    default_suffix = (
        f"path-{args.pa_model}-{k_tag}_cox"
        f"-roi{args.ct_roi_size}-attn{args.attention_branches}"
        f"-seed{args.seed}"
    )
    if args.checkpoint_root is None:
        args.checkpoint_root = os.path.join(
            "/home/gly001/cqj/pa_ct_surv", "checkpoints", default_suffix
        )
    if args.results_root is None:
        args.results_root = os.path.join(
            "/home/gly001/cqj/pa_ct_surv", "results", default_suffix
        )

    print(f"Using Device: {DEVICE}")
    msg = f"PA model: {args.pa_model} | k: {args.k}"
    msg += " | Cox PH loss"
    msg += f" | ABMIL dropout: {args.dropout}"
    msg += f" | attention_branches: {args.attention_branches}"
    print(msg)
    print(f"Checkpoints: {args.checkpoint_root}")
    print(f"Results: {args.results_root}")

    os.makedirs(args.checkpoint_root, exist_ok=True)
    os.makedirs(args.results_root, exist_ok=True)
    with open(os.path.join(args.results_root, "run_config.yaml"), "w") as f:
        yaml.dump(vars(args), f, default_flow_style=False, allow_unicode=True)

    dataset = Path_Dataset(args.data_dir, roi_size=args.ct_roi_size)
    print(f"Loaded {len(dataset)} samples")

    train_indices, test_indices = locked_split_indices(dataset.samples)
    print(f"Locked split: train={len(train_indices)}, test={len(test_indices)}")

    model_kwargs = {
        "model_name": args.pa_model,
        "feature_dim": 1024,
        "k": args.k if is_topk else None,
        "patch_sample_size": args.patch_sample_size,
        "abmil_dropout": args.dropout,
        "attention_branches": args.attention_branches,
    }
    fold_splits = [cv_fold_indices(dataset.samples, fold) for fold in range(5)]
    print("Test set is not accessed during CV")
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(fold_splits):
        print(f"\n{'=' * 50}\nFold {fold + 1}/5\n{'=' * 50}")
        fold_seed = args.seed + fold
        seed_everything(fold_seed)
        loader_generator = torch.Generator().manual_seed(fold_seed)

        train_loader = DataLoader(
            Subset(dataset, train_idx),
            batch_size=1,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            generator=loader_generator,
        )
        val_loader = DataLoader(
            Subset(dataset, val_idx),
            batch_size=1,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        train_stats_loader = None
        if args.pa_model == "abmil" and args.patch_sample_size is None:
            train_stats_loader = DataLoader(
                Subset(dataset, train_idx),
                batch_size=1,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=True,
            )

        model = Pa_Model(**model_kwargs).to(DEVICE)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
        checkpoint_dir = Path(args.checkpoint_root) / f"fold_{fold}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        metrics_dir = Path(args.results_root) / f"fold_{fold}"
        metrics_dir.mkdir(parents=True, exist_ok=True)

        if args.eval_only:
            ckpt_path = checkpoint_dir / "best_model.pth"
            model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
            _, val_cindex, _, _, metrics = evaluate_survival(
                model, train_loader, val_loader, DEVICE, save_dir=metrics_dir
            )
            print(f"Fold {fold} eval C-index: {val_cindex:.4f}")
            fold_results.append({"fold": fold, "cindex": val_cindex, **metrics})
            continue

        model = train_path(
            model,
            train_loader,
            train_stats_loader,
            val_loader,
            optimizer,
            args,
            DEVICE,
            fold,
            checkpoint_dir,
        )
        _, fold_cindex, _, _, metrics = evaluate_survival(
            model, train_loader, val_loader, DEVICE, save_dir=metrics_dir
        )
        fold_results.append({"fold": fold, "cindex": fold_cindex, **metrics})

    df = pd.DataFrame(fold_results)
    mean_row = {"fold": "mean"}
    for column in df.columns:
        if column != "fold":
            mean_row[column] = pd.to_numeric(df[column], errors="coerce").mean()
    results_df = pd.concat([df, pd.DataFrame([mean_row])], ignore_index=True)
    results_dir = Path(args.results_root)
    results_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(results_dir / "fold_metrics.csv", index=False)
    print(f"\n{'=' * 50}\n5-Fold CV Summary\n{'=' * 50}")
    print(f"  C-index mean: {mean_row['cindex']:.4f}")


if __name__ == "__main__":
    main()
