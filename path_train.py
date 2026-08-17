import argparse
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
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


def aem_loss(weights, eps=1e-12):
    """Negative entropy of already-softmax-normalized attention weights."""
    attention = weights.squeeze(-1).clamp_min(eps)
    return (attention * attention.log()).sum(dim=1).mean()


def get_aem_lambda(lambda0, epoch, total_epochs):
    """Cosine-decay AEM coefficient for the current epoch."""
    progress = (epoch - 1) / max(total_epochs - 1, 1)
    return lambda0 * 0.5 * (1.0 + math.cos(math.pi * progress))


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
    """Evaluate full WSIs and collect per-case attention statistics."""
    rows = []
    model.eval()
    with torch.no_grad():
        for feat, _, _, case_id in loader:
            output = model(feat.to(device, non_blocking=True))
            if not isinstance(output, tuple) or len(output) < 3:
                return pd.DataFrame()
            weights = output[2]
            if weights is None:
                return pd.DataFrame()
            for index in range(weights.size(0)):
                row = attention_statistics(weights[index])
                row.update(
                    {
                        "epoch": epoch,
                        "split": split,
                        "case_id": case_id[index],
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
        total_losses, cox_losses, aem_losses_values = [], [], []
        risks, times, events, aem_losses = [], [], [], []
        all_risks, all_times, all_events = [], [], []
        lambda_aem = get_aem_lambda(
            args.aem_lambda, epoch, args.num_epochs
        )

        for batch in train_loader:
            feat, event, time, case_id = batch
            feat = feat.to(device, non_blocking=True)
            output = model(feat)
            risk = output[0] if isinstance(output, tuple) else output
            all_risks.append(risk.detach().cpu())
            all_times.append(time.detach().cpu())
            all_events.append(event.detach().cpu())
            risks.append(risk)
            times.append(time.to(device))
            events.append(event.to(device))
            if args.aem_lambda > 0:
                if not isinstance(output, tuple) or len(output) < 3:
                    raise ValueError("AEM requires model attention weights")
                if output[2] is None:
                    raise ValueError("AEM requires model attention weights")
                aem_losses.append(aem_loss(output[2]))

            if len(risks) >= cox_batch_size:
                loss_cox = cox_loss(
                    torch.cat(risks), torch.cat(times), torch.cat(events)
                )
                loss_aem = (
                    torch.stack(aem_losses).mean()
                    if aem_losses
                    else loss_cox.new_zeros(())
                )
                loss = loss_cox + lambda_aem * loss_aem
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                total_losses.append(float(loss.detach().cpu()))
                cox_losses.append(float(loss_cox.detach().cpu()))
                aem_losses_values.append(float(loss_aem.detach().cpu()))
                risks, times, events, aem_losses = [], [], [], []

        if risks:
            loss_cox = cox_loss(
                torch.cat(risks), torch.cat(times), torch.cat(events)
            )
            loss_aem = (
                torch.stack(aem_losses).mean()
                if aem_losses
                else loss_cox.new_zeros(())
            )
            loss = loss_cox + lambda_aem * loss_aem
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_losses.append(float(loss.detach().cpu()))
            cox_losses.append(float(loss_cox.detach().cpu()))
            aem_losses_values.append(float(loss_aem.detach().cpu()))
        avg_total_loss = float(np.mean(total_losses)) if total_losses else np.nan
        avg_cox_loss = float(np.mean(cox_losses)) if cox_losses else np.nan
        avg_aem_loss = float(np.mean(aem_losses_values)) if aem_losses_values else 0.0
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
                summary = attention_df.groupby("split")[
                    [
                        "max_attention",
                        "attention_entropy_norm",
                        "effective_patch_num",
                        "effective_patch_ratio",
                    ]
                ].mean()
                summary_text = " | ".join(
                    f"{split} attention: max={row.max_attention:.4f}, "
                    f"entropy={row.attention_entropy_norm:.4f}, "
                    f"effective_patches={row.effective_patch_num:.1f}, "
                    f"effective_ratio={row.effective_patch_ratio:.4f}"
                    for split, row in summary.iterrows()
                )
                print(summary_text)

        print(
            f"Epoch {epoch}/{args.num_epochs} | "
            f"Cox Loss: {avg_cox_loss:.4f} | "
            f"AEM Loss: {avg_aem_loss:.4f} | "
            f"Lambda: {lambda_aem:.6f} | "
            f"Total Loss: {avg_total_loss:.4f} | "
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
            "abmil-proj",
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
        "--aem_lambda",
        type=float,
        default=0.0,
        help="Initial AEM negative-entropy coefficient; cosine-decayed to zero.",
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
        "--proj_dim",
        type=int,
        default=256,
        help="Projection dim for abmil-proj (default: 256).",
    )
    parser.add_argument(
        "--proj_type",
        default="linear",
        choices=["linear", "mlp"],
        help="Projection type for abmil-proj (default: linear).",
    )
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
    if args.aem_lambda < 0:
        raise ValueError("--aem_lambda must be non-negative")
    if args.aem_lambda > 0 and args.pa_model != "abmil":
        raise ValueError("--aem_lambda is currently supported only by standard abmil")
    if args.aem_lambda > 0 and args.patch_sample_size is not None:
        raise ValueError("AEM requires full-patch ABMIL; omit --patch_sample_size")
    k_tag = f"k{args.k}" if is_topk else "all"
    default_suffix = (
        f"path-{args.pa_model}-{k_tag}_cox"
        f"-roi{args.ct_roi_size}-seed{args.seed}"
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
    msg += f" | AEM lambda0: {args.aem_lambda}"
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
        "proj_dim": args.proj_dim,
        "proj_type": args.proj_type,
        "patch_sample_size": args.patch_sample_size,
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
