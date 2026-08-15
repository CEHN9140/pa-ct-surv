import argparse
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


def train_path(
    model,
    train_loader,
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
                loss = cox_loss(torch.cat(risks), torch.cat(times), torch.cat(events))
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                losses.append(float(loss.detach().cpu()))
                risks, times, events = [], [], []

        if risks:
            loss = cox_loss(torch.cat(risks), torch.cat(times), torch.cat(events))
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

        print(
            f"Epoch {epoch}/{args.num_epochs} | "
            f"Train Loss: {avg_loss:.4f} | Train C-index: {train_cindex:.4f} | "
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
    return parser.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)
    is_topk = args.pa_model.endswith("-topk")
    if is_topk and (args.k is None or args.k <= 0):
        raise ValueError("--k must be a positive integer for *-topk models")
    if not is_topk and args.k is not None:
        raise ValueError("--k is only valid for *-topk models")
    k_tag = f"k{args.k}" if is_topk else "all"
    default_suffix = f"path-{args.pa_model}-{k_tag}_cox-roi{args.ct_roi_size}"
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

        model = Pa_Model(**model_kwargs).to(DEVICE)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
        checkpoint_dir = Path(args.checkpoint_root) / f"fold_{fold}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        if args.eval_only:
            ckpt_path = checkpoint_dir / "best_model.pth"
            model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
            _, val_cindex, _, _ = evaluate_survival(
                model, train_loader, val_loader, DEVICE
            )
            print(f"Fold {fold} eval C-index: {val_cindex:.4f}")
            fold_results.append({"fold": fold, "cindex": val_cindex})
            continue

        model = train_path(
            model,
            train_loader,
            val_loader,
            optimizer,
            args,
            DEVICE,
            fold,
            checkpoint_dir,
        )
        metrics_dir = checkpoint_dir / "best_results"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        _, fold_cindex, _, _, _ = evaluate_survival(
            model, train_loader, val_loader, DEVICE, save_dir=metrics_dir
        )
        fold_results.append({"fold": fold, "cindex": fold_cindex})

    df = pd.DataFrame(fold_results)
    summary = {
        "cindex_mean": float(df["cindex"].mean()),
        "cindex_std": float(df["cindex"].std(ddof=1)),
        "n_folds": len(fold_results),
    }
    results_dir = Path(args.results_root)
    results_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(results_dir / "cv_fold_metrics.csv", index=False)
    pd.DataFrame([summary]).to_csv(results_dir / "cv_summary.csv", index=False)
    print(f"\n{'=' * 50}\n5-Fold CV Summary\n{'=' * 50}")
    print(f"  C-index: {summary['cindex_mean']:.4f} +/- {summary['cindex_std']:.4f}")


if __name__ == "__main__":
    main()
