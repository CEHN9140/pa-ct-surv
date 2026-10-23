import argparse
from pathlib import Path

import numpy as np
import torch
import yaml
from sksurv.metrics import concordance_index_censored

from cox_utils import cox_loss, discrete_nll_loss, evaluate_survival
from dataset import Path_Dataset, build_dataloader, cox_batch_sizes, resolve_data_dir, cv_splits
from utils import seed_everything, save_run_config, save_cv_results
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
    is_nll = args.survival_loss == "nll"
    best_cindex = -np.inf
    best_state = None
    best_epoch = None
    last_epoch = 0
    batch_sizes = ([1] * len(train_loader) if is_nll else
                   cox_batch_sizes(len(train_loader), args.cox_batch_size))
    batch_ends = set(np.cumsum(batch_sizes).tolist())
    wait = 0

    for epoch in range(1, args.num_epochs + 1):
        last_epoch = epoch
        model.train()
        optimizer.zero_grad()
        losses, risks, times, events = [], [], [], []
        all_risks, all_times, all_events = [], [], []

        for patient_index, batch in enumerate(train_loader, start=1):
            feat, event, time, _ = batch
            feat = feat.to(device, non_blocking=True)
            output = model(feat)
            risk = output[0] if isinstance(output, tuple) else output
            all_risks.append(risk.detach().cpu())
            all_times.append(time.detach().cpu())
            all_events.append(event.detach().cpu())
            if is_nll:
                hazards, survival = output[3:5]
                labels = torch.bucketize(time.to(device), model.time_bins).clamp(max=hazards.size(1) - 1)
                loss = discrete_nll_loss(hazards, survival, labels, event.to(device))
            else:
                risks.append(risk)
                times.append(time.to(device))
                events.append(event.to(device))

            if patient_index in batch_ends:
                if not is_nll:
                    loss = cox_loss(torch.cat(risks), torch.cat(times), torch.cat(events))
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                losses.append(float(loss.detach().cpu()))
                risks, times, events = [], [], []

        avg_loss = float(np.mean(losses)) if losses else np.nan
        train_cindex, *_ = concordance_index_censored(
            torch.cat(all_events).numpy().astype(bool),
            torch.cat(all_times).numpy(),
            torch.cat(all_risks).numpy().reshape(-1),
        )
        train_cindex = float(train_cindex)

        model.eval()
        val_risks_np, val_times_np, val_events_np, val_losses = [], [], [], []
        with torch.no_grad():
            for batch in val_loader:
                feat, event, time, _ = batch
                output = model(feat.to(device, non_blocking=True))
                risk = output[0] if isinstance(output, tuple) else output
                if is_nll:
                    hazards, survival = output[3:5]
                    labels = torch.bucketize(time.to(device), model.time_bins).clamp(max=hazards.size(1) - 1)
                    val_losses.append(float(discrete_nll_loss(hazards, survival, labels, event.to(device)).cpu()))
                val_risks_np.extend(risk.detach().cpu().numpy().reshape(-1).tolist())
                val_times_np.extend(time.detach().cpu().numpy().reshape(-1).tolist())
                val_events_np.extend(
                    event.detach().cpu().numpy().reshape(-1).astype(int).tolist()
                )

        val_risks_arr = np.asarray(val_risks_np, dtype=np.float32)
        val_times_arr = np.asarray(val_times_np, dtype=np.float32)
        val_events_arr = np.asarray(val_events_np, dtype=int)

        val_loss = float(np.mean(val_losses)) if is_nll else float(
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
            f"Train Loss: {avg_loss:.4f} | "
            f"Train C-index: {train_cindex:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val C-index: {val_cindex:.4f}"
        )

        if val_cindex > best_cindex:
            best_cindex = val_cindex
            best_epoch = epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            torch.save(model.state_dict(), checkpoint_dir / "best_model.pth")
            wait = 0
        else:
            wait += 1
        if args.patience > 0 and wait >= args.patience:
            print(f"Early stopping at epoch {epoch}")
            break

    # Save the parameters from the final completed epoch before restoring best.
    torch.save(model.state_dict(), checkpoint_dir / "last_model.pth")
    metadata = {"fold": int(fold), "best_epoch": int(best_epoch),
                "last_epoch": int(last_epoch), "best_val_cindex": float(best_cindex)}
    if is_nll:
        metadata.update(survival_loss="discrete_nll", time_bins=model.time_bins.detach().cpu().tolist(),
                        time_bins_source="training_fold_event_time_quartiles")
    with open(checkpoint_dir / "checkpoint_metadata.yaml", "w") as handle:
        yaml.safe_dump(metadata, handle, sort_keys=True)
    model.load_state_dict(best_state)
    print(f"Fold {fold} best C-index: {best_cindex:.4f}")
    return model


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train pathology MIL survival model (5-fold CV)."
    )
    parser.add_argument(
        "--data_dir", default=None,
        help="Directory containing all_label_roi*.csv for the chosen pathology features.",
    )
    parser.add_argument("--ct_roi_size", type=int, default=96)
    parser.add_argument(
        "--pa_model",
        default="abmil",
        choices=[
            "abmil",
            "abmil-topk",
            "abmil_randsample",
            "gabmil",
            "gabmil-topk",
            "meanpool",
            "transmil",
        ],
    )
    parser.add_argument(
        "--k",
        type=int,
        default=4096,
        help="Patch count for *-topk and abmil_randsample; default 4096; ignored by other models.",
    )
    parser.add_argument("--checkpoint_root", default=None)
    parser.add_argument("--results_root", default=None)
    parser.add_argument("--num_epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument(
        "--survival_loss",
        choices=["cox", "nll"],
        default="cox",
        help="Survival loss; NLL uses training-fold event-time quartiles.",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.0,
        help="Dropout in pathology feature projections and aggregation; 0 disables it.",
    )
    parser.add_argument(
        "--attention_branches",
        type=int,
        default=1,
        help="Number of independent ABMIL attention branches.",
    )
    parser.add_argument("--cox_batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Seed for initialization and training randomness.",
    )
    parser.add_argument("--eval_only", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)
    args.data_dir = resolve_data_dir(args.data_dir, args.ct_roi_size)
    label_file = Path(args.data_dir) / f"all_label_roi{args.ct_roi_size}.csv"
    is_topk = args.pa_model.endswith("-topk")
    is_random_sample = args.pa_model == "abmil_randsample"
    if (is_topk or is_random_sample) and (args.k is None or args.k <= 0):
        raise ValueError(
            "--k must be a positive integer for *-topk and abmil_randsample models"
        )
    if not is_topk and not is_random_sample:
        args.k = None
    if not 0.0 <= args.dropout < 1.0:
        raise ValueError("--dropout must be in [0, 1)")
    if args.attention_branches <= 0:
        raise ValueError("--attention_branches must be positive")
    if args.attention_branches != 1 and args.pa_model not in {
        "abmil",
        "abmil-topk",
        "abmil_randsample",
    }:
        raise ValueError(
            "--attention_branches is currently supported only by ABMIL models"
        )
    k_tag = f"k{args.k}" if (is_topk or is_random_sample) else "all"
    default_suffix = (
        f"path-{args.pa_model}-{k_tag}_{args.survival_loss}"
        f"-roi{args.ct_roi_size}-attn{args.attention_branches}"
        f"-seed{args.seed}"
    )
    print(f"Using Device: {DEVICE}")
    print(f"Data: {label_file}")
    msg = f"PA model: {args.pa_model} | k: {args.k}"
    msg += f" | {args.survival_loss} loss"
    msg += f" | Pathology dropout: {args.dropout}"
    msg += f" | attention_branches: {args.attention_branches}"
    print(msg)
    if args.survival_loss == "nll":
        print("NLL cut points: training-fold event-time quartiles (25%, 50%, 75%)")

    excluded = []
    if not is_topk and not is_random_sample:
        excluded.append("k")
    if args.pa_model not in {"abmil", "abmil-topk", "abmil_randsample"}:
        excluded.append("attention_branches")
    if args.survival_loss != "cox":
        excluded.append("cox_batch_size")
    save_run_config(args, "pathology", default_suffix, excluded)

    dataset = Path_Dataset(args.data_dir, roi_size=args.ct_roi_size)
    print(f"Loaded {len(dataset)} samples")

    model_kwargs = {
        "model_name": args.pa_model,
        "feature_dim": 1024,
        "k": args.k if (is_topk or is_random_sample) else None,
        "abmil_dropout": args.dropout,
        "attention_branches": args.attention_branches,
    }
    fold_splits = cv_splits(dataset.samples)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(fold_splits):
        print(f"\n{'=' * 50}\nFold {fold + 1}/5\n{'=' * 50}")
        seed_everything(args.seed)

        checkpoint_dir = Path(args.checkpoint_root) / f"fold_{fold}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        train_loader = build_dataloader(
            dataset, indices=train_idx,
            batch_size=1,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        val_loader = build_dataloader(
            dataset, indices=val_idx,
            batch_size=1,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        time_bins = None
        if args.survival_loss == "nll":
            if args.eval_only:
                state = torch.load(checkpoint_dir / "best_model.pth", map_location="cpu", weights_only=True)
                time_bins = state["time_bins"]
            else:
                train_samples = dataset.samples.iloc[train_idx]
                event_times = train_samples.loc[train_samples["event"] == 1, "time"].to_numpy()
                if len(event_times) == 0:
                    raise ValueError(f"Fold {fold} has no training events for NLL time bins")
                time_bins = np.unique(np.quantile(event_times, [0.25, 0.5, 0.75]).astype(np.float32))
            print(f"Fold {fold} NLL cut points (months): {list(map(float, time_bins))}")
        model_kwargs["time_bins"] = time_bins
        model = Pa_Model(**model_kwargs).to(DEVICE)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
        if args.eval_only:
            model.load_state_dict(torch.load(
                checkpoint_dir / "best_model.pth", map_location=DEVICE, weights_only=True
            ))
        else:
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

        metrics = evaluate_survival(
            model, train_loader, val_loader, DEVICE,
            save_dir=Path(args.results_root) / f"fold_{fold}",
        )
        print(
            f"Fold {fold} | Train Loss: {metrics['train_loss']:.4f} | "
            f"Train C-index: {metrics['train_cindex']:.4f} | "
            f"Val Loss: {metrics['val_loss']:.4f} | Val C-index: {metrics['val_cindex']:.4f}"
        )
        fold_results.append({"fold": fold, **metrics})

    save_cv_results(fold_results, args.results_root)


if __name__ == "__main__":
    main()
