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
from sklearn.decomposition import IncrementalPCA

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


class PCAFeatureTransform:
    """Fixed PCA projection fitted on training-fold patch features only."""

    def __init__(self, mean, components):
        self.mean = torch.as_tensor(mean, dtype=torch.float32)
        self.components = torch.as_tensor(components, dtype=torch.float32)

    def __call__(self, features):
        return (features - self.mean) @ self.components.t()


def save_pca_transform(transform, path):
    torch.save(
        {
            "mean": transform.mean,
            "components": transform.components,
            "n_components": int(transform.components.shape[0]),
        },
        path,
    )


def load_pca_transform(path):
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(path, map_location="cpu")
    return PCAFeatureTransform(state["mean"], state["components"])


def fit_patch_pca(
    dataset,
    indices,
    n_components,
    patches_per_patient=256,
    batch_size=4096,
    seed=42,
):
    """Fit IncrementalPCA on a fixed, patient-balanced training-patch sample."""
    if n_components <= 0:
        raise ValueError("pca_dim must be positive")
    pca = IncrementalPCA(n_components=n_components, batch_size=batch_size)
    buffer = []
    buffered_rows = 0
    total_rows = 0
    rng = np.random.default_rng(seed)
    for index in indices:
        path = dataset.samples.iloc[int(index)]["pa_path"]
        feat = torch.load(path, map_location="cpu").float()
        if feat.ndim != 2:
            raise ValueError(f"Expected 2D patch features for PCA, got {tuple(feat.shape)}")
        array = feat.numpy()
        if patches_per_patient is not None and patches_per_patient < len(array):
            selected = rng.choice(len(array), size=patches_per_patient, replace=False)
            array = array[np.sort(selected)]
        buffer.append(array)
        buffered_rows += array.shape[0]
        total_rows += array.shape[0]
        while buffered_rows >= batch_size:
            batch = np.concatenate(buffer, axis=0)
            pca.partial_fit(batch[:batch_size])
            remainder = batch[batch_size:]
            buffer = [remainder] if len(remainder) else []
            buffered_rows = len(remainder)
    fitted_rows = total_rows - buffered_rows
    if buffered_rows >= n_components:
        final_batch = np.concatenate(buffer, axis=0)
        pca.partial_fit(final_batch)
        fitted_rows += len(final_batch)
    if not hasattr(pca, "components_"):
        raise ValueError(
            f"Not enough training patches ({total_rows}) to fit pca_dim={n_components}"
        )
    print(
        f"Fitted PCA on {len(indices)} training patients and {fitted_rows} patches: "
        f"{pca.n_features_in_} -> {n_components}"
    )
    return PCAFeatureTransform(pca.mean_, pca.components_)


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
                    float(np.mean(pairwise_cosines)) if pairwise_cosines else np.nan
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
    best_epoch = None
    last_epoch = 0
    cox_batch_size = getattr(args, "cox_batch_size", 64)
    wait = 0

    for epoch in range(1, args.num_epochs + 1):
        last_epoch = epoch
        model.train()
        optimizer.zero_grad()
        losses, risks, times, events = [], [], [], []
        all_risks, all_times, all_events = [], [], []

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

        if args.pa_model in {"abmil", "abmil_randsample"}:
            attention_df = pd.concat(
                [
                    collect_attention_stats(
                        model, train_stats_loader, device, "train", epoch
                    ),
                    collect_attention_stats(model, val_loader, device, "val", epoch),
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
                diversity = (
                    attention_df.dropna(
                        subset=[
                            "branch_cosine_mean",
                            "branch_correlation_mean",
                        ]
                    )
                    .groupby("split")[["branch_cosine_mean", "branch_correlation_mean"]]
                    .mean()
                )
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
    with open(checkpoint_dir / "checkpoint_metadata.yaml", "w") as f:
        yaml.safe_dump(
            {
                "fold": int(fold),
                "best_epoch": int(best_epoch),
                "last_epoch": int(last_epoch),
                "best_val_cindex": float(best_cindex),
            },
            f,
            sort_keys=True,
        )
    model.load_state_dict(best_state)
    print(f"Fold {fold} best C-index: {best_cindex:.4f}")
    return model


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train pathology MIL survival model (5-fold CV)."
    )
    parser.add_argument("--ct_roi_size", type=int, default=96)
    parser.add_argument(
        "--pca_dim",
        type=int,
        default=None,
        help="Optional PCA dimension for UNI patch features; fitted per training fold only.",
    )
    parser.add_argument(
        "--pca_patches_per_patient",
        type=int,
        default=256,
        help="Number of training patches sampled per patient for PCA fitting.",
    )
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
        default=None,
        help="Patch count k; required for *-topk and abmil_randsample models.",
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
    parser.add_argument("--abmil_hidden_dim", type=int, default=512)
    parser.add_argument("--abmil_attention_dim", type=int, default=128)
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
    return parser.parse_args()


def main():
    args = parse_args()
    args.data_dir = str(
        Path("/home/gly001/cqj/pa_ct_surv/data") / f"seed_{args.seed}"
    )
    label_file = Path(args.data_dir) / f"all_label_roi{args.ct_roi_size}.csv"
    if not label_file.is_file():
        raise FileNotFoundError(f"Dataset CSV not found: {label_file}")
    is_topk = args.pa_model.endswith("-topk")
    is_random_sample = args.pa_model == "abmil_randsample"
    if (is_topk or is_random_sample) and (args.k is None or args.k <= 0):
        raise ValueError(
            "--k must be a positive integer for *-topk and abmil_randsample models"
        )
    if not is_topk and not is_random_sample and args.k is not None:
        raise ValueError(
            "--k is only valid for *-topk and abmil_randsample models"
        )
    if not 0.0 <= args.dropout < 1.0:
        raise ValueError("--dropout must be in [0, 1)")
    if args.pca_dim is not None and args.pca_dim <= 0:
        raise ValueError("--pca_dim must be positive when provided")
    if args.pca_dim is not None and args.pca_dim >= 1024:
        raise ValueError("--pca_dim must be smaller than the original 1024 feature dimensions")
    if args.pca_patches_per_patient <= 0:
        raise ValueError("--pca_patches_per_patient must be positive")
    if args.abmil_hidden_dim <= 0 or args.abmil_attention_dim <= 0:
        raise ValueError("ABMIL hidden and attention dimensions must be positive")
    if args.pca_dim is not None and args.pa_model not in {
        "abmil",
        "abmil-topk",
        "abmil_randsample",
    }:
        raise ValueError("PCA is currently supported only by ABMIL models")
    if args.dropout > 0 and args.pa_model not in {
        "abmil",
        "abmil-topk",
        "abmil_randsample",
    }:
        raise ValueError("--dropout is currently supported only by ABMIL models")
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
    pca_tag = f"-pca{args.pca_dim}" if args.pca_dim is not None else "-pca_none"
    pca_tag += f"-ppp{args.pca_patches_per_patient}"
    pca_tag += f"-hd{args.abmil_hidden_dim}-ad{args.abmil_attention_dim}"
    default_suffix = (
        f"path-{args.pa_model}-{k_tag}_cox"
        f"-roi{args.ct_roi_size}{pca_tag}-attn{args.attention_branches}"
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
    msg += f" | PCA dim: {args.pca_dim if args.pca_dim is not None else 'disabled'}"
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
        "feature_dim": args.pca_dim if args.pca_dim is not None else 1024,
        "k": args.k if (is_topk or is_random_sample) else None,
        "abmil_hidden_dim": args.abmil_hidden_dim,
        "abmil_attention_dim": args.abmil_attention_dim,
        "abmil_dropout": args.dropout,
        "attention_branches": args.attention_branches,
    }
    fold_splits = [cv_fold_indices(dataset.samples, fold) for fold in range(5)]
    print("Test set is not accessed during CV")
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(fold_splits):
        print(f"\n{'=' * 50}\nFold {fold + 1}/5\n{'=' * 50}")
        seed_everything(args.seed)

        checkpoint_dir = Path(args.checkpoint_root) / f"fold_{fold}"
        metrics_dir = Path(args.results_root) / f"fold_{fold}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        metrics_dir.mkdir(parents=True, exist_ok=True)
        if args.pca_dim is not None:
            pca_path = checkpoint_dir / "pca_transform.pt"
            if args.eval_only:
                if not pca_path.is_file():
                    raise FileNotFoundError(f"PCA transform not found: {pca_path}")
                pca_transform = load_pca_transform(pca_path)
            else:
                pca_transform = fit_patch_pca(
                    dataset,
                    train_idx,
                    args.pca_dim,
                    patches_per_patient=args.pca_patches_per_patient,
                    seed=args.seed + fold,
                )
                save_pca_transform(pca_transform, pca_path)
            dataset.set_pca_transform(pca_transform)
        else:
            dataset.set_pca_transform(None)

        train_loader = DataLoader(
            Subset(dataset, train_idx),
            batch_size=1,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        val_loader = DataLoader(
            Subset(dataset, val_idx),
            batch_size=1,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        train_stats_loader = None
        if args.pa_model in {"abmil", "abmil_randsample"}:
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
        if args.eval_only:
            ckpt_path = checkpoint_dir / "best_model.pth"
            model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
            train_cindex, val_cindex, _, _, metrics = evaluate_survival(
                model, train_loader, val_loader, DEVICE, save_dir=metrics_dir
            )
            gap = train_cindex - val_cindex
            print(
                f"Fold {fold} | Train C-index: {train_cindex:.4f} | "
                f"Val C-index: {val_cindex:.4f} | Gap: {gap:.4f}"
            )
            fold_results.append(
                {
                    "fold": fold,
                    "cindex": val_cindex,
                    "train_cindex": train_cindex,
                    "val_cindex": val_cindex,
                    "gap": gap,
                    **metrics,
                }
            )
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
        train_cindex, val_cindex, _, _, metrics = evaluate_survival(
            model, train_loader, val_loader, DEVICE, save_dir=metrics_dir
        )
        gap = train_cindex - val_cindex
        print(
            f"Fold {fold} | Train C-index: {train_cindex:.4f} | "
            f"Val C-index: {val_cindex:.4f} | Gap: {gap:.4f}"
        )
        fold_results.append(
            {
                "fold": fold,
                "cindex": val_cindex,
                "train_cindex": train_cindex,
                "val_cindex": val_cindex,
                "gap": gap,
                **metrics,
            }
        )

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
