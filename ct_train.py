import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from monai.data import worker_init_fn
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Subset

from cox_utils import (
    cox_loss,
    evaluate_survival,
    evaluate_survival_metrics,
    pairwise_ranking_loss,
    _as_case_id_list,
)
from dataset import CT_Dataset
from final_utils import locked_split_indices, save_final_artifacts, seed_everything
from model.build import CT_Model
from sksurv.metrics import concordance_index_censored

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def predict_ct_risk(model, batch, device):
    ct, event, time, case_id = batch
    return model(ct.to(device, non_blocking=True)), event, time, case_id


def build_ct_optimizer(model, lr, weight_decay, ct_backbone_lr=None):
    effective_backbone_lr = lr if ct_backbone_lr is None else ct_backbone_lr
    head_ids = {id(p) for p in model.ct.fc.parameters()}
    backbone_params, head_params = [], []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        if id(p) in head_ids:
            head_params.append(p)
        else:
            backbone_params.append(p)
    groups = []
    if backbone_params:
        groups.append({"params": backbone_params, "lr": effective_backbone_lr})
    if head_params:
        groups.append({"params": head_params, "lr": lr})
    return torch.optim.Adam(groups, weight_decay=weight_decay)


def train_ct(model, train_loader, train_eval_loader, val_loader, predict_fn,
             optimizer, args, device, fold, checkpoint_dir):
    best_cindex = -np.inf
    best_state = None
    wait = 0
    loss_fn = pairwise_ranking_loss if getattr(args, "loss_type", "cox") == "pairwise" else cox_loss

    for epoch in range(1, args.num_epochs + 1):
        model.train()
        train_losses = []
        for batch in train_loader:
            risk, event, time, _ = predict_fn(model, batch, device)
            loss = loss_fn(risk, time.to(device), event.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))

        avg_loss = float(np.mean(train_losses)) if train_losses else np.nan
        train_cindex, _ = evaluate_survival(model, train_eval_loader, predict_fn, device)

        model.eval()
        val_risks, val_times, val_events = [], [], []
        val_risks_np, val_times_np, val_events_np, val_case_ids = [], [], [], []
        with torch.no_grad():
            for batch in val_loader:
                risk, event, time, case_id = predict_fn(model, batch, device)
                val_risks.append(risk)
                val_times.append(time.to(device))
                val_events.append(event.to(device))
                val_risks_np.extend(risk.detach().cpu().numpy().reshape(-1).tolist())
                val_times_np.extend(time.detach().cpu().numpy().reshape(-1).tolist())
                val_events_np.extend(event.detach().cpu().numpy().reshape(-1).astype(int).tolist())
                val_case_ids.extend(_as_case_id_list(case_id))
        val_loss = cox_loss(torch.cat(val_risks), torch.cat(val_times), torch.cat(val_events))
        val_risks_arr = np.asarray(val_risks_np, dtype=np.float32)
        val_times_arr = np.asarray(val_times_np, dtype=np.float32)
        val_events_arr = np.asarray(val_events_np, dtype=int)
        val_cindex, *_ = concordance_index_censored(val_events_arr.astype(bool), val_times_arr, val_risks_arr)
        val_cindex = float(val_cindex)

        print(f"Epoch {epoch}/{args.num_epochs} | "
              f"Train Loss: {avg_loss:.4f} | Train C-index: {train_cindex:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val C-index: {val_cindex:.4f}")

        if val_cindex > best_cindex:
            best_cindex = val_cindex
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
        if args.patience > 0 and wait >= args.patience:
            print(f"Early stopping at epoch {epoch}")
            break

    model.load_state_dict(best_state)
    torch.save(best_state, checkpoint_dir / "best_model.pth")
    val_cindex, val_df = evaluate_survival(model, val_loader, predict_fn, device)
    _, train_df = evaluate_survival(model, train_eval_loader, predict_fn, device)
    print(f"Fold {fold} final C-index: {val_cindex:.4f}")
    return val_cindex, train_df, val_df


def train_ct_final(model, train_loader, optimizer, args, device):
    history = []
    loss_fn = pairwise_ranking_loss if getattr(args, "loss_type", "cox") == "pairwise" else cox_loss
    for epoch in range(1, args.num_epochs + 1):
        model.train()
        losses = []
        for batch in train_loader:
            risk, event, time, _ = predict_ct_risk(model, batch, device)
            loss = loss_fn(risk, time.to(device), event.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        avg_loss = float(np.mean(losses)) if losses else np.nan
        history.append({"epoch": epoch, "train_loss": avg_loss})
        print(f"Final train epoch {epoch}/{args.num_epochs} | Loss: {avg_loss:.4f}")
    return history


def parse_args():
    parser = argparse.ArgumentParser(description="Train CT survival model (5-fold CV).")
    parser.add_argument("--data_dir", default="/home/gly001/cqj/pa_ct_surv/data")
    parser.add_argument("--ct_roi_size", type=int, default=96, choices=[64, 96, 128])
    parser.add_argument("--ct_model", default="resnet18", choices=["resnet10", "resnet18"])
    parser.add_argument("--ct_pretrained_path", type=str, default=None)
    parser.add_argument("--ct_augment", action="store_true")
    parser.add_argument("--freeze_backbone", action="store_true")
    parser.add_argument("--checkpoint_root", default=None)
    parser.add_argument("--results_root", default=None)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--ct_backbone_lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=5e-5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed for initialization, training randomness, and CV split.")
    parser.add_argument("--final_train", action="store_true")
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--loss_type", default="cox", choices=["cox", "pairwise"])
    return parser.parse_args()


def main():
    args = parse_args()
    if args.final_train and args.eval_only:
        raise ValueError("--final_train and --eval_only cannot be used together")
    seeds = [args.seed, args.seed]
    seed_everything(args.seed)

    _ct = "_aug" if args.ct_augment else "_noaug"
    _pr = "_pretrain" if args.ct_pretrained_path else ""
    default_suffix = f"ct-{args.ct_model}-roi{args.ct_roi_size}{_ct}{_pr}"
    if args.checkpoint_root is None:
        args.checkpoint_root = os.path.join("/home/gly001/cqj/pa_ct_surv", "checkpoints", default_suffix)
    if args.results_root is None:
        args.results_root = os.path.join("/home/gly001/cqj/pa_ct_surv", "results", default_suffix)

    print(f"Using Device: {DEVICE} | ROI: {args.ct_roi_size}")
    print(f"Model: {args.ct_model} | Augment: {args.ct_augment}")
    print(f"LR: {args.lr:g} | Backbone LR: {args.ct_backbone_lr or args.lr:g} | Loss: {args.loss_type}")
    if args.ct_pretrained_path:
        print(f"Pretrained: {args.ct_pretrained_path}")
    print(f"Checkpoints: {args.checkpoint_root}")
    print(f"Results: {args.results_root}")

    os.makedirs(args.checkpoint_root, exist_ok=True)
    os.makedirs(args.results_root, exist_ok=True)
    with open(os.path.join(args.results_root, "run_config.yaml"), "w") as f:
        yaml.dump(vars(args), f, default_flow_style=False, allow_unicode=True)

    dataset = CT_Dataset(args.data_dir, roi_size=args.ct_roi_size, augment=args.ct_augment)
    val_dataset = CT_Dataset(args.data_dir, roi_size=args.ct_roi_size, augment=False)
    print(f"Loaded {len(dataset)} samples")

    train_indices, test_indices = locked_split_indices(dataset.samples)
    print(f"Locked split: train={len(train_indices)}, test={len(test_indices)}")

    model_kwargs = {
        "model_name": args.ct_model,
        "pretrained_path": args.ct_pretrained_path,
        "freeze_backbone": args.freeze_backbone,
    }

    if args.final_train:
        train_generator = torch.Generator().manual_seed(args.seed)
        train_loader = DataLoader(Subset(dataset, train_indices), batch_size=args.batch_size,
                                  shuffle=True, drop_last=True, num_workers=args.num_workers,
                                  pin_memory=True, worker_init_fn=worker_init_fn, generator=train_generator)
        model = CT_Model(**model_kwargs).to(DEVICE)
        optimizer = build_ct_optimizer(model, lr=args.lr, weight_decay=args.weight_decay,
                                       ct_backbone_lr=args.ct_backbone_lr)
        history = train_ct_final(model, train_loader, optimizer, args, DEVICE)
        paths = save_final_artifacts(model, args, args.checkpoint_root, args.results_root, history, model_type="ct")
        print(f"Final model: {paths[0]}")
        return

    train_labels = dataset.samples.loc[train_indices, "label"].to_numpy()
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    fold_splits = [(train_indices[t], train_indices[v]) for t, v in kf.split(train_indices, train_labels)]
    print("Test set is not accessed during CV")
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(fold_splits):
        print(f"\n{'=' * 50}\nFold {fold + 1}/5\n{'=' * 50}")
        fold_seed = args.seed + fold
        seed_everything(fold_seed)
        train_generator = torch.Generator().manual_seed(fold_seed)

        train_subset = Subset(dataset, train_idx)
        train_eval_subset = Subset(val_dataset, train_idx)
        val_subset = Subset(val_dataset, val_idx)

        train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True,
                                  drop_last=True, num_workers=args.num_workers, pin_memory=True,
                                  worker_init_fn=worker_init_fn, generator=train_generator)
        train_eval_loader = DataLoader(train_eval_subset, batch_size=args.batch_size, shuffle=False,
                                       drop_last=False, num_workers=args.num_workers, pin_memory=True,
                                       worker_init_fn=worker_init_fn)
        val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False,
                                drop_last=False, num_workers=args.num_workers, pin_memory=True,
                                worker_init_fn=worker_init_fn)

        model = CT_Model(**model_kwargs).to(DEVICE)
        optimizer = build_ct_optimizer(model, lr=args.lr, weight_decay=args.weight_decay,
                                       ct_backbone_lr=args.ct_backbone_lr)
        checkpoint_dir = Path(args.checkpoint_root) / f"fold_{fold}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        if args.eval_only:
            ckpt_path = checkpoint_dir / "best_model.pth"
            model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
            _, train_df = evaluate_survival(model, train_eval_loader, predict_ct_risk, DEVICE)
            val_cindex, val_df = evaluate_survival(model, val_loader, predict_ct_risk, DEVICE)
            print(f"Fold {fold} eval C-index: {val_cindex:.4f}")
            fold_results.append({"fold": fold, "cindex": val_cindex})
            continue

        fold_cindex, train_df, val_df = train_ct(model, train_loader, train_eval_loader,
                                                  val_loader, predict_ct_risk, optimizer, args,
                                                  DEVICE, fold, checkpoint_dir)
        fold_results.append({"fold": fold, "cindex": fold_cindex})
        metrics_dir = checkpoint_dir / "best_results"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        evaluate_survival_metrics(train_df, val_df, metrics_dir)

    df = pd.DataFrame(fold_results)
    summary = {"cindex_mean": float(df["cindex"].mean()),
               "cindex_std": float(df["cindex"].std(ddof=1)), "n_folds": len(fold_results)}
    results_dir = Path(args.results_root)
    results_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(results_dir / "cv_fold_metrics.csv", index=False)
    pd.DataFrame([summary]).to_csv(results_dir / "cv_summary.csv", index=False)
    print(f"\n{'=' * 50}\n5-Fold CV Summary\n{'=' * 50}")
    print(f"  C-index: {summary['cindex_mean']:.4f} +/- {summary['cindex_std']:.4f}")


if __name__ == "__main__":
    main()
