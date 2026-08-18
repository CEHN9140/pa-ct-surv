import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from monai.data import worker_init_fn
from torch.utils.data import DataLoader, Subset

from cox_utils import (
    cox_loss,
    evaluate_survival,
    pairwise_ranking_loss,
)
from dataset import CT_Dataset
from final_utils import cv_fold_indices, locked_split_indices, seed_everything
from model.build import CT_Model

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
        train_cindex, val_cindex, _, val_df = evaluate_survival(
            model, train_eval_loader, val_loader, device
        )
        val_loss = float(
            cox_loss(
                torch.as_tensor(val_df["risk_score"].to_numpy(), device=device),
                torch.as_tensor(val_df["dfs.month"].to_numpy(), device=device),
                torch.as_tensor(val_df["dfs.status"].to_numpy(), device=device),
            ).detach().cpu()
        )

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
    _, val_cindex, train_df, val_df = evaluate_survival(
        model, train_eval_loader, val_loader, device
    )
    print(f"Fold {fold} final C-index: {val_cindex:.4f}")
    return val_cindex, train_df, val_df


def parse_args():
    parser = argparse.ArgumentParser(description="Train CT survival model (5-fold CV).")
    parser.add_argument("--data_dir", default="/home/gly001/cqj/pa_ct_surv/data/seed_42")
    parser.add_argument("--ct_roi_size", type=int, default=96, choices=[64, 96, 128])
    parser.add_argument("--ct_model", default="resnet18", choices=["resnet10", "resnet18"])
    parser.add_argument("--ct_pretrained_path", type=str, default=None)
    parser.add_argument("--ct_augment", action="store_true")
    parser.add_argument("--freeze_backbone", action="store_true")
    parser.add_argument("--freeze_bn_stats", dest="freeze_bn_stats", action="store_true")
    parser.add_argument("--update_bn_stats", dest="freeze_bn_stats", action="store_false")
    parser.set_defaults(freeze_bn_stats=True)
    parser.add_argument("--dropout", type=float, default=0.5)
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
                        help="Seed for initialization and training randomness.")
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--loss_type", default="cox", choices=["cox", "pairwise"])
    return parser.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)

    _ct = "_aug" if args.ct_augment else "_noaug"
    _pr = "_pretrain" if args.ct_pretrained_path else ""
    _bn = "_bnfreeze" if args.freeze_bn_stats else "_bnupdate"
    backbone_lr = args.lr if args.ct_backbone_lr is None else args.ct_backbone_lr
    default_suffix = (
        f"ct-{args.ct_model}-roi{args.ct_roi_size}{_ct}{_pr}"
        f"-bs{args.batch_size}-lr{args.lr:g}-blr{backbone_lr:g}"
        f"-wd{args.weight_decay:g}{_bn}-seed{args.seed}"
    )
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
        "dropout": args.dropout,
        "freeze_bn_stats": args.freeze_bn_stats,
    }

    fold_splits = [
        cv_fold_indices(dataset.samples, fold)
        for fold in range(5)
    ]
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
                                  drop_last=False, num_workers=args.num_workers, pin_memory=True,
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
            model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE, weights_only=True))
            _, val_cindex, _, _ = evaluate_survival(
                model, train_eval_loader, val_loader, DEVICE
            )
            print(f"Fold {fold} eval C-index: {val_cindex:.4f}")
            fold_results.append({"fold": fold, "cindex": val_cindex})
            continue

        fold_cindex, train_df, val_df = train_ct(model, train_loader, train_eval_loader,
                                                  val_loader, predict_ct_risk, optimizer, args,
                                                  DEVICE, fold, checkpoint_dir)
        fold_results.append({"fold": fold, "cindex": fold_cindex})

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
