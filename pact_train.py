import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from monai.data import worker_init_fn
from sksurv.metrics import concordance_index_censored
from torch.utils.data import DataLoader, Subset

from cox_utils import (
    cox_loss,
    evaluate_survival,
)
from dataset import Pa_CT_Dataset
from final_utils import cv_fold_indices, locked_split_indices, seed_everything
from model.build import Pa_CT_Model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def update_ema_variables(model, ema_model, alpha, global_step):
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)
    for ema_buffer, buffer in zip(ema_model.buffers(), model.buffers()):
        ema_buffer.copy_(buffer)


def build_pact_optimizer(model, lr, weight_decay, ct_backbone_lr=None):
    effective_backbone_lr = lr if ct_backbone_lr is None else ct_backbone_lr
    ct_backbone_params, pa_params, fusion_params = [], [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("ct_backbone.") and not name.startswith("ct_backbone.fc."):
            ct_backbone_params.append(param)
        elif name.startswith("pa_branch."):
            pa_params.append(param)
        else:
            fusion_params.append(param)
    groups = []
    if ct_backbone_params:
        groups.append({"params": ct_backbone_params, "lr": effective_backbone_lr})
    if pa_params:
        groups.append({"params": pa_params, "lr": lr})
    if fusion_params:
        groups.append({"params": fusion_params, "lr": lr})
    return torch.optim.Adam(groups, weight_decay=weight_decay)


def train_pact(model, train_loader, val_loader, optimizer, args, device, fold,
               checkpoint_dir, ema_model=None):
    best_cindex = -np.inf
    best_state = None
    cox_batch_size = getattr(args, "cox_batch_size", 64)
    wait = 0
    ema_decay = getattr(args, "ema_decay", 0.99)
    ema_cons_weight = getattr(args, "ema_consistency_weight", 0.3)
    start_ema = getattr(args, "start_ema", 5)
    iter_num = 0

    for epoch in range(1, args.num_epochs + 1):
        model.train()
        if ema_model is not None:
            ema_model.train()
        optimizer.zero_grad()
        losses, losses_fused, losses_ct, losses_pa, losses_ema = [], [], [], [], []
        risks_fused, risks_ct, risks_pa, ema_fused, ema_ct, ema_pa = [], [], [], [], [], []
        times, events = [], []

        for batch in train_loader:
            ct, pa, event, time, _ = batch
            ct = ct.to(device)
            pa = pa.to(device)
            risk_fused, risk_ct, risk_pa = model(ct, pa)[:3]
            risks_fused.append(risk_fused)
            risks_ct.append(risk_ct)
            risks_pa.append(risk_pa)
            times.append(time.to(device))
            events.append(event.to(device))

            if ema_model is not None and epoch >= start_ema:
                with torch.no_grad():
                    er_f, er_c, er_p, _, _, _ = ema_model(ct, pa)
                    ema_fused.append(er_f)
                    ema_ct.append(er_c)
                    ema_pa.append(er_p)

            if len(risks_fused) >= cox_batch_size:
                cat_fused = torch.cat(risks_fused)
                cat_ct = torch.cat(risks_ct)
                cat_pa = torch.cat(risks_pa)
                cat_times = torch.cat(times)
                cat_events = torch.cat(events)
                loss_fused = cox_loss(cat_fused, cat_times, cat_events)
                loss_ct = cox_loss(cat_ct, cat_times, cat_events)
                loss_pa = cox_loss(cat_pa, cat_times, cat_events)
                loss_cox = loss_fused + args.lambda_ct * loss_ct + args.lambda_pa * loss_pa
                if ema_model is not None and epoch >= start_ema and ema_fused:
                    loss_ema = (torch.mean((cat_fused - torch.cat(ema_fused).detach()) ** 2) +
                                torch.mean((cat_ct - torch.cat(ema_ct).detach()) ** 2) +
                                torch.mean((cat_pa - torch.cat(ema_pa).detach()) ** 2)) / 3.0
                    loss = loss_cox + ema_cons_weight * loss_ema
                else:
                    loss_ema = torch.tensor(0.0, device=device)
                    loss = loss_cox
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                if ema_model is not None:
                    update_ema_variables(model, ema_model, ema_decay, iter_num)
                    iter_num += 1
                losses.append(float(loss.detach().cpu()))
                losses_fused.append(float(loss_fused.detach().cpu()))
                losses_ct.append(float(loss_ct.detach().cpu()))
                losses_pa.append(float(loss_pa.detach().cpu()))
                losses_ema.append(float(loss_ema.detach().cpu()))
                risks_fused, risks_ct, risks_pa = [], [], []
                ema_fused, ema_ct, ema_pa = [], [], []
                times, events = [], []

        if risks_fused:
            cat_fused = torch.cat(risks_fused)
            cat_ct = torch.cat(risks_ct)
            cat_pa = torch.cat(risks_pa)
            cat_times = torch.cat(times)
            cat_events = torch.cat(events)
            loss_fused = cox_loss(cat_fused, cat_times, cat_events)
            loss_ct = cox_loss(cat_ct, cat_times, cat_events)
            loss_pa = cox_loss(cat_pa, cat_times, cat_events)
            loss_cox = loss_fused + args.lambda_ct * loss_ct + args.lambda_pa * loss_pa
            if ema_model is not None and epoch >= start_ema and ema_fused:
                loss_ema = (torch.mean((cat_fused - torch.cat(ema_fused).detach()) ** 2) +
                            torch.mean((cat_ct - torch.cat(ema_ct).detach()) ** 2) +
                            torch.mean((cat_pa - torch.cat(ema_pa).detach()) ** 2)) / 3.0
                loss = loss_cox + ema_cons_weight * loss_ema
            else:
                loss_ema = torch.tensor(0.0, device=device)
                loss = loss_cox
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if ema_model is not None:
                update_ema_variables(model, ema_model, ema_decay, iter_num)
                iter_num += 1
            losses.append(float(loss.detach().cpu()))
            losses_fused.append(float(loss_fused.detach().cpu()))
            losses_ct.append(float(loss_ct.detach().cpu()))
            losses_pa.append(float(loss_pa.detach().cpu()))
            losses_ema.append(float(loss_ema.detach().cpu()))

        avg_loss = float(np.mean(losses)) if losses else np.nan
        avg_loss_fused = float(np.mean(losses_fused)) if losses_fused else np.nan
        train_cindex, _, _, _ = evaluate_survival(
            model, train_loader, train_loader, device
        )

        model.eval()
        val_risks_np, val_times_np, val_events_np, val_case_ids = [], [], [], []
        val_ct, val_pa = [], []
        with torch.no_grad():
            for batch in val_loader:
                ct, pa, event, time, case_id = batch
                risk, risk_ct, risk_pa = model(
                    ct.to(device), pa.to(device)
                )[:3]
                val_risks_np.extend(risk.detach().cpu().numpy().reshape(-1).tolist())
                val_times_np.extend(time.detach().cpu().numpy().reshape(-1).tolist())
                val_events_np.extend(event.detach().cpu().numpy().reshape(-1).astype(int).tolist())
                val_case_ids.extend(case_id)
                val_ct.extend(risk_ct.detach().cpu().numpy().reshape(-1).tolist())
                val_pa.extend(risk_pa.detach().cpu().numpy().reshape(-1).tolist())

        val_risks_arr = np.asarray(val_risks_np, dtype=np.float32)
        val_times_arr = np.asarray(val_times_np, dtype=np.float32)
        val_events_arr = np.asarray(val_events_np, dtype=int)
        val_cindex, *_ = concordance_index_censored(val_events_arr.astype(bool), val_times_arr, val_risks_arr)
        val_cindex = float(val_cindex)
        val_ct_cindex = float(concordance_index_censored(val_events_arr.astype(bool), val_times_arr,
                                                          np.asarray(val_ct, dtype=np.float32))[0])
        val_pa_cindex = float(concordance_index_censored(val_events_arr.astype(bool), val_times_arr,
                                                          np.asarray(val_pa, dtype=np.float32))[0])

        print(f"Epoch {epoch}/{args.num_epochs} | Train Loss: {avg_loss:.4f} (Fused: {avg_loss_fused:.4f}) | "
              f"Train C-index: {train_cindex:.4f} | "
              f"Val Fused: {val_cindex:.4f} | Val CT: {val_ct_cindex:.4f} | Val PA: {val_pa_cindex:.4f}")

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
    print(f"Fold {fold} best C-index: {best_cindex:.4f}")
    return model


def parse_args():
    parser = argparse.ArgumentParser(description="Train PA+CT fusion survival model (5-fold CV).")
    parser.add_argument("--data_dir", default="/home/gly001/cqj/pa_ct_surv/data/seed_42")
    parser.add_argument("--ct_roi_size", type=int, default=96, choices=[64, 96, 128])
    parser.add_argument("--ct_model", default="resnet18", choices=["resnet10", "resnet18"])
    parser.add_argument("--pa_model", default="abmil",
                        choices=["abmil", "abmil-topk", "gabmil", "gabmil-topk"])
    parser.add_argument("--k", type=int, default=None,
                        help="Top-k count; required only for *-topk PA models.")
    parser.add_argument("--checkpoint_root", default=None)
    parser.add_argument("--results_root", default=None)
    parser.add_argument("--ct_pretrained_path", type=str, default=None)
    parser.add_argument("--ct_augment", action="store_true")
    parser.add_argument("--fusion_type", default="concat",
                        choices=["concat", "bilinear", "gated", "crossattn", "weighted"])
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--ct_backbone_lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--lambda_ct", type=float, default=0.0)
    parser.add_argument("--lambda_pa", type=float, default=0.0)
    parser.add_argument("--cox_batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed for initialization and training randomness.")
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--ema_decay", type=float, default=0.99)
    parser.add_argument("--ema_consistency_weight", type=float, default=0.3)
    parser.add_argument("--start_ema", type=int, default=5)
    parser.add_argument("--eval_only", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)
    args.effective_ct_backbone_lr = args.lr if args.ct_backbone_lr is None else args.ct_backbone_lr

    aug_tag = "_aug" if args.ct_augment else "_noaug"
    pretrain_tag = "_pretrain" if args.ct_pretrained_path else ""
    is_topk = args.pa_model.endswith("-topk")
    if is_topk and (args.k is None or args.k <= 0):
        raise ValueError("--k must be a positive integer for *-topk PA models")
    if not is_topk and args.k is not None:
        raise ValueError("--k is only valid for *-topk PA models")
    k_tag = f"-k{args.k}" if is_topk else ""
    default_suffix = f"pact-{args.pa_model}{k_tag}-{args.ct_model}-roi{args.ct_roi_size}{aug_tag}{pretrain_tag}-{args.fusion_type}"
    if args.checkpoint_root is None:
        args.checkpoint_root = os.path.join("/home/gly001/cqj/pa_ct_surv", "checkpoints", default_suffix)
    if args.results_root is None:
        args.results_root = os.path.join("/home/gly001/cqj/pa_ct_surv", "results", default_suffix)

    print(f"Using Device: {DEVICE} | ROI: {args.ct_roi_size}")
    print(f"PA: {args.pa_model} | CT: {args.ct_model} | Fusion: {args.fusion_type} | Augment: {args.ct_augment}")
    print(f"LR: {args.lr:g} | CT Backbone LR: {args.effective_ct_backbone_lr:g}")
    print(f"Loss: Fused Cox + {args.lambda_ct:g}*CT + {args.lambda_pa:g}*PA")
    if args.ct_pretrained_path:
        print(f"CT pretrained: {args.ct_pretrained_path}")
    print(f"Checkpoints: {args.checkpoint_root}")
    print(f"Results: {args.results_root}")

    os.makedirs(args.checkpoint_root, exist_ok=True)
    os.makedirs(args.results_root, exist_ok=True)
    with open(os.path.join(args.results_root, "run_config.yaml"), "w") as f:
        yaml.dump(vars(args), f, default_flow_style=False, allow_unicode=True)

    dataset = Pa_CT_Dataset(
        args.data_dir, roi_size=args.ct_roi_size, augment=args.ct_augment
    )
    eval_dataset = Pa_CT_Dataset(
        args.data_dir, roi_size=args.ct_roi_size, augment=False
    )
    print(f"Loaded {len(dataset)} paired samples")

    train_indices, test_indices = locked_split_indices(dataset.samples)
    print(f"Locked split: train={len(train_indices)}, test={len(test_indices)}")

    model_kwargs = {
        "pa_model_name": args.pa_model,
        "ct_model_name": args.ct_model,
        "ct_pretrained_path": args.ct_pretrained_path,
        "pa_topk": args.k if is_topk else None,
        "fusion_type": args.fusion_type,
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
        loader_generator = torch.Generator().manual_seed(fold_seed)

        train_loader = DataLoader(
            Subset(dataset, train_idx),
            batch_size=1,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            worker_init_fn=worker_init_fn,
            generator=loader_generator,
        )
        noaug_train_loader = DataLoader(
            Subset(eval_dataset, train_idx),
            batch_size=1,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            worker_init_fn=worker_init_fn,
        )
        val_loader = DataLoader(
            Subset(eval_dataset, val_idx),
            batch_size=1,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            worker_init_fn=worker_init_fn,
        )

        model = Pa_CT_Model(**model_kwargs).to(DEVICE)
        ema_model = None
        if args.use_ema:
            ema_model = Pa_CT_Model(**model_kwargs).to(DEVICE)
            ema_model.load_state_dict(model.state_dict())
            for p in ema_model.parameters():
                p.detach_()

        optimizer = build_pact_optimizer(model, lr=args.lr, weight_decay=args.weight_decay,
                                         ct_backbone_lr=args.ct_backbone_lr)
        checkpoint_dir = Path(args.checkpoint_root) / f"fold_{fold}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        metrics_dir = Path(args.results_root) / f"fold_{fold}"
        metrics_dir.mkdir(parents=True, exist_ok=True)

        if args.eval_only:
            ckpt_path = checkpoint_dir / "best_model.pth"
            model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
            _, val_cindex, train_df, val_df, metrics = evaluate_survival(
                model,
                noaug_train_loader,
                val_loader,
                DEVICE,
                save_dir=metrics_dir,
            )
            print(f"Fold {fold} eval C-index: {val_cindex:.4f}")
            fold_results.append({"fold": fold, "cindex": val_cindex, **metrics})
            continue

        model = train_pact(
            model,
            train_loader,
            val_loader,
            optimizer,
            args,
            DEVICE,
            fold,
            checkpoint_dir,
            ema_model,
        )
        _, fold_cindex, _, _, metrics = evaluate_survival(
            model,
            noaug_train_loader,
            val_loader,
            DEVICE,
            save_dir=metrics_dir,
        )
        fold_results.append({"fold": fold, "cindex": fold_cindex, **metrics})

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
