"""
Knowledge Distillation: PACT Teacher → CT-Only Student (5-fold CV)
Survival risk distillation: Cox supervision + teacher fused-risk MSE.
Batched CT forward + list-based PA teacher forward.
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sksurv.metrics import concordance_index_censored
from torch.utils.data import DataLoader, Dataset, Subset

from cox_utils import (
    cox_loss,
    evaluate_survival_metrics,
)
from dataset import Pa_CT_Dataset
from final_utils import cv_fold_indices, locked_split_indices, save_final_artifacts, seed_everything
from loss import mse_distill_loss
from model.build import CT_Model, Pa_CT_Model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# IndexedSubset
# ============================================================


class IndexedSubset(Dataset):
    """Like torch Subset, but appends the local index (0..len(subset)-1) to each sample."""

    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = list(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        global_idx = self.indices[idx]
        sample = self.dataset[global_idx]
        return sample + (idx,)


# ============================================================
# Collate functions — CT batched, PA kept as list
# ============================================================


def paired_collate_fn(batch):
    """Collate paired samples while retaining PA bags as a Python list."""
    cts, pas, events, times, case_ids, local_indices = zip(*batch)
    ct_batch = torch.stack(cts, dim=0)
    pa_list = list(pas)
    event_batch = torch.stack(events, dim=0).view(-1)
    time_batch = torch.stack(times, dim=0).view(-1)
    local_idx_batch = torch.as_tensor(local_indices, dtype=torch.long)
    return ct_batch, pa_list, event_batch, time_batch, list(case_ids), local_idx_batch


def paired_val_collate_fn(batch):
    """Collate for val Subset (5-element batches, no local index)."""
    cts, pas, events, times, case_ids = zip(*batch)
    ct_batch = torch.stack(cts, dim=0)
    pa_list = list(pas)
    event_batch = torch.stack(events, dim=0).view(-1)
    time_batch = torch.stack(times, dim=0).view(-1)
    return ct_batch, pa_list, event_batch, time_batch, list(case_ids)


# ============================================================
# Teacher loading
# ============================================================


def load_frozen_teacher(
    ckpt_path,
    pa_model_name,
    k,
    fusion_type,
    device,
    ct_model_name="resnet18",
):
    """Load a trained PACT model and freeze all parameters."""
    teacher = Pa_CT_Model(
        pa_model_name=pa_model_name,
        ct_model_name=ct_model_name,
        ct_pretrained_path=None,
        pa_topk=k if pa_model_name.endswith("-topk") else None,
        fusion_type=fusion_type,
    ).to(device)

    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    teacher.load_state_dict(state)

    for param in teacher.parameters():
        param.requires_grad = False
    teacher.eval()
    return teacher


def build_student_optimizer(student, lr, ct_backbone_lr, weight_decay):
    """Use ct_backbone_lr for the CT backbone and lr for the risk head."""
    effective_backbone_lr = lr if ct_backbone_lr is None else ct_backbone_lr
    head_ids = {id(p) for p in student.ct.fc.parameters()}
    backbone_params, head_params = [], []
    for p in student.parameters():
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


# ============================================================
# Teacher batched helper (PA is a list, forward one-by-one)
# ============================================================


def teacher_forward_batch(teacher, ct_batch, pa_list, device):
    """Teacher forward for a batch: PA list loop, CT batch sliced per sample."""
    risks_t = []
    ct_feats_t = []

    with torch.no_grad():
        for i, pa in enumerate(pa_list):
            ct_i = ct_batch[i : i + 1]
            pa_i = pa.to(device, non_blocking=True)
            risk_t, _, _, _, ct_fea_t, _, _ = teacher(ct_i, pa_i)
            risks_t.append(risk_t)
            ct_feats_t.append(ct_fea_t)

    return torch.cat(risks_t, dim=0), torch.cat(ct_feats_t, dim=0)


# ============================================================
# Distillation training loop (single fold)
# ============================================================


def train_distill(
    teacher,
    student,
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
    wait = 0
    cox_bs = getattr(args, "cox_batch_size", 128)
    alpha = getattr(args, "alpha", 0.3)
    start_kd = getattr(args, "start_KD", 5)
    use_distill = alpha > 0

    for epoch in range(1, args.num_epochs + 1):
        # ── Train ──
        student.train()
        if use_distill:
            teacher.eval()
        optimizer.zero_grad()

        risks_s, risks_t = [], []
        times_gpu, events_gpu = [], []

        accum_n = 0
        losses, losses_cox, losses_kd = [], [], []

        for batch_idx, batch in enumerate(train_loader):
            ct, pa_list, event, time, case_ids, _ = batch
            teacher_ct = ct
            ct = ct.to(device, non_blocking=True)
            teacher_ct = teacher_ct.to(device, non_blocking=True)
            event_g = event.to(device, non_blocking=True)
            time_g = time.to(device, non_blocking=True)

            # ── Student forward (batched CT) ──
            ct_fea_s = student.ct.extract_features(ct)  # [B, 512]
            risk_s = student.ct.risk_forward(ct_fea_s)  # [B]

            risks_s.append(risk_s)
            times_gpu.append(time_g)
            events_gpu.append(event_g)

            accum_n += risk_s.numel()

            # ── Teacher fused-risk forward ──
            if use_distill and epoch >= start_kd:
                risk_t, _ = teacher_forward_batch(teacher, teacher_ct, pa_list, device)
                risks_t.append(risk_t)

            # ── Accumulate then backward ──
            if accum_n >= cox_bs:
                cat_risk_s = torch.cat(risks_s)
                loss_c = cox_loss(
                    cat_risk_s, torch.cat(times_gpu), torch.cat(events_gpu)
                )
                if use_distill and epoch >= start_kd:
                    cat_risk_t = torch.cat(risks_t)
                    loss_kd = alpha * mse_distill_loss(cat_risk_s, cat_risk_t)
                    loss = loss_c + loss_kd
                else:
                    loss_kd = torch.tensor(0.0, device=device)
                    loss = loss_c

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                losses.append(float(loss.detach().cpu()))
                losses_cox.append(float(loss_c.detach().cpu()))
                losses_kd.append(float(loss_kd.detach().cpu()) if use_distill else 0.0)

                risks_s, risks_t = [], []
                times_gpu, events_gpu = [], []
                accum_n = 0

        # ── Handle remaining samples ──
        if accum_n > 0:
            cat_risk_s = torch.cat(risks_s)
            loss_c = cox_loss(cat_risk_s, torch.cat(times_gpu), torch.cat(events_gpu))
            if use_distill and epoch >= start_kd and len(risks_t) > 0:
                cat_risk_t = torch.cat(risks_t)
                loss_kd = alpha * mse_distill_loss(cat_risk_s, cat_risk_t)
                loss = loss_c + loss_kd
            else:
                loss_kd = torch.tensor(0.0, device=device)
                loss = loss_c

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            losses.append(float(loss.detach().cpu()))
            losses_cox.append(float(loss_c.detach().cpu()))
            losses_kd.append(
                float(loss_kd.detach().cpu())
                if (use_distill and epoch >= start_kd and len(risks_t) > 0)
                else 0.0
            )

        avg_loss = float(np.mean(losses))
        avg_cox = float(np.mean(losses_cox)) if losses_cox else 0.0
        avg_kd = float(np.mean(losses_kd)) if losses_kd else 0.0

        # ── Val ──
        student.eval()
        val_risks_np, val_times_np, val_events_np, val_case_ids = [], [], [], []
        with torch.no_grad():
            for batch in val_loader:
                ct, _, event, time, case_ids = batch
                ct = ct.to(device, non_blocking=True)
                ct_fea = student.ct.extract_features(ct)
                risk = student.ct.risk_forward(ct_fea)
                val_risks_np.extend(risk.detach().cpu().numpy().reshape(-1).tolist())
                val_times_np.extend(time.numpy().reshape(-1).tolist())
                val_events_np.extend(event.numpy().reshape(-1).astype(int).tolist())
                val_case_ids.extend(case_ids)

        val_risks_arr = np.asarray(val_risks_np, dtype=np.float32)
        val_times_arr = np.asarray(val_times_np, dtype=np.float32)
        val_events_arr = np.asarray(val_events_np, dtype=int)
        val_cindex, *_ = concordance_index_censored(
            val_events_arr.astype(bool), val_times_arr, val_risks_arr
        )
        val_cindex = float(val_cindex)

        print(
            f"Epoch {epoch}/{args.num_epochs} | "
            f"Train Loss: {avg_loss:.4f} (Cox: {avg_cox:.4f}, KD: {avg_kd:.4f}) | "
            f"Val C-index: {val_cindex:.4f}"
        )

        if val_cindex > best_cindex:
            best_cindex = val_cindex
            best_state = {k: v.clone() for k, v in student.state_dict().items()}
            torch.save(best_state, checkpoint_dir / "best_model.pth")
            wait = 0
        else:
            wait += 1

        if args.patience > 0 and wait >= args.patience:
            print(f"Early stopping at epoch {epoch}")
            break

    student.load_state_dict(best_state)
    val_cindex, val_df = evaluate_survival_ct(student, val_loader, device)
    _, train_df = evaluate_survival_ct(student, train_loader, device)
    print(f"Fold {fold} final Val C-index: {val_cindex:.4f}")
    return val_cindex, train_df, val_df


def train_distill_final(teacher, student, train_loader, optimizer, args, device):
    history = []
    use_distill = args.alpha > 0
    for epoch in range(1, args.num_epochs + 1):
        student.train()
        teacher.eval()
        optimizer.zero_grad()
        risks_s, risks_t, times_gpu, events_gpu = [], [], [], []
        accum_n = 0
        losses, cox_losses, kd_losses = [], [], []

        def flush_batch():
            cat_risk_s = torch.cat(risks_s)
            loss_cox = cox_loss(
                cat_risk_s, torch.cat(times_gpu), torch.cat(events_gpu)
            )
            if use_distill and epoch >= args.start_KD and risks_t:
                loss_kd = args.alpha * mse_distill_loss(
                    cat_risk_s, torch.cat(risks_t)
                )
                loss = loss_cox + loss_kd
            else:
                loss_kd = torch.tensor(0.0, device=device)
                loss = loss_cox
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            losses.append(float(loss.detach().cpu()))
            cox_losses.append(float(loss_cox.detach().cpu()))
            kd_losses.append(float(loss_kd.detach().cpu()))

        for batch in train_loader:
            ct, pa_list, event, time, _, _ = batch
            teacher_ct = ct
            ct = ct.to(device, non_blocking=True)
            teacher_ct = teacher_ct.to(device, non_blocking=True)
            risk_s = student.ct.risk_forward(student.ct.extract_features(ct))
            risks_s.append(risk_s)
            times_gpu.append(time.to(device, non_blocking=True))
            events_gpu.append(event.to(device, non_blocking=True))
            accum_n += risk_s.numel()
            if use_distill and epoch >= args.start_KD:
                risk_t, _ = teacher_forward_batch(
                    teacher, teacher_ct, pa_list, device
                )
                risks_t.append(risk_t)
            if accum_n >= args.cox_batch_size:
                flush_batch()
                risks_s, risks_t, times_gpu, events_gpu = [], [], [], []
                accum_n = 0
        if accum_n > 0:
            flush_batch()

        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "cox_loss": float(np.mean(cox_losses)),
            "kd_loss": float(np.mean(kd_losses)),
        }
        history.append(row)
        print(
            f"Final train epoch {epoch}/{args.num_epochs} | "
            f"Loss: {row['train_loss']:.4f} | Cox: {row['cox_loss']:.4f} | "
            f"KD: {row['kd_loss']:.4f}"
        )
    return history


# ============================================================
# CT-specific eval helpers
# ============================================================


def predict_ct_risk_student(model, batch, device):
    """Batched forward — handles val and train batches."""
    if len(batch) == 7:
        ct, _, event, time, case_ids, _, _ = batch
    elif len(batch) == 6:
        ct, _, event, time, case_ids, _ = batch
    else:
        ct, _, event, time, case_ids = batch
    ct = ct.to(device, non_blocking=True)
    ct_fea = model.ct.extract_features(ct)
    risk = model.ct.risk_forward(ct_fea)
    return risk, event, time, case_ids


def evaluate_survival_ct(model, loader, device):
    model.eval()
    risks, times, events, case_ids = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            risk, event, time, case_id_list = predict_ct_risk_student(
                model, batch, device
            )
            risks.extend(risk.detach().cpu().numpy().reshape(-1).tolist())
            times.extend(time.numpy().reshape(-1).tolist())
            events.extend(event.numpy().reshape(-1).astype(int).tolist())
            case_ids.extend(case_id_list)

    risks_arr = np.asarray(risks, dtype=np.float32)
    times_arr = np.asarray(times, dtype=np.float32)
    events_arr = np.asarray(events, dtype=int)
    cindex, *_ = concordance_index_censored(
        events_arr.astype(bool), times_arr, risks_arr
    )
    cindex = float(cindex)
    df = pd.DataFrame(
        {
            "case_id": case_ids,
            "dfs.month": times,
            "dfs.status": events,
            "risk_score": risks,
        }
    )
    return cindex, df


# ============================================================
# CLI
# ============================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description="Distill PACT teacher → CT student (batched CT forward)"
    )

    # ── Data ──
    parser.add_argument("--data_dir", default="/home/gly001/cqj/pa_ct_surv/data/seed_42")
    parser.add_argument("--ct_roi_size", type=int, default=96, choices=[64, 96, 128])

    # ── Teacher ──
    parser.add_argument(
        "--teacher_ckpt_root",
        default="/home/gly001/cqj/pa_ct_surv/experiments/final_core/seed42/pact_teacher/checkpoints",
        help="Root dir containing fold_0/best_model.pth ... fold_4/best_model.pth",
    )
    parser.add_argument(
        "--teacher_final_checkpoint",
        default=None,
        help="Final PACT checkpoint required by --final_train.",
    )
    parser.add_argument(
        "--teacher_pa_model",
        default="abmil",
        choices=["abmil", "abmil-topk", "gabmil", "gabmil-topk"],
    )
    parser.add_argument("--teacher_k", type=int, default=None,
                        help="Top-k count; required only for *-topk teacher models.")
    parser.add_argument(
        "--teacher_fusion", default="concat",
        choices=["concat", "bilinear", "gated", "crossattn"],
    )
    parser.add_argument(
        "--teacher_ct_model", default="resnet18", choices=["resnet10", "resnet18"]
    )

    # ── Student ──
    parser.add_argument(
        "--student_model", default="resnet18", choices=["resnet10", "resnet18"]
    )
    parser.add_argument(
        "--student_pretrained_path",
        type=str,
        default="/home/gly001/cqj/pa_ct_surv/model/ct_pretrain/resnet_18_23dataset.pth",
    )
    parser.add_argument(
        "--augment",
        action="store_true",
        help="Enable CT augmentation for student training.",
    )

    # ── Distillation ──
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--distill_mode", default="mse", choices=["kl", "mse"])
    parser.add_argument("--kd_T", type=float, default=2.0)
    parser.add_argument("--start_KD", type=int, default=5)
    parser.add_argument("--ema_decay", type=float, default=0.99)

    # ── CRD ──
    parser.add_argument("--CRD_weight", type=float, default=0.1)
    parser.add_argument("--nce_k", type=int, default=20)
    parser.add_argument("--nce_p", type=int, default=15)
    parser.add_argument("--nce_p2", type=int, default=10)
    parser.add_argument("--nce_t", type=float, default=0.07)
    parser.add_argument("--nce_m", type=float, default=0.5)
    parser.add_argument("--feat_dim", type=int, default=128)

    # ── Training ──
    parser.add_argument("--num_epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--ct_backbone_lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument(
        "--batch_size", type=int, default=16, help="CT batch size for student forward."
    )
    parser.add_argument(
        "--cox_batch_size",
        type=int,
        default=128,
        help="Number of samples to accumulate before backward.",
    )
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed for initialization and training randomness.")

    # ── Output ──
    parser.add_argument("--checkpoint_root", default=None)
    parser.add_argument("--results_root", default=None)
    parser.add_argument(
        "--final_train",
        action="store_true",
        help="Train one final student on all rows marked split=0..4.",
    )
    parser.add_argument("--eval_only", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.final_train and args.eval_only:
        raise ValueError("--final_train and --eval_only cannot be used together")
    seed_everything(args.seed)

    is_teacher_topk = args.teacher_pa_model.endswith("-topk")
    if is_teacher_topk and (args.teacher_k is None or args.teacher_k <= 0):
        raise ValueError("--teacher_k must be positive for *-topk teacher models")
    if not is_teacher_topk and args.teacher_k is not None:
        raise ValueError("--teacher_k is only valid for *-topk teacher models")

    k_tag = f"-k{args.teacher_k}" if is_teacher_topk else ""
    suffix = f"distill-{args.teacher_pa_model}{k_tag}-{args.teacher_fusion}-survrisk-mse-a{args.alpha}-start{args.start_KD}-roi{args.ct_roi_size}"
    if args.checkpoint_root is None:
        args.checkpoint_root = os.path.join(
            "/home/gly001/cqj/pa_ct_surv/experiments/ct_distill", "checkpoints", suffix
        )
    if args.results_root is None:
        args.results_root = os.path.join(
            "/home/gly001/cqj/pa_ct_surv/experiments/ct_distill", "results", suffix
        )

    print(f"Teacher:   {args.teacher_ckpt_root}")
    print(
        f"Distill:   Cox + alpha*MSE(student_risk, teacher_fused_risk) | alpha={args.alpha} | start_KD={args.start_KD}"
    )
    print(
        f"Augment:   {args.augment} | batch_size={args.batch_size} | cox_batch_size={args.cox_batch_size}"
    )
    print(f"LR:        head={args.lr:g} | CT backbone={args.ct_backbone_lr or args.lr:g}")
    print(f"Checkpoints: {args.checkpoint_root}")
    print(f"Results:     {args.results_root}")

    os.makedirs(args.checkpoint_root, exist_ok=True)
    os.makedirs(args.results_root, exist_ok=True)
    with open(os.path.join(args.results_root, "run_config.yaml"), "w") as f:
        yaml.dump(vars(args), f, default_flow_style=False, allow_unicode=True)

    train_dataset = Pa_CT_Dataset(
        args.data_dir, roi_size=args.ct_roi_size, augment=args.augment
    )
    val_dataset = Pa_CT_Dataset(args.data_dir, roi_size=args.ct_roi_size, augment=False)
    print(f"Loaded {len(train_dataset)} paired samples")

    train_indices, test_indices = locked_split_indices(train_dataset.samples)
    print(
        f"Locked split from dataset CSV: train={len(train_indices)}, "
        f"test={len(test_indices)}"
    )

    if args.final_train:
        if not args.teacher_final_checkpoint:
            raise ValueError("--teacher_final_checkpoint is required with --final_train")
        if not os.path.exists(args.teacher_final_checkpoint):
            raise FileNotFoundError(args.teacher_final_checkpoint)
        train_subset = IndexedSubset(train_dataset, train_indices)
        train_loader = DataLoader(
            train_subset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=paired_collate_fn,
        )
        teacher = load_frozen_teacher(
            args.teacher_final_checkpoint,
            args.teacher_pa_model,
            args.teacher_k,
            args.teacher_fusion,
            DEVICE,
            ct_model_name=args.teacher_ct_model,
        )
        student = CT_Model(
            model_name=args.student_model,
            pretrained_path=args.student_pretrained_path,
        ).to(DEVICE)
        optimizer = build_student_optimizer(student, args.lr, args.ct_backbone_lr, args.weight_decay)
        history = train_distill_final(
            teacher, student, train_loader, optimizer, args, DEVICE
        )
        paths = save_final_artifacts(
            student,
            args,
            args.checkpoint_root,
            args.results_root,
            history,
            model_type="student",
        )
        print(f"Final model: {paths[0]}")
        return

    fold_splits = [
        cv_fold_indices(train_dataset.samples, fold)
        for fold in range(5)
    ]
    print("Test set is not accessed during CV")
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(fold_splits):
        print(f"\n{'=' * 50}\nFold {fold + 1}/5\n{'=' * 50}")
        fold_seed = args.seed + fold
        seed_everything(fold_seed)
        train_generator = torch.Generator().manual_seed(fold_seed)

        train_subset = IndexedSubset(train_dataset, train_idx)
        val_subset = Subset(val_dataset, val_idx)

        train_loader = DataLoader(
            train_subset,
            batch_size=args.batch_size,
            shuffle=True,
            generator=train_generator,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=paired_collate_fn,
        )
        val_loader = DataLoader(
            val_subset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=paired_val_collate_fn,
        )

        # ── Load teacher ──
        teacher_ckpt = os.path.join(
            args.teacher_ckpt_root, f"fold_{fold}", "best_model.pth"
        )
        if not os.path.exists(teacher_ckpt):
            raise FileNotFoundError(f"Teacher checkpoint not found: {teacher_ckpt}")
        teacher = load_frozen_teacher(
            teacher_ckpt,
            args.teacher_pa_model,
            args.teacher_k,
            args.teacher_fusion,
            DEVICE,
            ct_model_name=args.teacher_ct_model,
        )
        print(f"Loaded teacher from: {teacher_ckpt}")

        # ── Create CT student ──
        student = CT_Model(
            model_name=args.student_model,
            pretrained_path=args.student_pretrained_path,
        ).to(DEVICE)

        optimizer = build_student_optimizer(student, args.lr, args.ct_backbone_lr, args.weight_decay)

        checkpoint_dir = Path(args.checkpoint_root) / f"fold_{fold}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        if args.eval_only:
            ckpt_path = checkpoint_dir / "best_model.pth"
            student.load_state_dict(
                torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
            )
            val_cindex, val_df = evaluate_survival_ct(student, val_loader, DEVICE)
            _, train_df = evaluate_survival_ct(student, train_loader, DEVICE)
            print(f"Fold {fold} eval C-index: {val_cindex:.4f}")
            fold_results.append({"fold": fold, "cindex": val_cindex})
            metrics_dir = checkpoint_dir / "best_results"
            metrics_dir.mkdir(parents=True, exist_ok=True)
            evaluate_survival_metrics(train_df, val_df, metrics_dir)
            continue

        fold_cindex, train_df, val_df = train_distill(
            teacher,
            student,
            train_loader,
            val_loader,
            optimizer,
            args,
            DEVICE,
            fold,
            checkpoint_dir,
        )
        fold_results.append({"fold": fold, "cindex": fold_cindex})

        metrics_dir = checkpoint_dir / "best_results"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        evaluate_survival_metrics(train_df, val_df, metrics_dir)

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


def ResNetCox_feat_dim():
    return 512


if __name__ == "__main__":
    main()
