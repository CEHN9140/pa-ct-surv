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
from torch.utils.data import DataLoader, Subset

from cox_utils import cox_loss, evaluate_survival
from dataset import CT_Student_Dataset
from final_utils import cv_fold_indices, locked_split_indices, seed_everything
from loss import (
    mse_distill_loss,
    normalized_mse_distill_loss,
    risk_set_listwise_kd,
)
from model.build import CT_Model, Pa_CT_Model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# Collate functions — CT batched, PA kept as list
# ============================================================


def paired_collate_fn(batch):
    """Collate paired samples while retaining PA bags as a Python list."""
    cts, pas, events, times, case_ids, clean_cts = zip(*batch)
    ct_batch = torch.stack(cts, dim=0)
    pa_list = list(pas)
    event_batch = torch.stack(events, dim=0).view(-1)
    time_batch = torch.stack(times, dim=0).view(-1)
    return (
        ct_batch,
        pa_list,
        event_batch,
        time_batch,
        list(case_ids),
        torch.stack(clean_cts, dim=0),
    )


# ============================================================
# Teacher loading
# ============================================================


def resolve_teacher_config(teacher_ckpt_root):
    """Find the run_config.yaml paired with a checkpoint root."""
    checkpoint_root = Path(teacher_ckpt_root)
    parts = list(checkpoint_root.parts)
    candidates = []
    if "checkpoints" in parts:
        checkpoint_pos = parts.index("checkpoints")
        candidates.append(
            Path(*parts[:checkpoint_pos], "results", *parts[checkpoint_pos + 1 :])
            / "run_config.yaml"
        )
    candidates.append(checkpoint_root / "run_config.yaml")
    candidates.append(checkpoint_root.parent / "run_config.yaml")

    for config_path in candidates:
        if config_path.is_file():
            with config_path.open("r") as handle:
                return config_path, yaml.safe_load(handle)
    searched = "\n".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Teacher run_config.yaml not found. Searched:\n{searched}")


def load_frozen_teacher(ckpt_path, teacher_config, device):
    """Load a trained PACT model and freeze all parameters."""
    pa_model_name = teacher_config["pa_model"]
    pa_topk = teacher_config.get("k")
    if pa_model_name.endswith("-topk"):
        if pa_topk is None or int(pa_topk) <= 0:
            raise ValueError(
                "Teacher run_config.yaml must contain a positive k for top-k PA"
            )
        pa_topk = int(pa_topk)
    else:
        pa_topk = None

    teacher = Pa_CT_Model(
        pa_model_name=pa_model_name,
        ct_model_name=teacher_config.get("ct_model", "resnet18"),
        ct_pretrained_path=None,
        pa_topk=pa_topk,
        fusion_dropout=float(
            teacher_config.get(
                "fusion_dropout", teacher_config.get("dropout", 0.3)
            )
        ),
        fusion_type=teacher_config.get("fusion_type", "concat"),
        norm=teacher_config.get("norm", "none"),
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
# Distillation training loop (single fold)
# ============================================================


def train_student(
    teacher,
    student,
    train_loader,
    val_loader,
    optimizer,
    args,
    device,
    checkpoint_dir,
):
    best_cindex = -np.inf
    best_state = None
    wait = 0
    alpha_kd = getattr(args, "alpha_kd", 0.3)
    use_distill = alpha_kd > 0

    for epoch in range(1, args.num_epochs + 1):
        # ── Train ──
        student.train()
        if use_distill:
            teacher.eval()
        optimizer.zero_grad()

        losses, losses_cox, losses_kd = [], [], []
        teacher_risks_epoch = []
        teacher_risk_set_entropies = []

        for batch in train_loader:
            ct, pa_list, event, time, _, teacher_ct = batch
            ct = ct.to(device)
            teacher_ct = teacher_ct.to(device)
            event = event.to(device)
            time = time.to(device)
            pa_list = [pa.to(device) for pa in pa_list]

            risk_s = student(ct)

            # ── Teacher fused-risk forward ──
            if use_distill:
                batch_teacher_risks = []
                with torch.no_grad():
                    for i, pa in enumerate(pa_list):
                        risk_t, _, _, _, _, _, _ = teacher(
                            teacher_ct[i : i + 1],
                            pa,
                        )
                        batch_teacher_risks.append(risk_t)
                risk_t = torch.cat(batch_teacher_risks, dim=0)
                teacher_risks_epoch.append(risk_t.detach())
                with torch.no_grad():
                    for event_index in torch.where(event > 0)[0]:
                        risk_set = time >= time[event_index]
                        teacher_prob = torch.softmax(
                            risk_t[risk_set] / args.kd_temperature,
                            dim=0,
                        )
                        entropy = -(
                            teacher_prob
                            * torch.log(teacher_prob.clamp_min(1e-12))
                        ).sum()
                        teacher_risk_set_entropies.append(
                            float(entropy.cpu())
                        )
            else:
                risk_t = None

            loss_c = cox_loss(risk_s, time, event)
            if risk_t is not None:
                if args.distill_mode == "mse":
                    loss_kd = mse_distill_loss(risk_s, risk_t)
                elif args.distill_mode == "normalized_mse":
                    loss_kd = normalized_mse_distill_loss(risk_s, risk_t)
                else:
                    loss_kd = risk_set_listwise_kd(
                        risk_s,
                        risk_t,
                        time,
                        event,
                        temperature=args.kd_temperature,
                    )
                loss = loss_c + alpha_kd * loss_kd
            else:
                loss_kd = torch.tensor(0.0, device=device)
                loss = loss_c

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            losses.append(float(loss.detach().cpu()))
            losses_cox.append(float(loss_c.detach().cpu()))
            losses_kd.append(float(loss_kd.detach().cpu()))

        avg_loss = float(np.mean(losses))
        avg_cox_loss = float(np.mean(losses_cox)) if losses_cox else 0.0
        avg_kd_loss = float(np.mean(losses_kd)) if losses_kd else 0.0
        if teacher_risks_epoch:
            teacher_risk_std = float(
                torch.cat(teacher_risks_epoch).std(unbiased=False).cpu()
            )
        else:
            teacher_risk_std = float("nan")
        teacher_risk_set_entropy = (
            float(np.mean(teacher_risk_set_entropies))
            if teacher_risk_set_entropies
            else float("nan")
        )

        # ── Val ──
        student.eval()
        val_risks_np, val_times_np, val_events_np, val_case_ids = [], [], [], []
        with torch.no_grad():
            for batch in val_loader:
                ct, _, event, time, case_ids, _ = batch
                ct = ct.to(device)
                risk = student(ct)
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
            f"Train Loss: {avg_loss:.4f} (Cox Loss: {avg_cox_loss:.4f}, "
            f"KD Loss: {avg_kd_loss:.4f}) | "
            f"Val C-index: {val_cindex:.4f} | "
            f"Teacher risk std: {teacher_risk_std:.4f} | "
            f"Teacher risk-set entropy: {teacher_risk_set_entropy:.4f}"
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
    return student


# ============================================================
# CLI
# ============================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description="Distill PACT teacher → CT student (batched CT forward)"
    )

    # ── Teacher ──
    parser.add_argument(
        "--teacher_ckpt_root",
        default="/home/gly001/cqj/pa_ct_surv/experiments/final_core/seed42/pact_teacher/checkpoints",
        help="Root dir containing fold_0/best_model.pth ... fold_4/best_model.pth",
    )
    # ── Student ──
    parser.add_argument("--student_model", default="resnet18")
    parser.add_argument(
        "--student_pretrained_path",
        type=str,
        default="/home/gly001/cqj/pa_ct_surv/model/ct_pretrain/resnet_18_23dataset.pth",
    )
    parser.add_argument("--student_dropout", type=float, default=0.5)
    # parser.add_argument(
    #     "--augment",
    #     action="store_true",
    #     help="Enable CT augmentation for student training.",
    # )
    # ── Distillation ──
    parser.add_argument("--alpha_kd", type=float, default=0.3)
    parser.add_argument(
        "--distill_mode",
        default="mse",
        choices=["mse", "normalized_mse", "listwise_kd"],
    )
    parser.add_argument("--kd_temperature", type=float, default=2.0)

    # ── Training ──
    parser.add_argument("--num_epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--ct_backbone_lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument(
        "--cox_batch_size",
        type=int,
        default=64,
        help="Batch size for CT/PA loading and Cox loss updates.",
    )
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for initialization and training randomness.",
    )

    # ── Output ──
    parser.add_argument("--checkpoint_root", default=None)
    parser.add_argument("--results_root", default=None)
    parser.add_argument("--eval_only", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    teacher_config_path, teacher_config = resolve_teacher_config(args.teacher_ckpt_root)
    teacher_pa_model = teacher_config["pa_model"]
    teacher_k = teacher_config.get("k")
    teacher_fusion = teacher_config.get("fusion_type", "concat")
    teacher_ct_model = teacher_config.get("ct_model", "resnet18")
    ct_roi_size = int(teacher_config["ct_roi_size"])
    teacher_seed = teacher_config.get("seed")
    if teacher_seed is not None and int(teacher_seed) != args.seed:
        raise ValueError(
            f"Student seed ({args.seed}) must match teacher seed ({teacher_seed}) "
            "so that teacher and student use the same CSV split."
        )
    data_dir = str(
        Path("/home/gly001/cqj/pa_ct_surv/data") / f"seed_{args.seed}"
    )
    label_file = Path(data_dir) / f"all_label_roi{ct_roi_size}.csv"
    if not label_file.is_file():
        raise FileNotFoundError(f"Dataset CSV not found: {label_file}")
    if not 0.0 <= args.student_dropout < 1.0:
        raise ValueError("--student_dropout must be in [0, 1)")
    args.data_dir = data_dir
    is_teacher_topk = teacher_pa_model.endswith("-topk")

    k_tag = f"-k{teacher_k}" if is_teacher_topk else ""
    suffix = f"distill-{teacher_pa_model}{k_tag}-{teacher_fusion}-survrisk-{args.distill_mode}-akd{args.alpha_kd}-roi{ct_roi_size}"
    if args.checkpoint_root is None:
        args.checkpoint_root = os.path.join(
            "/home/gly001/cqj/pa_ct_surv/experiments/ct_distill", "checkpoints", suffix
        )
    if args.results_root is None:
        args.results_root = os.path.join(
            "/home/gly001/cqj/pa_ct_surv/experiments/ct_distill", "results", suffix
        )

    print(f"Teacher:   {args.teacher_ckpt_root}")
    print(f"Teacher config: {teacher_config_path}")
    print(f"Data:      {data_dir} | ROI: {ct_roi_size}")
    print(
        f"Teacher model: PA={teacher_pa_model} | CT={teacher_ct_model} | "
        f"Fusion={teacher_fusion} | k={teacher_k}"
    )
    print(
        f"Distill:   Cox + alpha_kd*{args.distill_mode}(student_risk, teacher_fused_risk) | alpha_kd={args.alpha_kd} | KD starts at epoch 1"
    )
    print(f"Student dropout: {args.student_dropout}")
    # print(f"Augment:   {args.augment} | cox_batch_size={args.cox_batch_size}")
    print(f"CT augmentation: disabled | cox_batch_size={args.cox_batch_size}")
    print(
        f"LR:        head={args.lr:g} | CT backbone={args.ct_backbone_lr or args.lr:g}"
    )
    print(f"Checkpoints: {args.checkpoint_root}")
    print(f"Results:     {args.results_root}")

    os.makedirs(args.checkpoint_root, exist_ok=True)
    os.makedirs(args.results_root, exist_ok=True)
    with open(os.path.join(args.results_root, "run_config.yaml"), "w") as f:
        run_config = vars(args).copy()
        run_config["teacher_config_path"] = str(teacher_config_path)
        run_config["data_dir"] = data_dir
        run_config["teacher"] = {
            "pa_model": teacher_pa_model,
            "k": teacher_k,
            "fusion_type": teacher_fusion,
            "ct_model": teacher_ct_model,
            "ct_roi_size": ct_roi_size,
        }
        yaml.dump(run_config, f, default_flow_style=False, allow_unicode=True)

    # train_dataset = CT_Student_Dataset(
    #     data_dir, roi_size=ct_roi_size, augment=args.augment
    # )
    train_dataset = CT_Student_Dataset(data_dir, roi_size=ct_roi_size)
    eval_dataset = CT_Student_Dataset(data_dir, roi_size=ct_roi_size)
    print(f"Loaded {len(train_dataset)} paired samples")

    train_indices, test_indices = locked_split_indices(train_dataset.samples)
    print(
        f"Locked split from dataset CSV: train={len(train_indices)}, "
        f"test={len(test_indices)}"
    )

    fold_splits = [cv_fold_indices(train_dataset.samples, fold) for fold in range(5)]
    print("Test set is not accessed during CV")
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(fold_splits):
        print(f"\n{'=' * 50}\nFold {fold + 1}/5\n{'=' * 50}")
        seed_everything(args.seed)

        train_subset = Subset(train_dataset, train_idx)
        # noaug_train_subset = Subset(eval_dataset, train_idx)
        val_subset = Subset(eval_dataset, val_idx)

        train_loader = DataLoader(
            train_subset,
            batch_size=args.cox_batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=paired_collate_fn,
        )
        # noaug_train_loader = DataLoader(
        #     noaug_train_subset,
        #     batch_size=args.cox_batch_size,
        #     shuffle=False,
        #     num_workers=args.num_workers,
        #     pin_memory=torch.cuda.is_available(),
        #     collate_fn=paired_collate_fn,
        # )
        val_loader = DataLoader(
            val_subset,
            batch_size=args.cox_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=paired_collate_fn,
        )

        # ── Load teacher ──
        teacher_ckpt = os.path.join(
            args.teacher_ckpt_root, f"fold_{fold}", "best_model.pth"
        )
        if not os.path.exists(teacher_ckpt):
            raise FileNotFoundError(f"Teacher checkpoint not found: {teacher_ckpt}")
        teacher = load_frozen_teacher(
            teacher_ckpt,
            teacher_config,
            DEVICE,
        )
        print(f"Loaded teacher from: {teacher_ckpt}")

        # ── Create CT student ──
        student = CT_Model(
            model_name=args.student_model,
            pretrained_path=args.student_pretrained_path,
            dropout=args.student_dropout,
        ).to(DEVICE)

        optimizer = build_student_optimizer(
            student, args.lr, args.ct_backbone_lr, args.weight_decay
        )

        checkpoint_dir = Path(args.checkpoint_root) / f"fold_{fold}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        if args.eval_only:
            ckpt_path = checkpoint_dir / "best_model.pth"
            student.load_state_dict(
                torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
            )
        else:
            student = train_student(
                teacher,
                student,
                train_loader,
                val_loader,
                optimizer,
                args,
                DEVICE,
                checkpoint_dir,
            )

        metrics_dir = Path(args.results_root) / f"fold_{fold}"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        _, fold_cindex, _, _, metrics = evaluate_survival(
            student,
            train_loader,
            val_loader,
            DEVICE,
            save_dir=metrics_dir,
        )
        print(f"Fold {fold} eval C-index: {fold_cindex:.4f}")
        fold_results.append({"fold": fold, "cindex": fold_cindex, **metrics})

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
