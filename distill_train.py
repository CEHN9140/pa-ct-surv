"""
Knowledge Distillation: PACT Teacher → CT-Only Student (5-fold CV)
    Survival risk distillation: selected survival objective + teacher fused-risk KD.
Batched CT forward + list-based PA teacher forward.
"""

import argparse
from functools import partial
import os
from pathlib import Path

import numpy as np
import torch
import yaml
from sksurv.metrics import concordance_index_censored

from cox_utils import evaluate_survival
from dataset import CT_Student_Dataset, build_dataloader, resolve_data_dir, cv_splits
from utils import seed_everything, save_run_config, save_cv_results, build_ct_optimizer
from loss import (
    survival_loss,
    mse_distill_loss,
    normalized_mse_distill_loss,
    risk_set_listwise_kd,
)
from model.build import CT_Model, Pa_CT_Model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Teacher loading


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
    if pa_model_name.endswith("-topk") or pa_model_name == "abmil_randsample":
        if pa_topk is None or int(pa_topk) <= 0:
            raise ValueError(
                "Teacher run_config.yaml must contain a positive k for this PA model"
            )
        pa_topk = int(pa_topk)
    else:
        pa_topk = None

    teacher = Pa_CT_Model(
        pa_model_name=pa_model_name,
        ct_model_name=teacher_config.get("ct_model", "resnet18"),
        feature_dim=int(teacher_config.get("fusion_dim", 512)),
        mmhid=int(teacher_config.get("fusion_hidden_dim", 512)),
        fusion_head_dim=teacher_config.get("fusion_head_dim"),
        ct_pretrained_path=None,
        pa_topk=pa_topk,
        fusion_dropout=float(
            teacher_config.get(
                "fusion_dropout", teacher_config.get("dropout", 0.3)
            )
        ),
        fusion_type=teacher_config.get("fusion_type", "concat"),
        dropout=teacher_config.get("dropout"),
        pa_feature_dim=int(teacher_config.get("pa_feature_dim", 1024)),
        pa_l2_normalize=bool(teacher_config.get("pa_l2_normalize", False)),
        fusion_l2_normalize=bool(teacher_config.get("fusion_l2_normalize", False)),
        shared_ct_feature=bool(teacher_config.get("shared_ct_feature", False)),
    ).to(device)

    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    teacher.load_state_dict(state)

    for param in teacher.parameters():
        param.requires_grad = False
    teacher.eval()
    return teacher


# Distillation training loop (single fold)


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
    loss_type = getattr(args, "survival_loss", "cox")
    criterion = partial(survival_loss, loss_type=loss_type,
                        rank_weight=getattr(args, "rank_weight", 1.0))
    loss_label = "Cox Loss" if loss_type == "cox" else "Cox+Rank Loss"

    for epoch in range(1, args.num_epochs + 1):
        student.train()
        if use_distill:
            teacher.eval()
        optimizer.zero_grad()

        losses, losses_survival, losses_kd = [], [], []
        train_risks_np, train_times_np, train_events_np = [], [], []

        for batch in train_loader:
            ct, pa_list, event, time, _ = batch
            ct = ct.to(device)
            event = event.to(device)
            time = time.to(device)
            pa_list = [pa.to(device) for pa in pa_list]

            risk_s = student(ct)
            train_risks_np.extend(
                risk_s.detach().cpu().numpy().reshape(-1).tolist()
            )
            train_times_np.extend(time.detach().cpu().numpy().reshape(-1).tolist())
            train_events_np.extend(
                event.detach().cpu().numpy().reshape(-1).astype(int).tolist()
            )

            if use_distill:
                batch_teacher_risks = []
                with torch.no_grad():
                    for i, pa in enumerate(pa_list):
                        risk_t, _, _, _, _, _, _ = teacher(
                            ct[i : i + 1],
                            pa,
                        )
                        batch_teacher_risks.append(risk_t)
                risk_t = torch.cat(batch_teacher_risks, dim=0)
            else:
                risk_t = None

            loss_surv = criterion(risk_s, time, event)
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
                loss = loss_surv + alpha_kd * loss_kd
            else:
                loss_kd = torch.tensor(0.0, device=device)
                loss = loss_surv

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            losses.append(float(loss.detach().cpu()))
            losses_survival.append(float(loss_surv.detach().cpu()))
            losses_kd.append(float(loss_kd.detach().cpu()))

        avg_loss = float(np.mean(losses))
        train_cindex, *_ = concordance_index_censored(
            np.asarray(train_events_np, dtype=int).astype(bool),
            np.asarray(train_times_np, dtype=np.float32),
            np.asarray(train_risks_np, dtype=np.float32),
        )
        train_cindex = float(train_cindex)
        student.eval()
        val_risks_np, val_times_np, val_events_np = [], [], []
        with torch.no_grad():
            for batch in val_loader:
                ct, _, event, time, _ = batch
                ct = ct.to(device)
                risk = student(ct)
                val_risks_np.extend(risk.detach().cpu().numpy().reshape(-1).tolist())
                val_times_np.extend(time.numpy().reshape(-1).tolist())
                val_events_np.extend(event.numpy().reshape(-1).astype(int).tolist())

        val_risks_arr = np.asarray(val_risks_np, dtype=np.float32)
        val_times_arr = np.asarray(val_times_np, dtype=np.float32)
        val_events_arr = np.asarray(val_events_np, dtype=int)
        val_loss = float(
            criterion(
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
            f"Epoch {epoch}/{args.num_epochs} | Train Loss: {avg_loss:.4f} "
            f"({loss_label}: {np.mean(losses_survival):.4f}, KD Loss: {np.mean(losses_kd):.4f}) | "
            f"Train C-index: {train_cindex:.4f} | Val Loss: {val_loss:.4f} | "
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
    return student


# CLI


def parse_args():
    parser = argparse.ArgumentParser(
        description="Distill PACT teacher → CT student (batched CT forward)"
    )

    parser.add_argument("--data_dir", default=None,
                        help="Directory containing all_label_roi*.csv; folds must match the teacher.")
    parser.add_argument(
        "--teacher_ckpt_root",
        default="/home/gly001/cqj/pa_ct_surv/checkpoints/pact_v6/pact/trident_features_uni_v1_cv5_roi64_abmil_resnet18_weighted_dim256_dropout0.25_epochs30_coxbs64_lr1e-4_blr1e-5_wd5e-4_lct0.5_lpa0.3",
        help="Root dir containing fold_0/best_model.pth ... fold_4/best_model.pth",
    )
    parser.add_argument("--student_model", default="resnet18")
    parser.add_argument(
        "--student_pretrained_path",
        type=str,
        default="/home/gly001/cqj/pa_ct_surv/model/ct_pretrain/resnet_18_23dataset.pth",
    )
    parser.add_argument("--student_dropout", type=float, default=0.5)
    #     "--augment",
    parser.add_argument("--alpha_kd", type=float, default=0.3)
    parser.add_argument(
        "--distill_mode",
        default="mse",
        choices=["mse", "normalized_mse", "listwise_kd"],
    )
    parser.add_argument("--kd_temperature", type=float, default=1.0)

    parser.add_argument("--num_epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--ct_backbone_lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--survival_loss", choices=["cox", "cox_rank"], default="cox")
    parser.add_argument("--rank_weight", type=float, default=1.0,
                        help="Pairwise ranking weight; used only by cox_rank.")
    parser.add_argument(
        "--cox_batch_size",
        type=int,
        default=64,
        help="Batch size for CT/PA loading and Cox loss updates.",
    )
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Seed for initialization and training randomness.",
    )

    parser.add_argument("--checkpoint_root", default=None)
    parser.add_argument("--results_root", default=None)
    parser.add_argument("--eval_only", action="store_true")
    args = parser.parse_args()
    if args.survival_loss == "cox_rank" and (not np.isfinite(args.rank_weight) or args.rank_weight < 0):
        parser.error("--rank_weight must be finite and nonnegative")
    return args


def main():
    args = parse_args()
    seed_everything(args.seed)

    teacher_config_path, teacher_config = resolve_teacher_config(args.teacher_ckpt_root)
    teacher_pa_model = teacher_config["pa_model"]
    teacher_k = teacher_config.get("k")
    teacher_fusion = teacher_config.get("fusion_type", "concat")
    teacher_ct_model = teacher_config.get("ct_model", "resnet18")
    ct_roi_size = int(teacher_config["ct_roi_size"])
    data_dir = resolve_data_dir(args.data_dir or teacher_config.get("data_dir"), ct_roi_size)
    if not 0.0 <= args.student_dropout < 1.0:
        raise ValueError("--student_dropout must be in [0, 1)")
    args.data_dir = data_dir
    is_teacher_topk = teacher_pa_model.endswith("-topk") or teacher_pa_model == "abmil_randsample"

    k_tag = f"-k{teacher_k}" if is_teacher_topk else ""
    suffix = f"distill-{teacher_pa_model}{k_tag}-{teacher_fusion}-survrisk-{args.distill_mode}-akd{args.alpha_kd}-roi{ct_roi_size}"
    if args.survival_loss == "cox_rank":
        suffix += f"-cox_rank-rankw{args.rank_weight:g}"
    print(f"Teacher:   {args.teacher_ckpt_root}")
    print(f"Teacher config: {teacher_config_path}")
    print(f"Data:      {data_dir} | ROI: {ct_roi_size}")
    print(
        f"Teacher model: PA={teacher_pa_model} | CT={teacher_ct_model} | "
        f"Fusion={teacher_fusion} | k={teacher_k}"
    )
    print(
        f"Distill:   {args.survival_loss} + alpha_kd*{args.distill_mode}(student_risk, teacher_fused_risk) | alpha_kd={args.alpha_kd} | KD starts at epoch 1"
    )
    if args.survival_loss == "cox_rank":
        print(f"Ranking weight: {args.rank_weight:g}")
    print(f"Student dropout: {args.student_dropout}")
    print(f"CT augmentation: disabled | cox_batch_size={args.cox_batch_size}")
    print(
        f"LR:        head={args.lr:g} | CT backbone={args.ct_backbone_lr or args.lr:g}"
    )

    teacher_info = {"pa_model": teacher_pa_model, "fusion_type": teacher_fusion,
                    "ct_model": teacher_ct_model, "ct_roi_size": ct_roi_size}
    if is_teacher_topk:
        teacher_info["k"] = teacher_k
    excluded = [] if args.survival_loss == "cox_rank" else ["rank_weight"]
    save_run_config(args, "ct_student", suffix, excluded, extra={
        "teacher_config_path": str(teacher_config_path), "teacher": teacher_info,
    })

    train_dataset = CT_Student_Dataset(data_dir, roi_size=ct_roi_size,
                                      feature_dim=int(teacher_config.get("pa_feature_dim", 1024)))
    print(f"Loaded {len(train_dataset)} paired samples")

    fold_splits = cv_splits(train_dataset.samples)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(fold_splits):
        print(f"\n{'=' * 50}\nFold {fold + 1}/5\n{'=' * 50}")
        seed_everything(args.seed)

        train_loader = build_dataloader(
            train_dataset, indices=train_idx, batch_size=args.cox_batch_size, shuffle=True,
            num_workers=args.num_workers, merge_tail=True, pin_memory=torch.cuda.is_available(),
        )
        val_loader = build_dataloader(
            train_dataset, indices=val_idx, batch_size=args.cox_batch_size,
            num_workers=args.num_workers, pin_memory=torch.cuda.is_available(),
        )

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

        student = CT_Model(
            model_name=args.student_model,
            pretrained_path=args.student_pretrained_path,
            dropout=args.student_dropout,
        ).to(DEVICE)

        optimizer = build_ct_optimizer(
            student, args.lr, args.weight_decay, args.ct_backbone_lr
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

        metrics = evaluate_survival(
            student, train_loader, val_loader, DEVICE,
            save_dir=Path(args.results_root) / f"fold_{fold}",
            loss_type=args.survival_loss, rank_weight=args.rank_weight,
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
