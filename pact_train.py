import argparse
from pathlib import Path

import numpy as np
import torch
from sksurv.metrics import concordance_index_censored

from cox_utils import (
    cox_loss,
    evaluate_survival,
)
from dataset import Pa_CT_Dataset, build_dataloader, cox_batch_sizes, resolve_data_dir, cv_splits
from utils import seed_everything, save_run_config, save_cv_results
from model.build import Pa_CT_Model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
               checkpoint_dir):
    best_cindex = -np.inf
    best_state = None
    cox_batch_size = getattr(args, "cox_batch_size", 64)
    batch_sizes = cox_batch_sizes(len(train_loader), cox_batch_size)
    batch_ends = set(np.cumsum(batch_sizes).tolist())
    wait = 0
    for epoch in range(1, args.num_epochs + 1):
        model.train()
        optimizer.zero_grad()
        losses, losses_fused, losses_ct, losses_pa = [], [], [], []
        losses_aux = []
        auxiliary_payloads = []
        risks_fused, risks_ct, risks_pa = [], [], []
        times, events = [], []
        train_risks_fused_np, train_times_np, train_events_np = [], [], []

        for patient_index, batch in enumerate(train_loader, start=1):
            ct, pa, event, time, _ = batch
            ct = ct.to(device)
            pa = pa.to(device)
            risk_fused, risk_ct, risk_pa, auxiliary = model(ct, pa)[:4]
            if auxiliary:
                auxiliary_payloads.append(auxiliary)
            train_risks_fused_np.extend(
                risk_fused.detach().cpu().numpy().reshape(-1).tolist()
            )
            train_times_np.extend(time.numpy().reshape(-1).tolist())
            train_events_np.extend(
                event.numpy().reshape(-1).astype(int).tolist()
            )
            risks_fused.append(risk_fused)
            risks_ct.append(risk_ct)
            risks_pa.append(risk_pa)
            times.append(time.to(device))
            events.append(event.to(device))

            if patient_index in batch_ends:
                cat_fused = torch.cat(risks_fused)
                cat_ct = torch.cat(risks_ct)
                cat_pa = torch.cat(risks_pa)
                cat_times = torch.cat(times)
                cat_events = torch.cat(events)
                loss_fused = cox_loss(cat_fused, cat_times, cat_events)
                loss_ct = cox_loss(cat_ct, cat_times, cat_events)
                loss_pa = cox_loss(cat_pa, cat_times, cat_events)
                loss = loss_fused + args.lambda_ct * loss_ct + args.lambda_pa * loss_pa
                loss_aux = model.fusion_auxiliary_loss(auxiliary_payloads) if auxiliary_payloads else loss.new_zeros(())
                loss = loss + getattr(args, "fusion_aux_weight", 0.1) * loss_aux
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                losses.append(float(loss.detach().cpu()))
                losses_fused.append(float(loss_fused.detach().cpu()))
                losses_ct.append(float(loss_ct.detach().cpu()))
                losses_pa.append(float(loss_pa.detach().cpu()))
                if getattr(model.fusion, "uses_auxiliary", False):
                    losses_aux.append(float(loss_aux.detach().cpu()))
                risks_fused, risks_ct, risks_pa = [], [], []
                times, events = [], []
                auxiliary_payloads = []

        avg_loss = float(np.mean(losses)) if losses else np.nan
        train_cindex, *_ = concordance_index_censored(
            np.asarray(train_events_np, dtype=int).astype(bool),
            np.asarray(train_times_np, dtype=np.float32),
            np.asarray(train_risks_fused_np, dtype=np.float32),
        )
        train_cindex = float(train_cindex)

        model.eval()
        val_risks_np, val_times_np, val_events_np = [], [], []
        with torch.no_grad():
            for batch in val_loader:
                ct, pa, event, time, _ = batch
                risk = model(ct.to(device), pa.to(device))[0]
                val_risks_np.extend(risk.detach().cpu().numpy().reshape(-1).tolist())
                val_times_np.extend(time.detach().cpu().numpy().reshape(-1).tolist())
                val_events_np.extend(event.detach().cpu().numpy().reshape(-1).astype(int).tolist())

        val_risks_arr = np.asarray(val_risks_np, dtype=np.float32)
        val_times_arr = np.asarray(val_times_np, dtype=np.float32)
        val_events_arr = np.asarray(val_events_np, dtype=int)
        val_cindex, *_ = concordance_index_censored(val_events_arr.astype(bool), val_times_arr, val_risks_arr)
        val_cindex = float(val_cindex)
        val_loss = float(cox_loss(
            torch.as_tensor(val_risks_arr, device=device),
            torch.as_tensor(val_times_arr, device=device),
            torch.as_tensor(val_events_arr, device=device),
        ).cpu())
        components = (f"Fused Cox: {np.mean(losses_fused):.4f}, "
                      f"CT Cox: {np.mean(losses_ct):.4f}, PA Cox: {np.mean(losses_pa):.4f}")
        if losses_aux:
            components += f", Auxiliary Loss: {np.mean(losses_aux):.4f}"
        print(
            f"Epoch {epoch}/{args.num_epochs} | Train Loss: {avg_loss:.4f} ({components}) | "
            f"Train C-index: {train_cindex:.4f} | Val Loss: {val_loss:.4f} | "
            f"Val C-index: {val_cindex:.4f}"
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
    print(f"Fold {fold} best C-index: {best_cindex:.4f}")
    return model


def parse_args():
    parser = argparse.ArgumentParser(description="Train PA+CT fusion survival model (5-fold CV).")
    parser.add_argument("--data_dir", default=None,
                        help="Directory containing all_label_roi*.csv with saved CV splits.")
    parser.add_argument("--ct_roi_size", type=int, default=64, choices=[64, 96, 128])
    parser.add_argument("--ct_model", default="resnet18", choices=["resnet10", "resnet18"])
    parser.add_argument("--pa_model", default="abmil",
                        choices=["abmil", "abmil-topk", "abmil_randsample", "gabmil", "gabmil-topk", "meanpool", "transmil"])
    parser.add_argument("--k", type=int, default=4096,
                        help="Patch count for *-topk and abmil_randsample; default 4096; ignored otherwise.")
    parser.add_argument("--checkpoint_root", default=None)
    parser.add_argument("--results_root", default=None)
    parser.add_argument("--ct_pretrained_path", type=str, default=None)
    parser.add_argument("--norm", choices=["none", "layernorm"], default="none")
    parser.add_argument("--fusion_type", default="concat")
    parser.add_argument("--fusion_aux_weight", type=float, default=0.1,
                        help="Auxiliary loss coefficient for shared_private/contrastive/dcca only.")
    parser.add_argument("--num_epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--ct_backbone_lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--lambda_ct", type=float, default=0.0)
    parser.add_argument("--lambda_pa", type=float, default=0.0)
    parser.add_argument("--cox_batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=123,
                        help="Seed for initialization and training randomness.")
    parser.add_argument("--eval_only", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)
    if not np.isfinite(args.fusion_aux_weight) or args.fusion_aux_weight < 0:
        raise ValueError("--fusion_aux_weight must be finite and nonnegative")
    if args.fusion_type in {"contrastive", "dcca"} and args.cox_batch_size < 2:
        raise ValueError("Alignment losses require --cox_batch_size >= 2")
    args.data_dir = resolve_data_dir(args.data_dir, args.ct_roi_size)
    args.fusion_dropout = 0.0
    backbone_lr = args.lr if args.ct_backbone_lr is None else args.ct_backbone_lr

    pretrain_tag = "_pretrain" if args.ct_pretrained_path else ""
    uses_k = args.pa_model.endswith("-topk") or args.pa_model == "abmil_randsample"
    if uses_k and (args.k is None or args.k <= 0):
        raise ValueError("--k must be positive for *-topk and abmil_randsample")
    if not uses_k:
        args.k = None
    k_tag = f"-k{args.k}" if uses_k else ""
    default_suffix = f"pact-{args.pa_model}{k_tag}-{args.ct_model}-roi{args.ct_roi_size}-nodropout-norm{args.norm}{pretrain_tag}-{args.fusion_type}"
    print(f"Using Device: {DEVICE} | ROI: {args.ct_roi_size}")
    print(f"PA: {args.pa_model} | CT: {args.ct_model} | Fusion: {args.fusion_type} | Norm: {args.norm}")
    print(f"LR: {args.lr:g} | CT Backbone LR: {backbone_lr:g}")
    print(f"Loss: Fused Cox + {args.lambda_ct:g}*CT + {args.lambda_pa:g}*PA")
    if args.ct_pretrained_path:
        print(f"CT pretrained: {args.ct_pretrained_path}")
    excluded = []
    if not uses_k:
        excluded.append("k")
    if args.fusion_type not in {"shared_private", "contrastive", "dcca"}:
        excluded.append("fusion_aux_weight")
    save_run_config(args, "pact", default_suffix, excluded)

    dataset = Pa_CT_Dataset(args.data_dir, roi_size=args.ct_roi_size)
    print(f"Loaded {len(dataset)} paired samples")

    model_kwargs = {
        "pa_model_name": args.pa_model,
        "ct_model_name": args.ct_model,
        "ct_pretrained_path": args.ct_pretrained_path,
        "pa_topk": args.k if uses_k else None,
        "fusion_dropout": args.fusion_dropout,
        "fusion_type": args.fusion_type,
        "norm": args.norm,
    }

    fold_splits = cv_splits(dataset.samples)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(fold_splits):
        print(f"\n{'=' * 50}\nFold {fold + 1}/5\n{'=' * 50}")
        seed_everything(args.seed)

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

        model = Pa_CT_Model(**model_kwargs).to(DEVICE)
        if getattr(model.fusion, "is_new_fusion", False):
            print(f"Parameters: total={sum(p.numel() for p in model.parameters())}, "
                  f"fusion={sum(p.numel() for p in model.fusion.parameters())}")
        optimizer = build_pact_optimizer(model, lr=args.lr, weight_decay=args.weight_decay,
                                         ct_backbone_lr=args.ct_backbone_lr)
        checkpoint_dir = Path(args.checkpoint_root) / f"fold_{fold}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        if args.eval_only:
            model.load_state_dict(torch.load(
                checkpoint_dir / "best_model.pth", map_location=DEVICE, weights_only=True
            ))
        else:
            model = train_pact(
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
