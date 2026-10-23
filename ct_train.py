import argparse
from pathlib import Path

import numpy as np
import torch
from sksurv.metrics import concordance_index_censored

from cox_utils import cox_loss, evaluate_survival
from dataset import CT_Dataset, build_dataloader, resolve_data_dir, cv_splits
from utils import seed_everything, save_run_config, save_cv_results, build_ct_optimizer
from model.build import CT_Model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_ct(
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
    wait = 0

    for epoch in range(1, args.num_epochs + 1):
        model.train()
        train_losses = []
        train_risks_np, train_times_np, train_events_np = [], [], []
        for batch in train_loader:
            ct, event, time, _ = batch
            risk = model(ct.to(device, non_blocking=True))
            train_risks_np.extend(risk.detach().cpu().numpy().reshape(-1).tolist())
            train_times_np.extend(time.detach().cpu().numpy().reshape(-1).tolist())
            train_events_np.extend(
                event.detach().cpu().numpy().reshape(-1).astype(int).tolist()
            )
            loss = cox_loss(risk, time.to(device), event.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))

        avg_loss = float(np.mean(train_losses)) if train_losses else np.nan
        train_cindex, *_ = concordance_index_censored(
            np.asarray(train_events_np, dtype=int).astype(bool),
            np.asarray(train_times_np, dtype=np.float32),
            np.asarray(train_risks_np, dtype=np.float32),
        )
        train_cindex = float(train_cindex)

        model.eval()
        val_risks_np, val_times_np, val_events_np = [], [], []
        with torch.no_grad():
            for batch in val_loader:
                ct, event, time, _ = batch
                risk = model(ct.to(device, non_blocking=True))
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
            torch.save(best_state, checkpoint_dir / "best_model.pth")
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
    parser = argparse.ArgumentParser(description="Train CT survival model (5-fold CV).")
    parser.add_argument("--data_dir", default=None,
                        help="Directory containing all_label_roi*.csv with saved CV splits.")
    parser.add_argument("--ct_roi_size", type=int, default=96, choices=[64, 96, 128])
    parser.add_argument(
        "--ct_model", default="resnet18", choices=["resnet10", "resnet18"]
    )
    parser.add_argument("--ct_pretrained_path", type=str, default=None)
    parser.add_argument("--freeze_backbone", action="store_true")
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--checkpoint_root", default=None)
    parser.add_argument("--results_root", default=None)
    parser.add_argument("--num_epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--ct_backbone_lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--batch_size", type=int, default=64)
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

    pretrain_tag = "_pretrain" if args.ct_pretrained_path else ""
    backbone_lr = args.lr if args.ct_backbone_lr is None else args.ct_backbone_lr
    default_suffix = (
        f"ct-{args.ct_model}-roi{args.ct_roi_size}{pretrain_tag}"
        f"-bs{args.batch_size}-lr{args.lr:g}-blr{backbone_lr:g}"
        f"-wd{args.weight_decay:g}-bnfreeze-seed{args.seed}"
    )
    print(f"Using Device: {DEVICE} | ROI: {args.ct_roi_size}")
    print(f"Model: {args.ct_model} | CT augmentation: disabled")
    print(
        f"LR: {args.lr:g} | Backbone LR: {backbone_lr:g} | Loss: Cox PH"
    )
    if args.ct_pretrained_path:
        print(f"Pretrained: {args.ct_pretrained_path}")

    save_run_config(args, "ct", default_suffix)

    dataset = CT_Dataset(args.data_dir, roi_size=args.ct_roi_size)
    print(f"Loaded {len(dataset)} samples")

    model_kwargs = {
        "model_name": args.ct_model,
        "pretrained_path": args.ct_pretrained_path,
        "freeze_backbone": args.freeze_backbone,
        "dropout": args.dropout,
    }

    fold_splits = cv_splits(dataset.samples)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(fold_splits):
        print(f"\n{'=' * 50}\nFold {fold + 1}/5\n{'=' * 50}")
        seed_everything(args.seed)

        train_loader = build_dataloader(
            dataset, indices=train_idx, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, merge_tail=True, pin_memory=True,
        )
        val_loader = build_dataloader(
            dataset, indices=val_idx, batch_size=args.batch_size,
            num_workers=args.num_workers, pin_memory=True,
        )

        model = CT_Model(**model_kwargs).to(DEVICE)
        optimizer = build_ct_optimizer(
            model,
            lr=args.lr,
            weight_decay=args.weight_decay,
            ct_backbone_lr=args.ct_backbone_lr,
        )
        checkpoint_dir = Path(args.checkpoint_root) / f"fold_{fold}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        if args.eval_only:
            ckpt_path = checkpoint_dir / "best_model.pth"
            model.load_state_dict(
                torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
            )
        else:
            model = train_ct(
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
