"""Experiment 05: EAGLE top25, ABMIL/MeanPool and real patient batches."""

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from cox_utils import evaluate_survival
from dataset import build_dataloader, cv_splits, cox_batch_sizes
from eagle_dataset import EagleDataset
from model.build import Pa_Model
from path_train import train_path
from utils import save_cv_results, save_run_config, seed_everything


def main():
    experiment = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description='Experiment 05: EAGLE top25, 5-fold Cox CV')
    parser.add_argument('--data_dir', default=str(ROOT / 'data/eagle/top25'))
    parser.add_argument('--pa_model', choices=['abmil', 'meanpool'], default='abmil')
    parser.add_argument('--ct_roi_size', type=int, default=64, choices=[64, 96, 128])
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.25)
    parser.add_argument('--num_epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--eval_only', action='store_true')
    args = parser.parse_args()
    if args.batch_size < 2 or args.num_epochs < 1:
        parser.error('--batch_size must be at least 2 and --num_epochs must be positive')
    if not 0 <= args.dropout < 1:
        parser.error('--dropout must be in [0, 1)')

    seed_everything(args.seed)
    dataset = EagleDataset(args.data_dir, args.ct_roi_size)
    args.feature_dim = dataset.feature_dim
    args.patch_count = dataset.patch_count
    args.survival_loss = 'cox'
    tag = (f'eagle_top25_roi{args.ct_roi_size}_{args.pa_model}_cox_dropout{args.dropout}'
           f'_epochs{args.num_epochs}_bs{args.batch_size}'
           f'_lr{args.lr:g}_wd{args.weight_decay:g}_seed{args.seed}')
    args.checkpoint_root = str(experiment / 'checkpoints' / tag)
    args.results_root = str(experiment / 'results' / tag)
    save_run_config(args, 'pathology', tag,
                    extra={'experiment_id': '05', 'feature_encoder': 'Virchow2-CLS',
                           'batch_mode': 'patient_batch', 'merge_tail': True})
    # Snapshot the exact cohort/splits used by this run, without regenerating them.
    dataset.samples.to_csv(Path(args.results_root) / 'data.csv', index=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Experiment 05 | Model: {args.pa_model} | Features: [25, 1280] | Device: {device}', flush=True)
    print(f'Patient batch size: {args.batch_size} | No cross-batch gradient accumulation', flush=True)

    # train_path counts loader iterations. A group of ONE already-batched
    # iteration means one forward, joint batch Cox loss and optimizer step.
    # This internal setting is not the patient batch size saved in run_config.
    train_args = argparse.Namespace(**vars(args))
    train_args.cox_batch_size = 1
    fold_results = []
    for fold, (train_indices, val_indices) in enumerate(cv_splits(dataset.samples)):
        print(f'\n{"=" * 50}\nFold {fold + 1}/5\n{"=" * 50}', flush=True)
        seed_everything(args.seed)
        train_loader = build_dataloader(
            dataset, train_indices, batch_size=args.batch_size, shuffle=True,
            merge_tail=True, num_workers=args.num_workers)
        val_loader = build_dataloader(
            dataset, val_indices, batch_size=args.batch_size,
            num_workers=args.num_workers)
        print(f'Training batch sizes: {cox_batch_sizes(len(train_indices), args.batch_size)}')
        model = Pa_Model(model_name=args.pa_model, feature_dim=dataset.feature_dim,
                         abmil_dropout=args.dropout).to(device)
        directory = Path(args.checkpoint_root) / f'fold_{fold}'
        directory.mkdir(parents=True, exist_ok=True)
        if args.eval_only:
            model.load_state_dict(torch.load(directory / 'best_model.pth',
                                             map_location=device, weights_only=True))
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                         weight_decay=args.weight_decay)
            model = train_path(model, train_loader, val_loader, optimizer, train_args,
                               device, fold, directory)
        metrics = evaluate_survival(model, train_loader, val_loader, device,
                                   save_dir=Path(args.results_root) / f'fold_{fold}')
        print(f"Fold {fold} | Train Loss: {metrics['train_loss']:.4f} | "
              f"Train C-index: {metrics['train_cindex']:.4f} | "
              f"Val Loss: {metrics['val_loss']:.4f} | Val C-index: {metrics['val_cindex']:.4f}")
        fold_results.append({'fold': fold, **metrics})
    save_cv_results(fold_results, args.results_root)


if __name__ == '__main__':
    main()
