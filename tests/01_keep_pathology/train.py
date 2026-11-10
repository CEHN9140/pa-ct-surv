"""Experiment 01: KEEP features with the existing pathology Cox training loop."""

import argparse
import sys
from pathlib import Path

import torch

# Allow direct execution without changing production imports or global settings.
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dataset import build_dataloader, cv_splits
from model.build import Pa_Model
from path_train import train_path
from cox_utils import evaluate_survival
from utils import save_cv_results, save_run_config, seed_everything
from keep_dataset import KeepDataset


def main():
    experiment = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description='Experiment 01: KEEP pathology, 5-fold Cox CV')
    parser.add_argument('--data_dir', default=str(ROOT / 'data/trident/features_uni_v1'))
    parser.add_argument('--feature_dir', default='/home/gly001/cqj/data/lung_cancer/pathology/trident/20x_256px_0px_overlap/features_keep')
    parser.add_argument('--ct_roi_size', type=int, default=64, choices=[64, 96, 128])
    parser.add_argument('--pa_model', default='abmil', choices=[
        'abmil', 'abmil-topk', 'abmil_randsample', 'gabmil', 'gabmil-topk', 'meanpool', 'transmil'])
    parser.add_argument('--k', type=int, default=4096)
    parser.add_argument('--dropout', type=float, default=0.25)
    parser.add_argument('--num_epochs', type=int, default=30)
    parser.add_argument('--cox_batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--eval_only', action='store_true')
    args = parser.parse_args()
    args.survival_loss = 'cox'
    uses_k = args.pa_model.endswith('-topk') or args.pa_model == 'abmil_randsample'
    if not uses_k:
        args.k = None
    elif args.k <= 0:
        parser.error('--k must be positive')
    if args.num_epochs < 1 or args.cox_batch_size < 1:
        parser.error('--num_epochs and --cox_batch_size must be positive')

    seed_everything(args.seed)
    dataset = KeepDataset(args.data_dir, args.feature_dir, args.ct_roi_size)
    args.feature_dim = dataset.feature_dim
    # Preserve the reference cohort and split exactly; never regenerate CV folds.
    data_output = experiment / 'data'
    data_output.mkdir(parents=True, exist_ok=True)
    dataset.samples.to_csv(data_output / f'all_label_roi{args.ct_roi_size}.csv', index=False)
    tag = (f'keep_roi{args.ct_roi_size}_{args.pa_model}_cox_dropout{args.dropout}'
           f'_epochs{args.num_epochs}_coxbs{args.cox_batch_size}'
           f'_lr{args.lr:g}_wd{args.weight_decay:g}_seed{args.seed}')
    if uses_k:
        tag += f'_k{args.k}'
    args.checkpoint_root = str(experiment / 'checkpoints' / tag)
    args.results_root = str(experiment / 'results' / tag)
    save_run_config(args, 'pathology', tag, excluded=() if uses_k else ('k',),
                    extra={'experiment_id': '01', 'feature_encoder': 'keep'})
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Experiment 01 | KEEP dimension: {dataset.feature_dim} | Device: {device}', flush=True)
    fold_results = []
    for fold, (train_indices, val_indices) in enumerate(cv_splits(dataset.samples)):
        print(f'\n{"=" * 50}\nFold {fold + 1}/5\n{"=" * 50}', flush=True)
        seed_everything(args.seed)
        train_loader = build_dataloader(dataset, train_indices, batch_size=1, shuffle=True,
                                        num_workers=args.num_workers)
        val_loader = build_dataloader(dataset, val_indices, batch_size=1,
                                      num_workers=args.num_workers)
        model = Pa_Model(model_name=args.pa_model, feature_dim=dataset.feature_dim,
                         k=args.k, abmil_dropout=args.dropout).to(device)
        directory = Path(args.checkpoint_root) / f'fold_{fold}'
        directory.mkdir(parents=True, exist_ok=True)
        if args.eval_only:
            model.load_state_dict(torch.load(directory / 'best_model.pth',
                                             map_location=device, weights_only=True))
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                         weight_decay=args.weight_decay)
            model = train_path(model, train_loader, val_loader, optimizer, args,
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
