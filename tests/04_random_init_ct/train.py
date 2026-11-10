"""Experiment 04: CT survival training without pretrained weights."""

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from cox_utils import evaluate_survival
from ct_train import train_ct
from dataset import CT_Dataset, build_dataloader, cv_splits
from model.build import CT_Model
from utils import save_cv_results, save_run_config, seed_everything


def main():
    experiment = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description='Experiment 04: randomly initialized CT, 5-fold Cox CV')
    parser.add_argument('--data_dir', default=str(ROOT / 'data/trident/features_uni_v1'))
    parser.add_argument('--ct_roi_size', type=int, default=96, choices=[64, 96, 128])
    parser.add_argument('--ct_model', default='resnet18', choices=['resnet10', 'resnet18'])
    parser.add_argument('--dropout', type=float, default=0.25)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--freeze_bn_stats', action='store_true',
                        help='Keep initial BN running statistics fixed, as in the main CT workflow.')
    parser.add_argument('--eval_only', action='store_true')
    args = parser.parse_args()
    if args.num_epochs < 1 or args.batch_size < 1:
        parser.error('--num_epochs and --batch_size must be positive')
    args.ct_pretrained_path = None
    args.freeze_backbone = False
    seed_everything(args.seed)
    dataset = CT_Dataset(args.data_dir, roi_size=args.ct_roi_size)
    bn_tag = 'bnfreeze' if args.freeze_bn_stats else 'bntrain'
    tag = (f'roi{args.ct_roi_size}_{args.ct_model}_randominit_{bn_tag}_dropout{args.dropout}'
           f'_epochs{args.num_epochs}_bs{args.batch_size}_lr{args.lr:g}'
           f'_wd{args.weight_decay:g}_seed{args.seed}')
    args.checkpoint_root = str(experiment / 'checkpoints' / tag)
    args.results_root = str(experiment / 'results' / tag)
    save_run_config(args, 'ct', tag,
                    extra={'experiment_id': '04', 'initialization': 'random'})
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Experiment 04 | {args.ct_model} | Pretrained: None | '
          f'Freeze backbone: False | Freeze BN statistics: {args.freeze_bn_stats}', flush=True)
    print(f'Device: {device} | ROI: {args.ct_roi_size} | Loaded {len(dataset)} samples | '
          f'LR (all parameters): {args.lr:g} | Loss: Cox PH', flush=True)
    fold_results = []
    for fold, (train_indices, val_indices) in enumerate(cv_splits(dataset.samples)):
        print(f'\n{"=" * 50}\nFold {fold + 1}/5\n{"=" * 50}', flush=True)
        seed_everything(args.seed)
        train_loader = build_dataloader(dataset, train_indices, batch_size=args.batch_size,
                                        shuffle=True, num_workers=args.num_workers, merge_tail=True)
        val_loader = build_dataloader(dataset, val_indices, batch_size=args.batch_size,
                                      num_workers=args.num_workers)
        model = CT_Model(model_name=args.ct_model, pretrained_path=None,
                         freeze_backbone=False, dropout=args.dropout).to(device)
        model.ct.freeze_bn_stats = args.freeze_bn_stats
        directory = Path(args.checkpoint_root) / f'fold_{fold}'
        directory.mkdir(parents=True, exist_ok=True)
        if args.eval_only:
            model.load_state_dict(torch.load(directory / 'best_model.pth',
                                             map_location=device, weights_only=True))
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                         weight_decay=args.weight_decay)
            model = train_ct(model, train_loader, val_loader, optimizer, args, device, fold, directory)
        metrics = evaluate_survival(model, train_loader, val_loader, device,
                                   save_dir=Path(args.results_root) / f'fold_{fold}')
        print(f"Fold {fold} | Train Loss: {metrics['train_loss']:.4f} | "
              f"Train C-index: {metrics['train_cindex']:.4f} | "
              f"Val Loss: {metrics['val_loss']:.4f} | Val C-index: {metrics['val_cindex']:.4f}")
        fold_results.append({'fold': fold, **metrics})
    save_cv_results(fold_results, args.results_root)


if __name__ == '__main__':
    main()
