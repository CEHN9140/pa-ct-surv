"""Experiment06: UNI L2 / KEEP / EAGLE ABMIL with NLL or Cox plus ranking."""

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
for directory in ['01_keep_pathology', '02_uni_l2_pathology', '05_eagle_top25_pathology']:
    sys.path.append(str(ROOT / 'tests' / directory))

from cox_utils import evaluate_survival
from dataset import build_dataloader, cv_splits, read_samples
from eagle_dataset import EagleDataset
from keep_dataset import KeepDataset
from model.build import Pa_Model
from path_train import train_path
from uni_l2_dataset import UniL2Dataset
from utils import save_cv_results, save_run_config, seed_everything


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--feature_source', choices=['uni_l2', 'keep', 'eagle'], default='uni_l2')
    parser.add_argument('--survival_loss', choices=['nll', 'cox_rank'], default='nll')
    parser.add_argument('--data_dir', default=None, help='Feature CSV directory; chosen automatically by source.')
    parser.add_argument('--reference_data_dir', default=str(ROOT / 'data/trident/features_uni_v1'),
                        help='Reference patient labels and existing five-fold assignments.')
    parser.add_argument('--keep_feature_dir', default='/home/gly001/cqj/data/lung_cancer/pathology/trident/20x_256px_0px_overlap/features_keep')
    parser.add_argument('--ct_roi_size', type=int, default=64, choices=[64, 96, 128])
    parser.add_argument('--batch_size', type=int, default=64, help='Real patient batch size for EAGLE.')
    parser.add_argument('--cox_batch_size', type=int, default=64, help='Joint risk-set size for variable-length bags.')
    parser.add_argument('--rank_weight', type=float, default=1.0)
    parser.add_argument('--dropout', type=float, default=0.25)
    parser.add_argument('--num_epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--run_suffix', default=None, help='Optional label for a separate repeat run.')
    parser.add_argument('--eval_only', action='store_true')
    args = parser.parse_args()
    if args.num_epochs < 1 or not 0 <= args.dropout < 1:
        parser.error('num_epochs must be positive and dropout must be in [0, 1)')
    if args.feature_source == 'eagle' and args.batch_size < 2:
        parser.error('EAGLE batch_size must be at least 2')
    if args.survival_loss == 'cox_rank':
        if not np.isfinite(args.rank_weight) or args.rank_weight < 0:
            parser.error('rank_weight must be finite and nonnegative')
        if args.feature_source != 'eagle' and args.cox_batch_size < 2:
            parser.error('cox_batch_size must be at least 2')
    return args


def make_dataset(args):
    args.reference_data_dir = str(Path(args.reference_data_dir).resolve())
    if args.data_dir is None:
        args.data_dir = str(ROOT / 'data/eagle/top25') if args.feature_source == 'eagle' else args.reference_data_dir
    args.data_dir = str(Path(args.data_dir).resolve())
    if args.feature_source == 'uni_l2':
        dataset = UniL2Dataset(args.data_dir, args.ct_roi_size)
    elif args.feature_source == 'keep':
        args.keep_feature_dir = str(Path(args.keep_feature_dir).resolve())
        dataset = KeepDataset(args.data_dir, args.keep_feature_dir, args.ct_roi_size)
    else:
        dataset = EagleDataset(args.data_dir, args.ct_roi_size)
    reference = read_samples(args.reference_data_dir, args.ct_roi_size,
                             ['pa_id', 'ct_id', 'event', 'time', 'split']).set_index('pa_id')
    selected = dataset.samples.set_index('pa_id')
    if not reference.index.is_unique or not selected.index.is_unique or set(reference.index) != set(selected.index):
        raise ValueError('Feature and reference tables must contain the same unique pathology patients')
    selected = selected.loc[reference.index]
    try:
        columns = ['ct_id', 'event', 'time', 'split']
        pd.testing.assert_frame_equal(reference[columns], selected[columns], check_dtype=False, check_exact=True)
    except AssertionError as exc:
        raise ValueError('CT IDs, outcomes or five-fold assignments differ from the reference') from exc
    dataset.samples = selected.reset_index()
    args.feature_dim = getattr(dataset, 'feature_dim', 1024)
    return dataset


def main():
    args = parse_args()
    experiment = Path(__file__).resolve().parent
    seed_everything(args.seed)
    dataset = make_dataset(args)
    args.pa_model = 'abmil'
    is_eagle = args.feature_source == 'eagle'
    tag = (f'{args.feature_source}_roi{args.ct_roi_size}_abmil_{args.survival_loss}'
           f'_dropout{args.dropout}_epochs{args.num_epochs}')
    if is_eagle:
        tag += f'_bs{args.batch_size}'
    elif args.survival_loss == 'cox_rank':
        tag += f'_coxbs{args.cox_batch_size}'
    if args.survival_loss == 'cox_rank':
        tag += f'_rankw{args.rank_weight:g}'
    tag += f'_lr{args.lr:g}_wd{args.weight_decay:g}_seed{args.seed}'
    if args.run_suffix:
        tag += f'_{args.run_suffix}'
    args.checkpoint_root = str(experiment / 'checkpoints' / tag)
    args.results_root = str(experiment / 'results' / tag)
    excluded = []
    if args.run_suffix is None:
        excluded.append('run_suffix')
    if args.feature_source != 'keep':
        excluded.append('keep_feature_dir')
    if not is_eagle:
        excluded.append('batch_size')
    if is_eagle or args.survival_loss == 'nll':
        excluded.append('cox_batch_size')
    if args.survival_loss != 'cox_rank':
        excluded.append('rank_weight')
    extra = {'experiment_id': '06', 'batch_mode': 'patient_batch' if is_eagle else 'individual_bag',
             'feature_encoder': {'uni_l2': 'uni_v1', 'keep': 'keep', 'eagle': 'Virchow2-CLS'}[args.feature_source],
             'normalization': {'uni_l2': 'per_patch_l2', 'keep': 'encoder_native', 'eagle': 'none'}[args.feature_source]}
    if args.feature_source == 'uni_l2':
        extra['normalization_eps'] = 1e-12
    if is_eagle:
        extra['patch_count'] = dataset.patch_count
    results = Path(args.results_root)
    if args.run_suffix and not args.eval_only and results.exists():
        raise FileExistsError(f'Repeat run already exists: {results}')
    if args.eval_only:
        config_path, data_path = results / 'run_config.yaml', results / 'data.csv'
        if not config_path.is_file() or not data_path.is_file():
            raise FileNotFoundError(f'Missing original training config or data snapshot: {results}')
        for fold in range(5):
            checkpoint = Path(args.checkpoint_root) / f'fold_{fold}' / 'best_model.pth'
            if not checkpoint.is_file():
                raise FileNotFoundError(f'Missing checkpoint: {checkpoint}')
        original = yaml.safe_load(config_path.read_text())
        for key in ['feature_source', 'survival_loss', 'feature_dim', 'pa_model', 'dropout',
                    'data_dir', 'reference_data_dir', 'keep_feature_dir', 'ct_roi_size',
                    'rank_weight', 'batch_size', 'cox_batch_size', 'seed']:
            if key not in excluded and original.get(key) != getattr(args, key):
                raise ValueError(f'eval_only setting differs from training: {key}')
        snapshot = pd.read_csv(data_path, dtype={'pa_id': str, 'ct_id': str}, float_precision='round_trip')
        try:
            pd.testing.assert_frame_equal(snapshot, dataset.samples, check_dtype=False, check_exact=True)
        except AssertionError as exc:
            raise ValueError('eval_only data differs from the saved training snapshot') from exc
        print(f'Using saved training config and data: {results}', flush=True)
    else:
        save_run_config(args, 'pathology', tag, excluded, extra)
        dataset.samples.to_csv(results / 'data.csv', index=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Experiment 06 | {args.feature_source}: {args.feature_dim}-D | {args.survival_loss} | {device}', flush=True)
    print(f'Patients: {len(dataset)} | Existing reference folds and outcomes verified', flush=True)
    if args.survival_loss == 'cox_rank':
        print(f'Ranking weight: {args.rank_weight:g}', flush=True)
    train_args = argparse.Namespace(**vars(args))
    # One EAGLE loader iteration already contains a full patient batch.
    if is_eagle:
        train_args.cox_batch_size = 1
    fold_results = []
    for fold, (train_indices, val_indices) in enumerate(cv_splits(dataset.samples)):
        print(f'\n{"=" * 50}\nFold {fold + 1}/5\n{"=" * 50}', flush=True)
        seed_everything(args.seed)
        train_loader = build_dataloader(
            dataset, train_indices, batch_size=args.batch_size if is_eagle else 1,
            shuffle=True, merge_tail=is_eagle, num_workers=args.num_workers)
        # train_path averages batch NLL values; one patient per validation batch
        # gives the exact patient mean, including an incomplete EAGLE tail.
        val_loader = build_dataloader(
            dataset, val_indices, batch_size=args.batch_size if is_eagle and args.survival_loss != 'nll' else 1,
            num_workers=args.num_workers)
        directory = Path(args.checkpoint_root) / f'fold_{fold}'
        directory.mkdir(parents=True, exist_ok=True)
        time_bins = None
        if args.survival_loss == 'nll':
            if args.eval_only:
                state = torch.load(directory / 'best_model.pth', map_location='cpu', weights_only=True)
                time_bins = state['time_bins']
            else:
                subset = dataset.samples.iloc[train_indices]
                event_times = subset.loc[subset.event == 1, 'time'].to_numpy()
                if not len(event_times):
                    raise ValueError(f'Fold {fold} has no training events for NLL cutpoints')
                time_bins = np.unique(np.quantile(event_times, [.25, .5, .75]).astype(np.float32))
            print(f'NLL cutpoints: {list(map(float, time_bins))}', flush=True)
        model = Pa_Model(model_name='abmil', feature_dim=args.feature_dim,
                         abmil_dropout=args.dropout, time_bins=time_bins).to(device)
        if args.eval_only:
            model.load_state_dict(torch.load(directory / 'best_model.pth', map_location=device, weights_only=True))
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            model = train_path(model, train_loader, val_loader, optimizer, train_args, device, fold, directory)
        metrics = evaluate_survival(model, train_loader, val_loader, device,
                                   save_dir=Path(args.results_root) / f'fold_{fold}',
                                   loss_type=args.survival_loss, rank_weight=args.rank_weight)
        print(f"Fold {fold} | Train Loss: {metrics['train_loss']:.4f} | "
              f"Train C-index: {metrics['train_cindex']:.4f} | "
              f"Val Loss: {metrics['val_loss']:.4f} | Val C-index: {metrics['val_cindex']:.4f}")
        fold_results.append({'fold': fold, **metrics})
    save_cv_results(fold_results, args.results_root)


if __name__ == '__main__':
    main()
