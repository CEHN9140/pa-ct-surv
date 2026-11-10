"""Experiment08: GL pathology ABMIL, native survival times and five-fold CV."""
import argparse
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('MPLCONFIGDIR', '/tmp/pact_matplotlib')
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import torch
import yaml
from lifelines import KaplanMeierFitter
from lifelines.plotting import add_at_risk_counts
from sksurv.metrics import concordance_index_censored

from cox_utils import discrete_nll_loss
from dataset import build_dataloader, cox_batch_sizes, cv_splits
from gl_dataset import GLDataset, file_hash
from loss import survival_loss
from model.build import Pa_Model
from path_train import train_path
from utils import save_cv_results, save_run_config, seed_everything
from visualization.survival_plots import fold_metrics


def parse_args():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data_dir', default=str(here/'data_154_eventtime'))
    parser.add_argument('--output_dir', default=str(here))
    parser.add_argument('--l2_normalize', action='store_true',
                        help='L2-normalize each patch over its feature dimension in both train and validation.')
    parser.add_argument('--survival_loss', choices=['cox', 'nll', 'cox_rank'], default='cox')
    parser.add_argument('--rank_weight', type=float, default=1.)
    parser.add_argument('--dropout', type=float, default=.25)
    parser.add_argument('--num_epochs', type=int, default=30)
    parser.add_argument('--cox_batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--eval_only', action='store_true')
    args = parser.parse_args()
    if args.num_epochs < 1 or not 0 <= args.dropout < 1 or args.num_workers < 0:
        parser.error('Invalid epochs, dropout or num_workers')
    if not np.isfinite(args.lr) or args.lr <= 0 or not np.isfinite(args.weight_decay) or args.weight_decay < 0:
        parser.error('lr must be positive and weight_decay nonnegative')
    if args.survival_loss != 'nll' and args.cox_batch_size < 2:
        parser.error('cox_batch_size must be at least 2')
    if args.survival_loss == 'cox_rank' and (not np.isfinite(args.rank_weight) or args.rank_weight < 0):
        parser.error('rank_weight must be finite and nonnegative')
    return args


def plot_native_km(val, metrics, path):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 7))
    fitters = []
    for group, name, color in [(1, 'High risk', 'crimson'), (0, 'Low risk', 'steelblue')]:
        patients = val[val.risk_group_binary == group]
        if patients.empty:
            continue
        fitter = KaplanMeierFitter().fit(patients.survival_time, patients.event,
                                         label=f'{name} (n={len(patients)})')
        fitter.plot_survival_function(ax=ax, ci_show=True, show_censors=True, color=color)
        fitters.append(fitter)
    ax.set(xlabel='Survival time (original units; unconfirmed)', ylabel='Survival probability',
           ylim=(0, 1.02), title=f"log-rank p = {metrics['logrank_p']:.3g}")
    ax.grid(alpha=.2)
    if fitters:
        add_at_risk_counts(*fitters, ax=ax, rows_to_show=['At risk'])
    fig.subplots_adjust(bottom=.3, top=.86)
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def evaluate_fold(model, train_loader, val_loader, args, device, destination):
    model.eval()
    destination.mkdir(parents=True, exist_ok=True)
    frames, metrics = [], {}
    is_nll = args.survival_loss == 'nll'
    with torch.no_grad():
        for part, loader in [('train', train_loader), ('val', val_loader)]:
            risks, times, events, ids, nll_losses = [], [], [], [], []
            for features, event, time, case in loader:
                output = model(features.to(device))
                risks.append(output[0].detach().cpu().reshape(-1))
                times.append(time.reshape(-1))
                events.append(event.reshape(-1))
                ids.extend(case)
                if is_nll:
                    labels = torch.bucketize(time.to(device), model.time_bins).clamp(max=output[3].shape[1]-1)
                    nll_losses.append(float(discrete_nll_loss(output[3], output[4], labels, event.to(device)))*len(time))
            risk, time, event = torch.cat(risks), torch.cat(times), torch.cat(events)
            loss = sum(nll_losses)/len(time) if is_nll else float(survival_loss(
                risk.to(device), time.to(device), event.to(device), args.survival_loss, args.rank_weight))
            metrics[f'{part}_loss'] = loss
            metrics[f'{part}_cindex'] = float(concordance_index_censored(event.numpy().astype(bool), time.numpy(), risk.numpy())[0])
            frame = pd.DataFrame(dict(case_id=ids, survival_time=time.numpy(), event=event.numpy(), risk_score=risk.numpy()))
            frames.append(frame)
    # Reuse the existing dimensionless CI/HR/log-rank calculations. These temporary
    # internal aliases do not convert time or label this cohort as DFS/months.
    compatible = [f.rename(columns={'survival_time': 'dfs.month', 'event': 'dfs.status'}) for f in frames]
    grouped, extra = fold_metrics(*compatible, eval_times=[])
    frames[1]['risk_group_binary'] = grouped.risk_group_binary.to_numpy()
    for part, frame in zip(['train', 'val'], frames):
        frame.to_csv(destination/f'{part}_predictions.csv', index=False)
    plot_native_km(frames[1], extra, destination/'km_high_vs_low.png')
    notes = {k: v for k, v in extra.items() if k.endswith('_status') and v != 'ok'}
    notes['monthly_auc'] = 'Not calculated: source survival_time unit is unconfirmed'
    (destination/'evaluation_notes.json').write_text(json.dumps(notes, ensure_ascii=False, indent=2)+'\n')
    metrics.update({k: v for k, v in extra.items() if not k.endswith('_status')})
    print('\nExtra survival metrics:')
    for key in ['n_val', 'n_event_val', 'n_high_risk', 'n_low_risk', 'logrank_p',
                'HR_high_vs_low', 'HR_95CI_lower', 'HR_95CI_upper', 'cox_p']:
        value = metrics[key]
        print(f'{key}: {value:.6f}' if isinstance(value, float) else f'{key}: {value}')
    return metrics


def main():
    args = parse_args()
    dataset = GLDataset(args.data_dir, l2_normalize=args.l2_normalize)
    args.data_dir = str(Path(args.data_dir).resolve())
    args.output_dir = str(Path(args.output_dir).resolve())
    args.pa_model, args.feature_dim = 'abmil', dataset.feature_dim
    encoder_tag = dataset.metadata['feature_encoder']+('_l2' if args.l2_normalize else '')
    tag = (f'{dataset.metadata.get("cohort_tag", "gl")}_{encoder_tag}_abmil_{args.survival_loss}'
           f'_dropout{args.dropout}_epochs{args.num_epochs}')
    if args.survival_loss != 'nll':
        tag += f'_coxbs{args.cox_batch_size}'
    if args.survival_loss == 'cox_rank':
        tag += f'_rankw{args.rank_weight:g}'
    tag += f'_lr{args.lr:g}_wd{args.weight_decay:g}_seed{args.seed}'
    args.checkpoint_root = str(Path(args.output_dir)/'checkpoints'/tag)
    args.results_root = str(Path(args.output_dir)/'results'/tag)
    excluded = ['rank_weight'] if args.survival_loss != 'cox_rank' else []
    if args.survival_loss == 'nll':
        excluded.append('cox_batch_size')
    extra = dict(experiment_id='08', feature_encoder=dataset.metadata['feature_encoder'],
                 time_unit='original_unconfirmed', time_conversion='none',
                 endpoint='unspecified_survival', normalization='per_patch_l2' if args.l2_normalize else 'none',
                 cohort_sha256=file_hash(Path(args.data_dir)/'all_label.csv'),
                 feature_signature=dataset.metadata['feature_signature'],
                 labels_sha256=dataset.metadata['labels_sha256'])
    if args.l2_normalize:
        extra['normalization_eps'] = 1e-12
    results = Path(args.results_root)
    config_path = results/'run_config.yaml'
    if config_path.is_file():
        saved = yaml.safe_load(config_path.read_text())
        for key in ['feature_dim', 'pa_model', 'dropout', 'survival_loss', 'seed',
                    'cox_batch_size', 'rank_weight', 'data_dir']:
            if key not in excluded and saved.get(key) != getattr(args, key):
                raise ValueError(f'Existing run has a different setting: {key}')
        for key in ['cohort_sha256', 'feature_signature', 'labels_sha256', 'time_unit']:
            if saved.get(key) != extra[key]:
                raise ValueError(f'Existing run provenance differs: {key}')
        if saved.get('normalization', 'none') != extra['normalization']:
            raise ValueError('Existing run uses a different feature normalization')
        if args.l2_normalize and saved.get('normalization_eps') != extra['normalization_eps']:
            raise ValueError('Existing run uses a different normalization epsilon')
    if args.eval_only:
        if not config_path.is_file() or not (results/'data.csv').is_file():
            raise FileNotFoundError(f'Original training config/data missing: {results}')
        if file_hash(results/'data.csv') != extra['cohort_sha256']:
            raise ValueError('Saved training data differs from the prepared cohort')
        for fold in range(5):
            path = Path(args.checkpoint_root)/f'fold_{fold}'/'best_model.pth'
            if not path.is_file():
                raise FileNotFoundError(f'Missing checkpoint: {path}')
    else:
        save_run_config(args, 'pathology', tag, excluded=excluded, extra=extra)
        (results/'data.csv').write_bytes((Path(args.data_dir)/'all_label.csv').read_bytes())
    seed_everything(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Experiment08 | GL {dataset.feature_dim}-D {args.survival_loss} | {device}', flush=True)
    print(f"Feature normalization: {extra['normalization']} (train and validation)", flush=True)
    print(f'Patients: {len(dataset)} | Events: {int(dataset.samples.event.sum())} | '
          'Time: original values, unit unconfirmed | Monthly AUC disabled', flush=True)
    print(f'Patch count min/median/max: {dataset.samples.n_patches.min()}/'
          f'{dataset.samples.n_patches.median():g}/{dataset.samples.n_patches.max()}', flush=True)
    fold_results = []
    for fold, (train_indices, val_indices) in enumerate(cv_splits(dataset.samples)):
        print(f'\n{"="*50}\nFold {fold+1}/5\n{"="*50}', flush=True)
        seed_everything(args.seed)
        train_loader = build_dataloader(dataset, train_indices, batch_size=1, shuffle=True, num_workers=args.num_workers)
        val_loader = build_dataloader(dataset, val_indices, batch_size=1, num_workers=args.num_workers)
        directory = Path(args.checkpoint_root)/f'fold_{fold}'
        directory.mkdir(parents=True, exist_ok=True)
        time_bins = None
        if args.survival_loss == 'nll':
            if args.eval_only:
                time_bins = torch.load(directory/'best_model.pth', map_location='cpu', weights_only=True)['time_bins']
            else:
                subset = dataset.samples.iloc[train_indices]
                time_bins = np.unique(np.quantile(subset.loc[subset.event == 1, 'time'], [.25, .5, .75]).astype(np.float32))
            print(f'NLL cutpoints (original units): {list(map(float, time_bins))}', flush=True)
        else:
            print(f'Cox risk-set sizes (tail merged): {cox_batch_sizes(len(train_loader), args.cox_batch_size)}', flush=True)
        model = Pa_Model(model_name='abmil', feature_dim=args.feature_dim, abmil_dropout=args.dropout,
                         time_bins=time_bins).to(device)
        if args.eval_only:
            model.load_state_dict(torch.load(directory/'best_model.pth', map_location=device, weights_only=True))
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            model = train_path(model, train_loader, val_loader, optimizer, args, device, fold, directory)
        metrics = evaluate_fold(model, train_loader, val_loader, args, device, results/f'fold_{fold}')
        print(f"Fold {fold} | Train Loss: {metrics['train_loss']:.4f} | Train C-index: {metrics['train_cindex']:.4f} | "
              f"Val Loss: {metrics['val_loss']:.4f} | Val C-index: {metrics['val_cindex']:.4f}")
        fold_results.append(dict(fold=fold, **metrics))
    save_cv_results(fold_results, results)


if __name__ == '__main__':
    main()
