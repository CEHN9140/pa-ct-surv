"""Within-validation-fold CT and query-only shuffles of fixed test15 models."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import torch
import yaml
from scipy.stats import spearmanr
from sksurv.metrics import concordance_index_censored, cumulative_dynamic_auc
from sksurv.util import Surv

from dataset import Pa_CT_Dataset, build_dataloader, cv_fold_indices
from all_query_dataset import AllPatchQueryDataset
from path_query_ct import PathQueryCTModel
from utils import seed_everything
from visualization.survival_plots import read_predictions


def digest(path):
    sha = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            sha.update(chunk)
    return sha.hexdigest()


def patient_permutations(n, repeats, seed, fold):
    """Uniform random permutations conditional on no recipient being its donor."""
    if n < 2 or repeats < 1:
        raise ValueError('Need at least two patients and one repeat')
    rng = np.random.default_rng(np.random.SeedSequence([seed, fold]))
    orders = []
    for _ in range(repeats):
        order = rng.permutation(n)
        while np.any(order == np.arange(n)):
            order = rng.permutation(n)
        orders.append(order)
    return np.stack(orders)


@torch.inference_mode()
def cache_fusion_inputs(model, loader, device):
    """Capture existing fusion inputs, without reimplementing the encoders."""
    model.eval()
    captured = []
    handle = model.fusion.register_forward_pre_hook(
        lambda module, inputs: captured.append(tuple(x.detach().cpu() for x in inputs)))
    risks, case_ids = [], []
    try:
        for step, (ct, pa, _, _, ids) in enumerate(loader, 1):
            risks.append(model(ct.to(device), pa.to(device))[0].detach().cpu().reshape(-1))
            case_ids.extend(ids)
            if step % 25 == 0 or step == len(loader):
                print(f'  Encoded {step}/{len(loader)} validation batches', flush=True)
    finally:
        handle.remove()
    if len(captured) != len(loader):
        raise ValueError('Expected exactly one fusion call per model forward')
    features = {}
    for i, key in enumerate(['pa_patches', 'ct_tokens', 'pa_patient']):
        values = [batch[i] for batch in captured]
        if key == 'pa_patches' and any(x.shape[1:] != values[0].shape[1:] for x in values):
            features[key] = [sample for value in values for sample in value]
        else:
            features[key] = torch.cat(values)
    features.update(risk=torch.cat(risks), case_id=case_ids)
    return features


@torch.inference_mode()
def permuted_risks(fusion, features, ct_order, query_order, device, batch_size):
    """Only query patches/CT tokens move; direct patient pathology stays fixed."""
    risks = []
    if isinstance(features['pa_patches'], list):
        # Variable-length bags stay separate; no padding participates in attention.
        for i, (ct_idx, query_idx) in enumerate(zip(ct_order, query_order)):
            patches = features['pa_patches'][int(query_idx)][None].to(device)
            ct = features['ct_tokens'][int(ct_idx):int(ct_idx)+1].to(device)
            patient = features['pa_patient'][i:i+1].to(device)
            risks.append(fusion(patches, ct, patient)[0].cpu().reshape(-1))
        return torch.cat(risks).numpy()
    for start in range(0, len(ct_order), batch_size):
        end = start + batch_size
        patches = features['pa_patches'][query_order[start:end]].to(device)
        ct = features['ct_tokens'][ct_order[start:end]].to(device)
        patient = features['pa_patient'][start:end].to(device)
        risks.append(fusion(patches, ct, patient)[0].cpu().reshape(-1))
    return torch.cat(risks).numpy()


def score_risks(train, val, risk, baseline):
    result = {'cindex': float(concordance_index_censored(
        val['dfs.status'].to_numpy().astype(bool), val['dfs.month'].to_numpy(), risk)[0])}
    y_train = Surv.from_arrays(train['dfs.status'].astype(bool), train['dfs.month'])
    y_val = Surv.from_arrays(val['dfs.status'].astype(bool), val['dfs.month'])
    for month in [36., 60.]:
        result[f'auc_{month:g}m'] = float(cumulative_dynamic_auc(
            y_train, y_val, risk, [month])[0][0])
    difference = risk.astype(np.float64) - baseline.astype(np.float64)
    result['risk_rmse'] = float(np.sqrt(np.mean(difference ** 2)))
    baseline_sd = float(np.std(baseline))
    result['risk_rmse_over_baseline_sd'] = result['risk_rmse'] / baseline_sd if baseline_sd else np.nan
    result['risk_spearman'] = (float(spearmanr(risk, baseline).statistic)
                               if np.ptp(risk) > 0 and np.ptp(baseline) > 0 else np.nan)
    return result


def summarize(rows, destination):
    frame = pd.DataFrame(rows)
    frame.to_csv(destination / 'repeat_metrics.csv', index=False)
    numeric = [c for c in frame.select_dtypes(include=np.number) if c not in ['fold', 'repeat']]
    folds = frame.groupby(['fold', 'condition'])[numeric].mean().reset_index()
    folds.to_csv(destination / 'fold_metrics.csv', index=False)
    summary = folds.groupby('condition')[numeric].agg(['mean', 'std'])
    summary.columns = ['_'.join(c) for c in summary.columns]
    summary['n_folds'] = folds.groupby('condition').size()
    summary.to_csv(destination / 'summary.csv')
    print(summary[['cindex_mean', 'cindex_std', 'drop_cindex_mean']].to_string(), flush=True)


def run_fold(run, config, dataset, fold, args, destination, device):
    destination.mkdir(parents=True)
    train_indices, val_indices = cv_fold_indices(dataset.samples, fold)
    train = read_predictions(run / f'fold_{fold}/train_predictions.csv').sort_values('case_id').reset_index(drop=True)
    val = read_predictions(run / f'fold_{fold}/val_predictions.csv').sort_values('case_id').reset_index(drop=True)
    for indices, reference in [(train_indices, train), (val_indices, val)]:
        rows = dataset.samples.iloc[indices].copy()
        rows['case_id'] = rows.pa_id + '|' + rows.ct_id
        rows = rows.sort_values('case_id')
        np.testing.assert_array_equal(reference.case_id, rows.case_id)
        np.testing.assert_array_equal(reference['dfs.status'], rows.event)
        np.testing.assert_allclose(reference['dfs.month'], rows.time.astype(np.float32), rtol=0, atol=1e-5)
    if set(train.case_id) & set(val.case_id):
        raise ValueError('Training/validation overlap')
    checkpoint = Path(config['checkpoint_root']) / f'fold_{fold}/best_model.pth'
    checkpoint_sha = digest(checkpoint)
    model = PathQueryCTModel(dropout=config['dropout'],
                             pa_l2_normalize=config.get('pa_l2_normalize', True),
                             all_patch_queries=config.get('all_patch_queries', False),
                             qk_norm=config.get('qk_norm', False),
                             ct_zscore=config.get('ct_zscore', False),
                             ct_bn_adapt=config.get('ct_bn_adapt', False),
                             fusion_control=config.get('fusion_control', 'crossattention')).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True), strict=True)
    model.eval().requires_grad_(False)
    loader = build_dataloader(dataset, indices=val_indices, num_workers=args.num_workers,
                              pin_memory=device.type == 'cuda')
    features = cache_fusion_inputs(model, loader, device)
    order = np.argsort(features['case_id'])
    ids = np.asarray(features.pop('case_id'))[order]
    np.testing.assert_array_equal(ids, val.case_id)
    features = {key: [value[int(i)] for i in order] if isinstance(value, list) else value[order]
                for key, value in features.items()}
    ct_count = 0 if config.get('fusion_control') == 'pathology_only' else 512
    if tuple(features['ct_tokens'].shape[1:]) != (ct_count, 512):
        raise ValueError('Expected ROI64 layer4 CT tokens [N,512,512]')
    for key, value in features.items():
        if any(not torch.isfinite(x).all() for x in (value if isinstance(value, list) else [value])):
            raise ValueError(f'Nonfinite {key}')
    full_risk = features['risk'].numpy()
    np.testing.assert_allclose(full_risk, val.risk_score, rtol=1e-4, atol=2e-5,
                               err_msg='Full-model risks differ from saved validation predictions')
    identity = np.arange(len(val))
    baseline = permuted_risks(model.fusion, features, identity, identity, device, args.batch_size)
    np.testing.assert_allclose(baseline, full_risk, rtol=1e-4, atol=2e-5,
                               err_msg='Cached fusion differs from full model')
    normal = score_risks(train, val, baseline, baseline)
    normal.update(drop_cindex=0., drop_auc_36m=0., drop_auc_60m=0.)
    records = [dict(fold=fold, condition='paired', repeat=-1, **normal)]
    val.assign(recomputed_risk=baseline).to_csv(destination / 'baseline_predictions.csv', index=False)
    orders = patient_permutations(len(val), args.repeats, args.seed, fold)
    risk_arrays = {}
    print(f'  Fold {fold} paired C-index={normal["cindex"]:.4f}; baseline parity passed', flush=True)
    for condition in ['shuffle_ct', 'shuffle_query']:
        risks = []
        for repeat, donors in enumerate(orders):
            ct_order, query_order = (donors, identity) if condition == 'shuffle_ct' else (identity, donors)
            risk = permuted_risks(model.fusion, features, ct_order, query_order, device, args.batch_size)
            if not np.isfinite(risk).all():
                raise ValueError(f'Nonfinite risks: {condition}, repeat {repeat}')
            metrics = score_risks(train, val, risk, baseline)
            for metric in ['cindex', 'auc_36m', 'auc_60m']:
                metrics['drop_' + metric] = normal[metric] - metrics[metric]
            records.append(dict(fold=fold, condition=condition, repeat=repeat, **metrics))
            risks.append(risk)
            if (repeat + 1) % 20 == 0 or repeat + 1 == args.repeats:
                print(f'  Fold {fold} {condition}: {repeat + 1}/{args.repeats}', flush=True)
        risk_arrays[condition] = np.stack(risks)
    if digest(checkpoint) != checkpoint_sha:
        raise ValueError('Source checkpoint changed during inference')
    np.savez_compressed(destination / 'permutations_and_risks.npz', case_id=ids, baseline=baseline,
                        donor_indices=orders, **risk_arrays)
    (destination / 'provenance.json').write_text(json.dumps(dict(
        checkpoint=str(checkpoint), checkpoint_sha256=checkpoint_sha,
        run_config_sha256=digest(run / 'run_config.yaml'),
        cohort_sha256=digest(Path(config['data_dir']) / 'all_label_roi64.csv'),
        val_predictions_sha256=digest(run / f'fold_{fold}/val_predictions.csv'),
        implementation_sha256=digest(__file__),
        full_vs_saved_max_abs_error=float(np.max(np.abs(full_risk - val.risk_score.to_numpy()))),
        cached_vs_full_max_abs_error=float(np.max(np.abs(baseline - full_risk)))), indent=2) + '\n')
    pd.DataFrame(records).to_csv(destination / 'repeat_metrics.csv', index=False)
    return records


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run_dirs', nargs='+', default=None)
    parser.add_argument('--output_dir', default=str(here / 'inference_checks'))
    parser.add_argument('--repeats', type=int, default=100)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    args = parser.parse_args()
    if args.repeats < 1 or args.batch_size < 1 or args.num_workers < 0 or args.seed < 0:
        parser.error('Invalid repeats, batch size, workers or seed')
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    device = torch.device(args.device)
    seed_everything(args.seed)
    runs = [Path(p).resolve() for p in args.run_dirs] if args.run_dirs else [
        p.parent for p in sorted((here / 'results').glob('*/run_config.yaml'))
        if yaml.safe_load(p.read_text()).get('fusion_type') == 'path_query_ct_layer4']
    if not runs or len(set(runs)) != len(runs):
        raise ValueError('No runs or duplicate run directories')
    configurations = [yaml.safe_load((p / 'run_config.yaml').read_text()) for p in runs]
    cohort_hashes = set()
    for run, config in zip(runs, configurations):
        if (config['fusion_type'] != 'path_query_ct_layer4' or config['ct_roi_size'] != 64
                or config['pa_feature_dim'] != 1280):
            raise ValueError(f'Not a supported test15 run: {run}')
        for fold in range(5):
            for file in [run / f'fold_{fold}/train_predictions.csv', run / f'fold_{fold}/val_predictions.csv',
                         Path(config['checkpoint_root']) / f'fold_{fold}/best_model.pth']:
                if not file.is_file():
                    raise FileNotFoundError(f'Incomplete run: {file}')
        cohort_hashes.add(digest(Path(config['data_dir']) / 'all_label_roi64.csv'))
    if len(cohort_hashes) != 1:
        raise ValueError('Runs use different cohort/split tables')
    destination = Path(args.output_dir)
    destination.mkdir(parents=True, exist_ok=False)
    (destination / 'inference_config.yaml').write_text(yaml.safe_dump(dict(
        run_dirs=[str(p) for p in runs], repeats=args.repeats, seed=args.seed, batch_size=args.batch_size,
        num_workers=args.num_workers, device=args.device, folds=list(range(5)),
        shuffle_scope='validation fold only; derangements; same donors for both conditions and variants',
        query_shuffle='only query-path patch features; direct patient pathology and CT unchanged')))
    comparison = []
    print(f'Fixed-checkpoint inference: {len(runs)} runs, 5 folds, {args.repeats} repeats; device={device}', flush=True)
    for run, config in zip(runs, configurations):
        output = destination / run.name
        output.mkdir()
        print(f'\nRun: {run.name}', flush=True)
        dataset = (AllPatchQueryDataset(config['data_dir'], config['all_feature_dir'])
                   if config.get('all_patch_queries', False)
                   else Pa_CT_Dataset(config['data_dir'], roi_size=64, feature_dim=1280))
        records = []
        for fold in range(5):
            print(f'Fold {fold}/4', flush=True)
            records.extend(run_fold(run, config, dataset, fold, args, output / f'fold_{fold}', device))
            summarize(records, output)
        summary = pd.read_csv(output / 'summary.csv')
        summary.insert(0, 'run', run.name)
        comparison.append(summary)
    pd.concat(comparison, ignore_index=True).to_csv(destination / 'comparison.csv', index=False)
    (destination / 'completed.json').write_text(json.dumps(dict(runs=len(runs), folds_per_run=5,
        repeats_per_condition=args.repeats, status='complete'), indent=2) + '\n')
    print(f'Completed all inference checks: {destination}', flush=True)


if __name__ == '__main__':
    main()
