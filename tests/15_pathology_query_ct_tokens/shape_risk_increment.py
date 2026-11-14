"""Fixed-lambda Shape CT risk conditional increment; scheme A, not a teacher."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import joblib
import numpy as np
import pandas as pd
import torch
import yaml
from sksurv.metrics import concordance_index_censored
from threadpoolctl import threadpool_limits

from ct_shape_aux_model import CTWithShape
from ct_shape_aux_train import fit_auxiliary, load_sources
from dataset import cv_splits
from feature_increment import file_hash, load_bundle
from increment_cox import IncrementCox
from loss import cox_loss
from risk_calibration import paired_comparison, risk_scaler, standardize_risk
from risk_increment import ordered_predictions
from utils import save_cv_results, save_run_config, seed_everything
from visualization.survival_plots import analyze_fold


def fixed_shape_policy(selection, weight):
    records = [row for row in selection['inner_models'] if row['weight'] == weight]
    if (len(records) != 3 or {row['inner_fold'] for row in records} != {0, 1, 2}
            or any(not np.isfinite(row['selected_epoch']) or row['selected_epoch'] < 1
                   or int(row['selected_epoch']) != row['selected_epoch'] for row in records)):
        raise ValueError('Fixed weight needs exactly three valid STOP-selected inner epochs')
    epochs = [int(row['selected_epoch']) for row in sorted(records, key=lambda row: row['inner_fold'])]
    return dict(weight=float(weight), inner_STOP_epochs=epochs, epochs=int(np.median(epochs)),
                policy='fixed lambda; floor median fixed-candidate STOP epochs; HELD scores ignored')


def checked_predictions(frame, samples, indices):
    truth = samples.iloc[indices].copy()
    truth.index = (truth.pa_id + '|' + truth.ct_id).to_numpy()
    lookup = dict(zip(samples.ct_id, samples.pa_id + '|' + samples.ct_id))
    frame = frame.copy()
    frame['case_id'] = [str(case) if '|' in str(case) else lookup.get(str(case), '') for case in frame.case_id]
    if not frame.case_id.is_unique or set(frame.case_id) != set(truth.index):
        raise ValueError('Prediction patients do not match the intended role')
    frame = frame.set_index('case_id').loc[truth.index]
    if (not np.isfinite(frame[['dfs.month', 'dfs.status', 'risk_score']]).all().all()
            or not np.array_equal(frame['dfs.status'], truth.event)
            or not np.array_equal(frame['dfs.month'].to_numpy(dtype=np.float32), truth.time.to_numpy(dtype=np.float32))):
        raise ValueError('Prediction labels/risks differ from source cohort')
    # Source loaders store time as float32. Use the canonical source CSV labels
    # after validating this encoding, for every new head and exported table.
    frame['dfs.month'], frame['dfs.status'] = truth.time.to_numpy(), truth.event.to_numpy()
    frame.index.name = 'case_id'
    return frame.reset_index()


def assemble_ct_oof(samples, training, validation, blocks):
    allowed, excluded = set(training), set(validation)
    if allowed & excluded or allowed | excluded != set(range(len(samples))):
        raise ValueError('Outer training/validation must partition independent patients')
    frames, scalers, all_held = [], [], []
    if len(blocks) != 3 or {block['inner_fold'] for block in blocks} != {0, 1, 2}:
        raise ValueError('Require the same complete three inner folds')
    for block in sorted(blocks, key=lambda row: row['inner_fold']):
        fit, stop, held = [set(block[key]) for key in ['fit', 'stop', 'held']]
        if (fit & stop or fit & held or stop & held or fit | stop | held != allowed
                or (fit | stop | held) & excluded
                or any(len(block[key]) != len(set(block[key])) for key in ['fit', 'stop', 'held'])):
            raise ValueError('CT OOF patient entered FIT/STOP or outer validation leaked')
        fitting = checked_predictions(block['fit_predictions'], samples, block['fit'])
        prediction = checked_predictions(block['held_predictions'], samples, block['held'])
        scaler = risk_scaler(fitting.risk_score)
        prediction = prediction.rename(columns={'risk_score': 'raw_ct_risk'})
        prediction['ct_risk'] = standardize_risk(prediction.raw_ct_risk, scaler)
        prediction['inner_fold'] = block['inner_fold']
        frames.append(prediction)
        scalers.append(scaler)
        all_held.extend(block['held'])
    if len(all_held) != len(training) or set(all_held) != allowed:
        raise ValueError('Every outer-training patient needs exactly one honest CT OOF risk')
    result = pd.concat(frames, ignore_index=True).set_index('case_id')
    order = (samples.iloc[training].pa_id + '|' + samples.iloc[training].ct_id).to_numpy()
    return result.loc[order], scalers


def fit_risk_heads(oof, ridge):
    if not np.isfinite(ridge) or ridge < 0:
        raise ValueError('Fixed Cox ridge must be finite and nonnegative')
    if not oof.index.is_unique or not np.isfinite(oof[['pathology_risk', 'ct_A', 'ct_C', 'dfs.month', 'dfs.status']]).all().all():
        raise ValueError('Need unique finite complete calibration OOF rows')
    return {name: IncrementCox(ridge=ridge).fit(oof.pathology_risk,
            None if name == 'M0' else oof[['ct_A' if name == 'M1' else 'ct_C']].to_numpy(),
            oof['dfs.month'], oof['dfs.status']) for name in ['M0', 'M1', 'M2']}


def check_shape_metadata(path, samples, fit, shapes, args, weight, epoch):
    metadata = json.loads(path.read_text())
    scaler = metadata['scaler']
    ids = samples.iloc[fit].ct_id.tolist()
    targets = shapes.loc[ids, args.shape_columns].to_numpy()
    if (metadata['weight'] != weight or metadata['selected_epoch'] != epoch
            or scaler['columns'] != args.shape_columns or scaler['n_fit'] != len(fit)
            or scaler['fit_ct_ids_sha256'] != hashlib.sha256('\n'.join(ids).encode()).hexdigest()
            or metadata['initialization'] != 'PUBLIC MedicalNet + fresh original risk head; no downstream warm-start'):
        raise ValueError(f'Weight/epoch/targets/fit membership mismatch: {path}')
    np.testing.assert_allclose(scaler['mean'], targets.mean(0), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(scaler['std'], targets.std(0), rtol=1e-6, atol=1e-6)
    return metadata


def audit_sources(args):
    args.shape_columns = ['log_volume_mm3', 'diameter_mm', 'sphericity']
    dataset, _, shapes, partitions, _, _, _, provenance = load_sources(args)
    source, auxiliary = Path(args.source_run), Path(args.auxiliary_run)
    aux_config = yaml.safe_load((auxiliary / 'run_config.yaml').read_text())
    if not (auxiliary / 'completed.json').is_file() or aux_config['target'] != 'allshape':
        raise ValueError('Require completed ALL-Shape auxiliary source')
    if Path(aux_config['source_run']).resolve() != source.resolve():
        raise ValueError('Auxiliary source uses different CT/PA roles')
    for key in ['data_dir', 'batch_size', 'cox_batch_size', 'dropout', 'lr', 'ct_backbone_lr',
                'weight_decay', 'seed', 'num_epochs', 'patience', 'cohort_sha256', 'public_pretrained_sha256']:
        if aux_config[key] != getattr(args, key):
            raise ValueError(f'A/C settings differ: {key}')
    for path, sha in provenance.items():
        if aux_config['source_hashes'].get(path) != sha:
            raise ValueError(f'Auxiliary source provenance changed: {path}')
    policies, bundles = {}, {}
    checkpoint_root = Path(aux_config['checkpoint_root'])
    for fold, (training, validation) in enumerate(cv_splits(dataset.samples)):
        selection_path = auxiliary / f'fold_{fold}/selection.json'
        selection = json.loads(selection_path.read_text())
        policy = fixed_shape_policy(selection, args.shape_weight)
        for inner, fit, stop, held in partitions[fold]:
            folder = checkpoint_root / f'fold_{fold}/lambda_{args.shape_weight:g}/inner_{inner}'
            epoch = policy['inner_STOP_epochs'][inner]
            metadata = check_shape_metadata(folder / 'checkpoint_metadata.json', dataset.samples,
                                             fit, shapes, args, args.shape_weight, epoch)
            history = pd.read_csv(folder / 'training_history.csv', float_precision='round_trip')
            if (len(history) != metadata['epochs_completed']
                    or int(history.loc[history.val_cindex.idxmax(), 'epoch']) != epoch
                    or metadata['selection'] != 'STOP C-index only'):
                raise ValueError('Inner epoch was not selected exclusively from STOP history')
            checked_predictions(pd.read_csv(folder / 'held_predictions.csv', dtype={'case_id': str},
                                float_precision='round_trip'), dataset.samples, held)
            for filename in ['best_model.pth', 'checkpoint_metadata.json', 'training_history.csv', 'held_predictions.csv']:
                path = folder / filename
                provenance[str(path)] = file_hash(path)
        reuse = selection['weight'] == args.shape_weight and selection['epochs'] == policy['epochs']
        policy['reuse_outer'] = bool(reuse)
        if reuse:
            folder = checkpoint_root / f'fold_{fold}/outer'
            check_shape_metadata(folder / 'checkpoint_metadata.json', dataset.samples, training,
                                 shapes, args, args.shape_weight, policy['epochs'])
            for path in [folder / 'best_model.pth', folder / 'checkpoint_metadata.json',
                         *[auxiliary / f'fold_{fold}/{split}_predictions.csv' for split in ['train', 'val']]]:
                provenance[str(path)] = file_hash(path)
        policy['source_checkpoint_root'] = str(checkpoint_root / f'fold_{fold}')
        policies[fold] = policy
        bundles[fold] = load_bundle(fold, source, dataset.samples)
        calibration = yaml.safe_load((source / f'fold_{fold}/calibration.yaml').read_text())
        for split, indices in [('train', training), ('val', validation)]:
            covariates = bundles[fold][split]
            path = source / f'late_fusion/fold_{fold}/{split}_predictions.csv'
            provenance[str(path)] = file_hash(path)
            for modality in ['pathology', 'ct']:
                path = source / f'{modality}/fold_{fold}/{split}_predictions.csv'
                reference = checked_predictions(pd.read_csv(path, dtype={'case_id': str},
                    float_precision='round_trip'), dataset.samples, indices).set_index('case_id').loc[covariates.index]
                np.testing.assert_allclose(covariates['raw_' + modality + '_risk'], reference.risk_score,
                                           rtol=1e-12, atol=1e-12)
                np.testing.assert_allclose(covariates[modality + '_risk'],
                    standardize_risk(reference.risk_score, calibration['outer_scalers'][modality]),
                    rtol=1e-12, atol=1e-12)
                provenance[str(path)] = file_hash(path)
        provenance[str(selection_path)] = file_hash(selection_path)
    for path in [auxiliary / 'run_config.yaml', auxiliary / 'completed.json']:
        provenance[str(path)] = file_hash(path)
    print('Fixed-Shape audit: 15 honest inner checkpoints; outer epochs=' +
          str([policies[i]['epochs'] for i in range(5)]) +
          '; new outer fits=' + str(sum(not p['reuse_outer'] for p in policies.values())), flush=True)
    return dataset, shapes, partitions, policies, bundles, provenance


def ct_risks(fold, dataset, shapes, partitions, policy, args, device, folder):
    samples = dataset.samples
    training, validation = next(pair for number, pair in enumerate(cv_splits(samples)) if number == fold)
    blocks = []
    for inner, fit, stop, held in partitions:
        checkpoint = Path(policy['source_checkpoint_root']) / f'lambda_{args.shape_weight:g}/inner_{inner}'
        print(f'Outer{fold} Inner{inner}: inference only, fixed lambda={args.shape_weight:g}; '
              f'FIT={len(fit)}, STOP={len(stop)}, OOF={len(held)}', flush=True)
        seed_everything(args.seed)
        model = CTWithShape(3, dropout=args.dropout).to(device)
        model.load_state_dict(torch.load(checkpoint / 'best_model.pth', map_location=device, weights_only=True), strict=True)
        fitting = ordered_predictions(model, dataset, fit, 'ct', args, device)
        prediction = ordered_predictions(model, dataset, held, 'ct', args, device)
        saved = checked_predictions(pd.read_csv(checkpoint / 'held_predictions.csv', dtype={'case_id': str},
                                    float_precision='round_trip'), samples, held)
        np.testing.assert_allclose(prediction.risk_score, saved.risk_score, rtol=1e-5, atol=1e-6)
        blocks.append(dict(inner_fold=inner, fit=fit, stop=stop, held=held,
                           fit_predictions=fitting, held_predictions=prediction))
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    oof, scalers = assemble_ct_oof(samples, training, validation, blocks)
    oof.reset_index().to_csv(folder / 'ct_fixed_inner_oof.csv', index=False)
    if policy['reuse_outer']:
        print(f'Outer{fold}: reuse existing fixed-lambda outer checkpoint/predictions, epochs={policy["epochs"]}', flush=True)
        raw = {split: checked_predictions(pd.read_csv(Path(args.auxiliary_run) / f'fold_{fold}/{split}_predictions.csv',
               dtype={'case_id': str}, float_precision='round_trip'), samples, indices).set_index('case_id')
               for split, indices in [('train', training), ('val', validation)]}
        checkpoint = Path(policy['source_checkpoint_root']) / 'outer/best_model.pth'
    else:
        print(f'Outer{fold}: fresh PUBLIC initialization; lambda={args.shape_weight:g}; fixed epochs={policy["epochs"]}', flush=True)
        directory = Path(args.checkpoint_root) / f'fold_{fold}/outer'
        model, _ = fit_auxiliary(dataset, shapes, training, None, args, device, directory,
                                 args.shape_weight, fixed_epochs=policy['epochs'])
        raw = {split: checked_predictions(ordered_predictions(model, dataset, indices, 'ct', args, device),
               samples, indices).set_index('case_id') for split, indices in [('train', training), ('val', validation)]}
        checkpoint = directory / 'best_model.pth'
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    scaler = risk_scaler(raw['train'].risk_score)
    for split, frame in raw.items():
        frame['ct_risk'] = standardize_risk(frame.risk_score, scaler)
        frame.reset_index().to_csv(folder / f'ct_fixed_{split}_predictions.csv', index=False)
    metadata = dict(**policy, inner_fit_scalers=scalers, outer_train_scaler=scaler,
                    final_checkpoint=str(checkpoint), final_checkpoint_sha256=file_hash(checkpoint),
                    oof_policy='HELD patients excluded from both FIT and STOP; fixed weight, no HELD lambda selection')
    (folder / 'ct_risk_provenance.json').write_text(json.dumps(metadata, indent=2))
    return oof, raw


def export_heads(fold, heads, bundle, ct_frames, root):
    rows, predictions = {}, {}
    for name, head in heads.items():
        destination = root / name / f'fold_{fold}'
        destination.mkdir(parents=True)
        joblib.dump(head, destination / 'head.joblib')
        parameters = dict(coefficients=head.model.coef_.tolist(), pa_ooF_mean=head.pa_scaler.mean_.tolist(),
            pa_oof_scale=head.pa_scaler.scale_.tolist(), n_oof=int(head.pa_scaler.n_samples_seen_),
            ct_oof_mean=None if head.ct_pipeline is None else head.ct_pipeline.named_steps['scale'].mean_.tolist(),
            ct_oof_scale=None if head.ct_pipeline is None else head.ct_pipeline.named_steps['scale'].scale_.tolist(),
            ridge_ct=head.ridge, ridge_pa=0., coefficient_constraints='unrestricted real slopes')
        (destination / 'head_parameters.json').write_text(json.dumps(parameters, indent=2))
        frames = {}
        for split in ['train', 'val']:
            base = bundle[split]
            ct = (None if name == 'M0' else base.ct_risk.to_numpy()[:, None] if name == 'M1'
                  else ct_frames[split].loc[base.index, 'ct_risk'].to_numpy()[:, None])
            frame = base[['dfs.month', 'dfs.status']].copy()
            frame['risk_score'] = head.predict(base.pathology_risk, ct)
            frames[split] = frame.reset_index()
        frames['train'].to_csv(destination / 'train_predictions.csv', index=False)
        metrics = analyze_fold(frames['train'], frames['val'], destination)
        for split, frame in frames.items():
            metrics[split + '_loss'] = float(cox_loss(*[torch.tensor(frame[key].to_numpy(), dtype=torch.float64)
                for key in ['risk_score', 'dfs.month', 'dfs.status']]))
        rows[name], predictions[name] = dict(fold=fold, **metrics), frames['val']
        print(f'{name} Outer{fold}: Train={metrics["train_cindex"]:.4f} | Val={metrics["val_cindex"]:.4f} | '
              f'Val Cox NLL={metrics["val_loss"]:.4f}', flush=True)
    return rows, predictions


def report_results(root, metrics, predictions, ct_only, args):
    comparisons = {}
    for name, reference in [('M1_minus_M0', ('M1', 'M0')), ('M2_minus_M0', ('M2', 'M0')), ('M2_minus_M1', ('M2', 'M1'))]:
        candidate, baseline = reference
        paired = []
        for current, control in zip(predictions[candidate], predictions[baseline]):
            control = control.set_index('case_id').loc[current.case_id]
            frame = current.copy()
            frame['reference'] = control.risk_score.to_numpy()
            paired.append(frame)
        result = paired_comparison(paired, seed=args.seed, reference_columns={'reference': 'reference'})['fusion_minus_reference']
        result['positive_folds'] = sum(a['val_cindex'] > b['val_cindex'] for a, b in zip(metrics[candidate], metrics[baseline]))
        comparisons[name] = result
    (root / 'paired_comparisons.json').write_text(json.dumps(dict(comparisons=comparisons, repeats=2000,
        method='paired patients resampled within each fixed outer fold; mean fold differences',
        limitation='conditional on trained models; excludes retraining, tuning and earlier exploration uncertainty'), indent=2))
    table = pd.DataFrame([dict(model=name, **{key + '_' + stat: float(getattr(pd.DataFrame(rows)[key], stat)(skipna=False))
        for key in ['train_cindex', 'val_cindex', 'train_loss', 'val_loss', 'auc_36m', 'auc_60m', 'gap']
        for stat in ['mean', 'std']}) for name, rows in metrics.items()])
    table.to_csv(root / 'comparison_summary.csv', index=False)
    per_fold = pd.DataFrame([dict(fold=fold, **{name + '_' + key: metrics[name][fold][key]
        for name in metrics for key in ['val_cindex', 'val_loss', 'auc_36m', 'auc_60m']},
        delta_A=metrics['M1'][fold]['val_cindex']-metrics['M0'][fold]['val_cindex'],
        delta_C=metrics['M2'][fold]['val_cindex']-metrics['M0'][fold]['val_cindex'],
        delta_new=metrics['M2'][fold]['val_cindex']-metrics['M1'][fold]['val_cindex']) for fold in range(5)])
    per_fold.to_csv(root / 'per_fold_comparison.csv', index=False)
    pd.DataFrame(ct_only).to_csv(root / 'ct_only_context.csv', index=False)
    lines = ['# Fixed-weight Shape CT risk conditional increment (scheme A)', '',
        f'721 patients, 251 events; original outer5/inner3 roles. Fixed ALL-Shape lambda={args.shape_weight:g}; '
        f'fixed CT ridge={args.cox_ridge:g}, PA slope unpenalized. Both risk slopes are free.', '',
        '| Model | Train C-index | Val C-index | AUC36 | AUC60 | Val Cox NLL |', '|---|---|---|---|---|---|']
    for row in table.to_dict('records'):
        lines.append('| ' + row['model'] + ' | ' + ' | '.join(f'{row[key+"_mean"]:.4f} ± {row[key+"_std"]:.4f}'
            for key in ['train_cindex', 'val_cindex', 'auc_36m', 'auc_60m', 'val_loss']) + ' |')
    lines += ['', '| Contrast | Mean ΔC-index | Conditional 95% interval | Positive folds |', '|---|---|---|---|']
    for name, row in comparisons.items():
        lines.append(f'| {name} | {row["delta_cindex"]:+.4f} | [{row["lower"]:+.4f}, {row["upper"]:+.4f}] | {row["positive_folds"]}/5 |')
    fixed = pd.DataFrame(ct_only).C_val_cindex
    lines += ['', f'Fixed-lambda C CT-only Val C-index: {fixed.mean():.4f} ± {fixed.std():.4f}. '
        'This differs from the historical automatic-lambda strategy (0.7136).', '',
        '## Scope and safeguards', '',
        '- M0/M1/M2 refit the same Cox procedure on unchanged PA OOF risks. M1 uses source A CT OOF; M2 uses fixed-weight C CT OOF. No head hyperparameter is selected.',
        '- Each C HELD prediction comes from its matching FIT/STOP checkpoint. Risk scalers use FIT eval predictions only; outer scalers use outer-training eval predictions only. Cox scalers use OOF covariates only.',
        '- Reused checkpoints are inference/reference only. New outer CT fits start from PUBLIC MedicalNet/new heads, never a project-trained unimodal checkpoint. No PA, teacher or student is trained.',
        '- Outer epochs are the median of fixed-candidate STOP epochs, not the previous selected-lambda epochs. HELD scores and outer outcomes cannot select lambda or outer epochs.',
        '- Final Train metrics are in-sample outer-refit eval predictions; they are not independent validation. Fusion training covariates and predictions are saved separately.',
        '- This is exploratory fixed-setting diagnosis after previous same-cohort exploration, not evaluation of automatic lambda selection, independent confirmation, or a causal Shape effect.',
        '- A null scalar-risk increment does not prove the 512-dimensional representation has no complementary information. A positive result does not guarantee joint-teacher/student improvement.',
        '- Cohort source pathology refits still use historical epochs [1,14,1,14,23]; two short fits remain a baseline stability limitation. Time/event and IPCW/KM conventions are unchanged.',
        '- Shape targets require masks during auxiliary training. Risk inference uses CT only on the existing mask-localized ROI64; upstream CT acquisition/intensity processing remains unknown.']
    (root / 'report.md').write_text('\n'.join(lines) + '\n')


def run(args):
    if (np.__version__ != '1.26.4' or args.shape_weight not in [.05, .2]
            or not np.isfinite(args.cox_ridge) or args.cox_ridge < 0 or args.num_workers < 0):
        raise ValueError('Use isolated NumPy1.26.4, an existing fixed weight .05/.2 and nonnegative ridge/workers')
    print('Fixed-Shape conditional risk diagnostic: auditing source roles, predictions and checkpoints...', flush=True)
    dataset, shapes, partitions, policies, bundles, provenance = audit_sources(args)
    if args.audit_only:
        print('AUDIT ONLY: source checks passed; no fitting/inference/teacher/student started.', flush=True)
        return
    root, checkpoints = Path(args.results_root), Path(args.checkpoint_root)
    if root.exists() or checkpoints.exists():
        raise FileExistsError('Output/checkpoint path exists; refusing to overwrite')
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    torch.set_num_threads(4)
    args.fusion_type = 'fixed_shape_risk_increment_diagnostic'
    args.survival_loss = 'cox'
    save_run_config(args, 'ct', 'shape_risk_increment', extra=dict(teacher_student_unchanged=True,
        mode='A: fixed lambda, fixed Cox penalty; no evaluation of automatic lambda selection'))
    (root / 'source_audit.json').write_text(json.dumps(dict(fold_policies=policies, source_hashes=provenance), indent=2))
    metrics = {name: [] for name in ['M0', 'M1', 'M2']}
    predictions = {name: [] for name in metrics}
    ct_only = []
    for fold in range(5):
        folder = root / f'fold_{fold}'
        folder.mkdir()
        ct_oof, ct_frames = ct_risks(fold, dataset, shapes, partitions[fold], policies[fold], args, device, folder)
        bundle = bundles[fold]
        oof = bundle['oof'][['dfs.month', 'dfs.status', 'pathology_risk', 'inner_fold']].copy()
        if not set(oof.index) == set(ct_oof.index):
            raise ValueError('PA/A/C OOF patient mismatch')
        np.testing.assert_array_equal(oof.inner_fold, ct_oof.loc[oof.index, 'inner_fold'])
        np.testing.assert_array_equal(oof['dfs.status'], ct_oof.loc[oof.index, 'dfs.status'])
        np.testing.assert_array_equal(oof['dfs.month'].to_numpy(dtype=np.float32), ct_oof.loc[oof.index, 'dfs.month'].to_numpy(dtype=np.float32))
        oof['ct_A'], oof['ct_C'] = bundle['oof'].ct_risk, ct_oof.loc[oof.index, 'ct_risk']
        oof.reset_index().to_csv(folder / 'calibration_oof_covariates.csv', index=False)
        heads = fit_risk_heads(oof, args.cox_ridge)
        fold_metrics, fold_predictions = export_heads(fold, heads, bundle, ct_frames, root)
        for name in heads:
            metrics[name].append(fold_metrics[name])
            predictions[name].append(fold_predictions[name])
        ct_only.append(dict(fold=fold, fixed_lambda=args.shape_weight, fixed_epoch=policies[fold]['epochs'],
            A_val_cindex=float(concordance_index_censored(bundle['val']['dfs.status'].astype(bool), bundle['val']['dfs.month'], bundle['val'].raw_ct_risk)[0]),
            C_val_cindex=float(concordance_index_censored(ct_frames['val']['dfs.status'].astype(bool), ct_frames['val']['dfs.month'], ct_frames['val'].risk_score)[0])))
        (folder / 'completed.json').write_text(json.dumps(dict(fold=fold, metrics=fold_metrics), indent=2))
    for name, rows in metrics.items():
        print('\n' + name, flush=True)
        save_cv_results(rows, root / name)
    report_results(root, metrics, predictions, ct_only, args)
    for path, sha in provenance.items():
        if file_hash(path) != sha:
            raise ValueError(f'Source modified during diagnostic: {path}')
    (root / 'completed.json').write_text(json.dumps(dict(patients=721, events=251, folds=5,
        fixed_shape_weight=args.shape_weight, fixed_cox_ridge=args.cox_ridge,
        new_outer_ct_fits=sum(not p['reuse_outer'] for p in policies.values()),
        source_hashes_unchanged=True, teacher_student_training=False), indent=2))
    print('Fixed-Shape conditional risk increment complete; report.md saved.', flush=True)


if __name__ == '__main__':
    experiment = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source_run', default=str(experiment / 'results/nested_risk_increment'))
    parser.add_argument('--feature_run', default=str(experiment / 'results/feature_increment'))
    parser.add_argument('--auxiliary_run', default=str(experiment / 'results/ct_shape_aux/allshape'))
    parser.add_argument('--shape_weight', type=float, default=.2)
    parser.add_argument('--cox_ridge', type=float, default=1.)
    parser.add_argument('--results_root', default=str(experiment / 'results/shape_risk_increment'))
    parser.add_argument('--checkpoint_root', default=str(experiment / 'checkpoints/shape_risk_increment'))
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument('--audit_only', action='store_true')
    with threadpool_limits(limits=2):
        run(parser.parse_args())
