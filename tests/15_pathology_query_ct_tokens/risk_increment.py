"""Nested-CV diagnostic of CT risk added to pathology risk; not a teacher model."""

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
from sksurv.metrics import concordance_index_censored

from dataset import CT_Dataset, Path_Dataset, build_dataloader, cox_batch_sizes, cv_splits, resolve_data_dir
from loss import cox_loss
from model.build import CT_Model, Pa_Model
from risk_calibration import fit_alpha, inner_splits, paired_comparison, risk_scaler, standardize_risk
from utils import build_ct_optimizer, save_cv_results, save_run_config, seed_everything
from visualization.survival_plots import analyze_fold


@torch.inference_mode()
def predict_risk(model, loader, device):
    model.eval()
    records = []
    for features, event, time, identifiers in loader:
        output = model(features.to(device, non_blocking=True))
        risk = (output[0] if isinstance(output, tuple) else output).reshape(-1).cpu().numpy()
        for identity, t, e, r in zip(identifiers, time.reshape(-1), event.reshape(-1), risk):
            identity = str(identity.item()) if torch.is_tensor(identity) else str(identity)
            records.append(dict(case_id=identity, **{'dfs.month': float(t), 'dfs.status': int(e)}, risk_score=float(r)))
    return pd.DataFrame(records)


def train_risk_model(model, optimizer, train_loader, stop_loader, args, device,
                     checkpoint_dir, forwards_per_update=1, fixed_epochs=None):
    if (fixed_epochs is None) == (stop_loader is None):
        raise ValueError('Use either inner stopping or fixed-epoch refitting, never both/neither')
    epochs = args.num_epochs if fixed_epochs is None else fixed_epochs
    if epochs < 1:
        raise ValueError('Training epochs must be positive')
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best, selected_epoch, wait, history = -np.inf, None, 0, []
    ends = set(np.cumsum(cox_batch_sizes(len(train_loader), forwards_per_update)).tolist())
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        risks, times, events, losses = [], [], [], []
        all_risk, all_time, all_event = [], [], []
        for step, (features, event, time, _) in enumerate(train_loader, 1):
            output = model(features.to(device, non_blocking=True))
            risk = (output[0] if isinstance(output, tuple) else output).reshape(-1)
            all_risk.append(risk.detach().cpu()); all_time.append(time.reshape(-1)); all_event.append(event.reshape(-1))
            risks.append(risk); times.append(time.to(device).reshape(-1)); events.append(event.to(device).reshape(-1))
            if step in ends:
                loss = cox_loss(torch.cat(risks), torch.cat(times), torch.cat(events))
                loss.backward(); optimizer.step(); optimizer.zero_grad()
                losses.append(float(loss.detach()))
                risks, times, events = [], [], []
        row = dict(epoch=epoch, train_loss=float(np.mean(losses)), train_cindex=float(concordance_index_censored(
            torch.cat(all_event).numpy().astype(bool), torch.cat(all_time).numpy(), torch.cat(all_risk).numpy())[0]))
        line = f'Epoch {epoch}/{epochs} | Train Loss: {row["train_loss"]:.4f} | Train C-index: {row["train_cindex"]:.4f}'
        if stop_loader is not None:
            predictions = predict_risk(model, stop_loader, device)
            t, e, r = [predictions[column].to_numpy() for column in ('dfs.month', 'dfs.status', 'risk_score')]
            row.update(val_loss=float(cox_loss(torch.tensor(r), torch.tensor(t), torch.tensor(e))),
                       val_cindex=float(concordance_index_censored(e.astype(bool), t, r)[0]))
            line += f' | Val Loss: {row["val_loss"]:.4f} | Val C-index: {row["val_cindex"]:.4f}'
            if row['val_cindex'] > best:
                best, selected_epoch, wait = row['val_cindex'], epoch, 0
                torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, checkpoint_dir / 'best_model.pth')
            else:
                wait += 1
        history.append(row)
        print(line, flush=True)
        pd.DataFrame(history).to_csv(checkpoint_dir / 'training_history.csv', index=False)
        if stop_loader is not None and args.patience > 0 and wait >= args.patience:
            print(f'Early stopping at epoch {epoch}', flush=True)
            break
    if stop_loader is None:
        selected_epoch = epochs
        torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, checkpoint_dir / 'best_model.pth')
    else:
        model.load_state_dict(torch.load(checkpoint_dir / 'best_model.pth', map_location=device, weights_only=True), strict=True)
    metadata = dict(selected_epoch=int(selected_epoch), epochs_completed=epoch,
                    selection='fixed_epochs_from_inner_training' if stop_loader is None else 'inner_stop_cindex')
    if stop_loader is not None:
        metadata['best_inner_stop_cindex'] = best
    (checkpoint_dir / 'checkpoint_metadata.yaml').write_text(yaml.safe_dump(metadata))
    return model, metadata


def fit_modality(modality, dataset, fit, stop, args, device, checkpoint, fixed_epochs=None):
    seed_everything(args.seed)
    if modality == 'pathology':
        model = Pa_Model(feature_dim=1280, abmil_dropout=args.dropout, l2_normalize=True).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        model = CT_Model(pretrained_path=args.ct_pretrained_path, dropout=args.dropout).to(device)
        optimizer = build_ct_optimizer(model, args.lr, args.weight_decay, args.ct_backbone_lr)
    train = build_dataloader(dataset, indices=fit, batch_size=1 if modality == 'pathology' else args.batch_size,
                             shuffle=True, merge_tail=modality == 'ct', num_workers=args.num_workers)
    stopping = None if stop is None else build_dataloader(dataset, indices=stop,
        batch_size=1 if modality == 'pathology' else args.batch_size, num_workers=args.num_workers)
    return train_risk_model(model, optimizer, train, stopping, args, device, checkpoint,
                            forwards_per_update=args.cox_batch_size if modality == 'pathology' else 1,
                            fixed_epochs=fixed_epochs)


def ordered_predictions(model, dataset, indices, modality, args, device):
    loader = build_dataloader(dataset, indices=indices, batch_size=1 if modality == 'pathology' else args.batch_size,
                              num_workers=args.num_workers)
    frame = predict_risk(model, loader, device)
    rows = dataset.samples.iloc[indices]
    key = 'pa_id' if modality == 'pathology' else 'ct_id'
    np.testing.assert_array_equal(frame.case_id, rows[key].to_numpy())
    np.testing.assert_allclose(frame['dfs.month'], rows.time, rtol=1e-7, atol=1e-5)
    np.testing.assert_array_equal(frame['dfs.status'], rows.event)
    frame['case_id'] = (rows.pa_id + '|' + rows.ct_id).to_numpy()
    return frame


def outer_fold(fold, train_indices, val_indices, datasets, args, device):
    samples = datasets['pathology'].samples
    partitions = inner_splits(samples, train_indices, args.inner_folds, args.stop_fraction, args.seed)
    destination = Path(args.results_root) / f'fold_{fold}'
    destination.mkdir(parents=True)
    roles, calibrators, epoch_choices = [], [], {'pathology': [], 'ct': []}
    oof = pd.DataFrame(index=train_indices)
    oof['case_id'] = (samples.iloc[train_indices].pa_id + '|' + samples.iloc[train_indices].ct_id).to_numpy()
    oof['dfs.month'], oof['dfs.status'] = samples.iloc[train_indices].time, samples.iloc[train_indices].event
    for inner, (fit, stop, held) in enumerate(partitions):
        for role, indices in [('fit', fit), ('stop', stop), ('oof', held)]:
            for index in indices:
                roles.append(dict(inner_fold=inner, case_id=f'{samples.iloc[index].pa_id}|{samples.iloc[index].ct_id}', role=role))
        oof.loc[held, 'inner_fold'] = inner
        for modality, dataset in datasets.items():
            print(f'\nOuter {fold} | Inner {inner + 1}/{args.inner_folds} | {modality} | '
                  f'fit={len(fit)}, stop={len(stop)}, OOF={len(held)}', flush=True)
            checkpoint = Path(args.checkpoint_root) / f'fold_{fold}/inner_{inner}/{modality}'
            model, metadata = fit_modality(modality, dataset, fit, stop, args, device, checkpoint)
            fit_prediction = ordered_predictions(model, dataset, fit, modality, args, device)
            held_prediction = ordered_predictions(model, dataset, held, modality, args, device)
            scaler = risk_scaler(fit_prediction.risk_score)
            oof.loc[held, modality + '_risk'] = standardize_risk(held_prediction.risk_score, scaler)
            oof.loc[held, 'raw_' + modality + '_risk'] = held_prediction.risk_score.to_numpy()
            epoch_choices[modality].append(metadata['selected_epoch'])
            calibrators.append(dict(inner_fold=inner, modality=modality, scaler=scaler, **metadata))
            del model
            if device.type == 'cuda':
                torch.cuda.empty_cache()
    assert len(oof) == len(train_indices) and not oof.isna().any().any()
    assert not set(oof.case_id) & set((samples.iloc[val_indices].pa_id + '|' + samples.iloc[val_indices].ct_id))
    oof.to_csv(destination / 'inner_oof_predictions.csv', index=False)
    pd.DataFrame(roles).to_csv(destination / 'inner_split_roles.csv', index=False)
    calibration = fit_alpha(oof.pathology_risk, oof.ct_risk, oof['dfs.month'], oof['dfs.status'])
    calibration.update(inner_models=calibrators, selected_epochs={k: int(np.median(v)) for k, v in epoch_choices.items()},
                       epoch_rule='floor(median(inner-selected epochs)); fresh outer refit; no outer stopping', outer_scalers={})
    print(f'Outer {fold} | OOF-selected alpha: {calibration["alpha"]:.6f} | '
          f'Cox {calibration["pathology_only_cox"]:.6f} -> {calibration["fitted_cox"]:.6f} | '
          f'Refit epochs: {calibration["selected_epochs"]}', flush=True)
    predictions = {'train': {}, 'val': {}}
    # No outer-validation prediction occurs until alpha and all epoch counts are fixed.
    for modality, dataset in datasets.items():
        print(f'\nOuter {fold} | Fresh {modality} refit on all {len(train_indices)} training patients', flush=True)
        checkpoint = Path(args.checkpoint_root) / f'fold_{fold}/outer/{modality}'
        model, _ = fit_modality(modality, dataset, train_indices, None, args, device, checkpoint,
                                fixed_epochs=calibration['selected_epochs'][modality])
        training = ordered_predictions(model, dataset, train_indices, modality, args, device)
        calibration['outer_scalers'][modality] = risk_scaler(training.risk_score)
        predictions['train'][modality] = training
        predictions['val'][modality] = ordered_predictions(model, dataset, val_indices, modality, args, device)
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    for split, tables in predictions.items():
        np.testing.assert_array_equal(tables['pathology'].case_id, tables['ct'].case_id)
        fused = tables['pathology'][['case_id', 'dfs.month', 'dfs.status']].copy()
        for modality in datasets:
            fused['raw_' + modality + '_risk'] = tables[modality].risk_score.to_numpy()
            fused[modality + '_risk'] = standardize_risk(tables[modality].risk_score, calibration['outer_scalers'][modality])
        fused['risk_score'] = fused.pathology_risk + calibration['alpha'] * fused.ct_risk
        predictions[split]['late_fusion'] = fused
    (destination / 'calibration.yaml').write_text(yaml.safe_dump(calibration))
    metrics = {}
    for name in ('pathology', 'ct', 'late_fusion'):
        root = Path(args.results_root) / name / f'fold_{fold}'
        root.mkdir(parents=True)
        train, val = predictions['train'][name], predictions['val'][name]
        train.to_csv(root / 'train_predictions.csv', index=False)
        print(f'\nOuter {fold} final evaluation | {name}', flush=True)
        extra = analyze_fold(train, val, root)
        losses = {split + '_loss': float(cox_loss(torch.tensor(frame.risk_score.to_numpy()),
                    torch.tensor(frame['dfs.month'].to_numpy()), torch.tensor(frame['dfs.status'].to_numpy())))
                  for split, frame in [('train', train), ('val', val)]}
        metrics[name] = dict(fold=fold, **losses, **extra)
        print(f'Train C-index: {extra["train_cindex"]:.4f} | Val C-index: {extra["val_cindex"]:.4f}', flush=True)
    (destination / 'completed.json').write_text(json.dumps(dict(outer_fold=fold, alpha=calibration['alpha'],
        metrics=metrics), indent=2))
    return metrics, predictions['val']['late_fusion'], calibration['alpha']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    experiment = Path(__file__).resolve().parent
    base = experiment.parents[1]
    parser.add_argument('--data_dir', default=str(base / 'data/eagle/top25'))
    parser.add_argument('--ct_pretrained_path', default=str(base / 'model/ct_pretrain/resnet_18_23dataset.pth'))
    parser.add_argument('--results_root', default=str(experiment / 'results/nested_risk_increment'))
    parser.add_argument('--checkpoint_root', default=str(experiment / 'checkpoints/nested_risk_increment'))
    parser.add_argument('--inner_folds', type=int, default=3)
    parser.add_argument('--stop_fraction', type=float, default=.15)
    parser.add_argument('--outer_folds', type=int, nargs='+', default=list(range(5)))
    parser.add_argument('--num_epochs', type=int, default=30)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--cox_batch_size', type=int, default=32)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--dropout', type=float, default=.25)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--ct_backbone_lr', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    args = parser.parse_args()
    if (args.num_epochs < 1 or args.batch_size < 1 or args.cox_batch_size < 1 or args.inner_folds < 2
            or not 0 < args.stop_fraction < .5 or not 0 <= args.dropout < 1 or args.num_workers < 0
            or not args.outer_folds or len(set(args.outer_folds)) != len(args.outer_folds)
            or not set(args.outer_folds).issubset(range(5))):
        parser.error('Invalid epoch, batch, split, worker, dropout or outer-fold setting')
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    torch.set_num_threads(4)
    args.data_dir = resolve_data_dir(args.data_dir, 64)
    args.fusion_type, args.experiment_type = 'risk_increment_nested_cv', 'diagnostic_only_not_teacher'
    args.survival_loss, args.pathology_patch_l2, args.ct_pooled_l2 = 'cox', True, False
    args.cohort_sha256 = hashlib.sha256((Path(args.data_dir) / 'all_label_roi64.csv').read_bytes()).hexdigest()
    args.public_pretrained_sha256 = hashlib.sha256(Path(args.ct_pretrained_path).read_bytes()).hexdigest()
    for root in (Path(args.results_root), Path(args.checkpoint_root)):
        if root.exists():
            raise FileExistsError(f'Refusing to overwrite existing experiment: {root}')
    datasets = {'pathology': Path_Dataset(args.data_dir, roi_size=64, feature_dim=1280),
                'ct': CT_Dataset(args.data_dir, roi_size=64)}
    samples = datasets['pathology'].samples
    pd.testing.assert_frame_equal(samples[['pa_id', 'ct_id', 'time', 'event', 'split']],
                                  datasets['ct'].samples[['pa_id', 'ct_id', 'time', 'event', 'split']])
    if not samples.pa_id.is_unique or not samples.ct_id.is_unique:
        raise ValueError('This paired diagnostic requires one unique PA/CT pair per patient')
    splits = cv_splits(samples)
    seed_everything(args.seed)
    save_run_config(args, 'diagnostic', 'nested_risk_increment')
    samples[['pa_id', 'ct_id', 'time', 'event', 'split']].to_csv(Path(args.results_root) / 'outer_splits.csv', index=False)
    print(f'Using Device: {device} | Loaded {len(samples)} paired patients | Outer folds: {args.outer_folds}', flush=True)
    print(f'Nested risk-increment diagnostic: {args.inner_folds} inner folds, separate stopping fraction {args.stop_fraction:g}; '
          'alpha from OOF Cox; outer validation never selects checkpoint/alpha', flush=True)
    print('Fresh independent unimodal fits only; public CT weights; no downstream checkpoint inputs; '
          'formal teacher/student training unchanged', flush=True)
    results, validation, alphas = {k: [] for k in ('pathology', 'ct', 'late_fusion')}, [], []
    for fold in args.outer_folds:
        print(f'\n{"=" * 50}\nOuter Fold {fold}/4\n{"=" * 50}', flush=True)
        metrics, prediction, alpha = outer_fold(fold, *splits[fold], datasets, args, device)
        for name, row in metrics.items():
            results[name].append(row)
        validation.append(prediction); alphas.append(dict(fold=fold, alpha=alpha))
    pd.DataFrame(alphas).to_csv(Path(args.results_root) / 'alphas.csv', index=False)
    if sorted(args.outer_folds) == list(range(5)):
        for name, rows in results.items():
            print(f'\n{name} nested outer-CV results', flush=True)
            save_cv_results(rows, Path(args.results_root) / name)
        paired = paired_comparison(validation, seed=args.seed)
        for name in ('pathology', 'ct'):
            expected = np.mean([a['val_cindex'] - b['val_cindex']
                               for a, b in zip(results['late_fusion'], results[name])])
            np.testing.assert_allclose(paired['fusion_minus_' + name]['delta_cindex'], expected, atol=1e-12)
        (Path(args.results_root) / 'paired_comparison.json').write_text(json.dumps(paired, indent=2))
        print('Paired outer-validation comparison: ' + json.dumps(paired), flush=True)
    else:
        for name, rows in results.items():
            pd.DataFrame(rows).to_csv(Path(args.results_root) / name / 'partial_metrics.csv', index=False)
        print('Partial outer-fold run; no five-fold summary generated', flush=True)
    (Path(args.results_root) / 'completed.json').write_text(json.dumps(dict(outer_folds=args.outer_folds,
        independent_fits=len(args.outer_folds)*2*(args.inner_folds+1), diagnostic_only=True), indent=2))


if __name__ == '__main__':
    main()
