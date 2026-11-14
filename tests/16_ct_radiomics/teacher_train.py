"""Joint pathology/Radiomics teacher, with training-only epoch selection."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import joblib
import numpy as np
import pandas as pd
import torch
from sksurv.metrics import concordance_index_censored

from dataset import build_dataloader, cox_batch_sizes
from loss import cox_loss
from utils import seed_everything, save_run_config, save_cv_results
from visualization.survival_plots import analyze_fold
from teacher_data import (RadiomicsPreprocessor, load_teacher_inputs, teacher_partitions,
                          save_preprocessor, file_digest)
from teacher_model import RadiomicsTeacher


def cindex(frame):
    return float(concordance_index_censored(frame['dfs.status'].astype(bool),
                                           frame['dfs.month'], frame.risk_score)[0])


def prediction_frame(ids, times, events, risks):
    return pd.DataFrame({'case_id': list(ids), 'dfs.month': times,
                         'dfs.status': events, 'risk_score': risks})


def full_cox(frame):
    return float(cox_loss(torch.tensor(frame.risk_score.to_numpy(), dtype=torch.float64),
                          torch.tensor(frame['dfs.month'].to_numpy(), dtype=torch.float64),
                          torch.tensor(frame['dfs.status'].to_numpy())))


@torch.inference_mode()
def predict_teacher(model, dataset, preprocessor, indices, args, device):
    model.eval()
    loader = build_dataloader(dataset.with_preprocessor(preprocessor), indices,
                              batch_size=args.cox_batch_size, num_workers=args.num_workers,
                              pin_memory=device.type == 'cuda')
    frames = []
    for ct, pa, event, time, ids in loader:
        risk = model(ct.to(device), pa.to(device)).cpu().numpy()
        if not np.isfinite(risk).all():
            raise ValueError('Nonfinite teacher prediction')
        frames.append(prediction_frame(ids, time.numpy(), event.numpy(), risk))
    return pd.concat(frames, ignore_index=True)


def fit_teacher(dataset, fit, stop, args, device, directory, fixed_epochs=None):
    fit = np.asarray(fit, dtype=int)
    if (stop is None) != (fixed_epochs is not None):
        raise ValueError('Use either STOP selection or a fixed epoch count')
    if len(set(fit)) != len(fit) or len(fit) < 2 or args.cox_batch_size < 2:
        raise ValueError('Cox training requires at least two unique FIT patients and batch size >=2')
    if stop is not None and (not len(stop) or set(fit) & set(stop)):
        raise ValueError('FIT and STOP must be nonempty and disjoint')
    epochs = args.num_epochs if fixed_epochs is None else int(fixed_epochs)
    if epochs < 1:
        raise ValueError('Need at least one training epoch')
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    prep = (None if args.fusion_type == 'pa_only' else RadiomicsPreprocessor().fit(
        dataset.radiomics.iloc[fit], dataset.samples.iloc[fit].case_id.tolist()))
    save_preprocessor(prep, directory)
    seed_everything(args.seed)
    model = RadiomicsTeacher(None if prep is None else prep.group_indices, args.dropout, args.fusion_type).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loader = build_dataloader(dataset.with_preprocessor(prep), fit, batch_size=1, shuffle=True,
                              num_workers=args.num_workers, pin_memory=device.type == 'cuda')
    sizes = cox_batch_sizes(len(fit), args.cox_batch_size)
    history, best_score, selected_epoch, stale = [], -np.inf, 0, 0
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        risks, times, events, frames, losses = [], [], [], [], []
        group_index = 0
        for ct, pa, event, time, ids in loader:
            risk = model(ct.to(device), pa.to(device))
            risks.append(risk)
            times.append(time.to(device))
            events.append(event.to(device))
            frames.append(prediction_frame(ids, time.numpy(), event.numpy(), risk.detach().cpu().numpy()))
            if len(risks) == sizes[group_index]:
                loss = cox_loss(torch.cat(risks), torch.cat(times), torch.cat(events))
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite fused Cox loss')
                loss.backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                losses.append(float(loss.detach()))
                risks, times, events = [], [], []
                group_index += 1
        if risks or group_index != len(sizes):
            raise RuntimeError('Cox tail was not included in the final update')
        row = dict(epoch=epoch, train_loss=float(np.mean(losses)),
                   train_cindex=cindex(pd.concat(frames, ignore_index=True)))
        if stop is not None:
            validation = predict_teacher(model, dataset, prep, stop, args, device)
            row.update(val_loss=full_cox(validation), val_cindex=cindex(validation))
            improved = row['val_cindex'] > best_score
            if improved:
                best_score, selected_epoch, stale = row['val_cindex'], epoch, 0
            else:
                stale += 1
        else:
            improved, selected_epoch = True, epoch
        if improved:
            torch.save({name: value.detach().cpu().clone() for name, value in model.state_dict().items()},
                       directory/'best_model.pth')
        history.append(row)
        pd.DataFrame(history).to_csv(directory/'training_history.csv', index=False)
        label = 'STOP' if stop is not None else 'fixed'
        print(f'{directory.name} {label} epoch {epoch}/{epochs}: ' + ' | '.join(
            f'{key}={value:.4f}' for key, value in row.items() if key != 'epoch'), flush=True)
        if stop is not None and stale >= args.patience:
            break
    metadata = dict(selected_epoch=selected_epoch, epochs_completed=epoch,
                    training_risk_set_sizes=sizes, fit_n=len(fit),
                    stop_n=0 if stop is None else len(stop), dropout=args.dropout,
                    fusion_type=args.fusion_type, seed=args.seed,
                    preprocessing=None if prep is None else prep.metadata(),
                    initialization='All learned modules newly initialized; no downstream checkpoint',
                    loss=('PA-only Cox' if prep is None else 'Fused Cox only') + ', Breslow, divided by events',
                    epoch_policy='STOP best C-index' if stop is not None else 'Fixed before outer evaluation',
                    train_metric='Online train-mode epoch risks; final metrics separately reevaluated in eval mode')
    (directory/'checkpoint_metadata.json').write_text(json.dumps(metadata, indent=2))
    model.load_state_dict(torch.load(directory/'best_model.pth', map_location=device, weights_only=True), strict=True)
    return model.eval(), prep, metadata


def load_teacher(directory, device):
    directory = Path(directory)
    metadata = json.loads((directory/'checkpoint_metadata.json').read_text())
    prep = None if metadata['preprocessing'] is None else joblib.load(directory/'preprocessor.joblib')
    if prep is not None and prep.metadata() != metadata['preprocessing']:
        raise ValueError('Checkpoint and preprocessing layout differ')
    model = RadiomicsTeacher(None if prep is None else prep.group_indices,
                             metadata['dropout'], metadata['fusion_type']).to(device)
    model.load_state_dict(torch.load(directory/'best_model.pth', map_location=device, weights_only=True), strict=True)
    return model.eval(), prep, metadata


def run(args):
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; use --device cpu for CPU')
    dataset, sources = load_teacher_inputs(args.cohort_csv, args.features_csv)
    samples = dataset.samples
    if len(samples) != 721 or int(samples.event.sum()) != 251 or set(samples.split) != set(range(5)):
        raise ValueError('This protocol requires the unchanged721-patient/251-event five-fold cohort')
    folds = [(np.flatnonzero(samples.split.to_numpy() != fold),
              np.flatnonzero(samples.split.to_numpy() == fold)) for fold in range(5)]
    roles = [teacher_partitions(samples, train, args.seed) for train, _ in folds]
    if args.audit_only:
        if args.fusion_type == 'pa_only':
            print('PA-only: paired cohort audited; no CT feature filtering/scaling used.', flush=True)
            return
        for fold, ((train, _), partitions) in enumerate(zip(folds, roles)):
            for name, fit in [(f'inner_{i}', part[0]) for i, part in enumerate(partitions)] + [('outer', train)]:
                prep = RadiomicsPreprocessor().fit(dataset.radiomics.iloc[fit], samples.iloc[fit].case_id.tolist())
                print(f'Audit fold{fold}/{name}: FIT={len(fit)}, selected={len(prep.selected_)}, '
                      f'group_dims={[len(group) for group in prep.group_indices]}', flush=True)
        return
    experiment = Path(__file__).resolve().parent
    args.results_root = args.results_root or str(experiment/'teacher_results'/args.fusion_type)
    args.checkpoint_root = args.checkpoint_root or str(experiment/'teacher_checkpoints'/args.fusion_type)
    for path in (args.results_root, args.checkpoint_root):
        if Path(path).exists():
            raise FileExistsError(f'Refusing to overwrite teacher run: {path}')
    pa_only = args.fusion_type == 'pa_only'
    save_run_config(args, 'pact', args.fusion_type, excluded=('audit_only',), extra=dict(
        source_sha256=sources, ct_tokens='Unused' if pa_only else 'Seven Original Radiomics families; empty families masked',
        pathology='EAGLE Top25 Virchow2-CLS1280; patch L2 before ABMIL',
        attention=('PA Q, Radiomics K/V, single head256, attention-weight dropout0'
                   if args.fusion_type == 'crossattention' else 'No cross-attention'),
        supervision=('PA-only Cox' if pa_only else 'Fused Cox only') + '; all learned modules newly initialized',
        validation='Saved outer5; inner3 FIT/STOP/HELD; median STOP-selected epochs; outer evaluated once',
        scope=('Matched pathology-only comparison baseline' if pa_only else
               'Experimental Mask-assisted Radiomics teacher; no student implementation')))
    metrics, oof = [], []
    for fold, ((train, val), partitions) in enumerate(zip(folds, roles)):
        root = Path(args.checkpoint_root)/f'fold_{fold}'
        root.mkdir()
        selected, role_rows = [], []
        print(f'\n===== Outer fold {fold}: Train={len(train)}, Val={len(val)} =====', flush=True)
        for inner, (fit, stop, held) in enumerate(partitions):
            for role, indices in [('FIT', fit), ('STOP', stop), ('HELD', held)]:
                role_rows.extend(dict(case_id=samples.iloc[i].case_id, inner=inner, role=role) for i in indices)
            model, prep, info = fit_teacher(dataset, fit, stop, args, device, root/f'inner_{inner}')
            selected.append(info['selected_epoch'])
            del model, prep
        pd.DataFrame(role_rows).to_csv(root/'inner_roles.csv', index=False)
        epochs = int(np.median(selected))
        print(f'Outer fold{fold} epochs fixed to median{selected}={epochs}; fresh joint fit', flush=True)
        model, prep, info = fit_teacher(dataset, train, None, args, device, root/'outer', fixed_epochs=epochs)
        train_pred = predict_teacher(model, dataset, prep, train, args, device)
        val_pred = predict_teacher(model, dataset, prep, val, args, device)
        destination = Path(args.results_root)/f'fold_{fold}'
        extra = analyze_fold(train_pred, val_pred, destination)
        train_pred.to_csv(destination/'train_predictions.csv', index=False)
        save_preprocessor(prep, destination)
        (destination/'selection.json').write_text(json.dumps(dict(inner_best_epochs=selected,
            outer_epochs=epochs, outer_validation_used_for_selection=False), indent=2))
        row = dict(fold=fold, **extra, train_loss=full_cox(train_pred), val_loss=full_cox(val_pred),
                   selected_epoch=epochs, n_selected=0 if prep is None else len(prep.selected_))
        metrics.append(row)
        oof.append(val_pred.assign(fold=fold))
        print(f'Final fold{fold}: Train C-index={row["train_cindex"]:.4f}, '
              f'Val C-index={row["val_cindex"]:.4f}', flush=True)
        pd.DataFrame(metrics).to_csv(Path(args.results_root)/'partial_metrics.csv', index=False)
        del model, prep
    save_cv_results(metrics, args.results_root)
    pd.concat(oof, ignore_index=True).to_csv(Path(args.results_root)/'oof_predictions.csv', index=False)
    for path, digest in sources.items():
        if file_digest(path) != digest:
            raise ValueError(f'Source changed during training: {path}')
    (Path(args.results_root)/'completed.json').write_text(json.dumps(dict(
        patients=len(samples), events=int(samples.event.sum()), outer_folds=5, neural_fits=20,
        source_hashes_unchanged=True, outer_validation_used_for_selection=False), indent=2))


def main():
    experiment = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cohort_csv', default=str(experiment.parents[1]/'data/eagle/top25/all_label_roi64.csv'))
    parser.add_argument('--features_csv', default=str(experiment/'output/features.csv'))
    parser.add_argument('--fusion_type', choices=['crossattention', 'concat', 'pa_only'], default='crossattention')
    parser.add_argument('--results_root', default=None)
    parser.add_argument('--checkpoint_root', default=None)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--dropout', type=float, default=.25)
    parser.add_argument('--cox_batch_size', type=int, default=32)
    parser.add_argument('--num_epochs', type=int, default=30)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--audit_only', action='store_true')
    args = parser.parse_args()
    if (args.lr <= 0 or args.weight_decay < 0 or not 0 <= args.dropout < 1 or
            args.cox_batch_size < 2 or min(args.num_epochs, args.patience) < 1 or args.num_workers < 0):
        parser.error('Invalid optimizer, dropout, Cox batch or training settings')
    run(args)


if __name__ == '__main__':
    main()
