"""Stage3: matched Raw/div40 ABMIL parameterizations, fixed optimizer steps."""

import argparse
import copy
import hashlib
import json
import os
import platform
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent.parent / 'pathology_stage2'))

from dataset import cox_batch_sizes, load_path_feature
from diagnose import compensated_model, joint_gradients, sha256
from model.abmil_cox import ABMIL
from utils import seed_everything
from scale_eval import evaluate, compare


def create_groups(base):
    models, optimizers = {}, {}
    for group in ('A', 'B', 'C', 'D'):
        model = compensated_model(base, 40.) if group in ('C', 'D') else copy.deepcopy(base)
        weight = model.compression[0].weight
        optimizer = torch.optim.Adam([
            dict(params=[weight], lr=4e-3 if group == 'D' else 1e-4),
            dict(params=[p for p in model.parameters() if p is not weight], lr=1e-4),
        ], weight_decay=0., betas=(.9, .999), eps=1e-8, foreach=False)
        models[group], optimizers[group] = model, optimizer
    return models, optimizers


def batch_schedule(indices, cox_batch_size, steps, seed):
    if steps < 1 or cox_batch_size < 2 or len(indices) < 2:
        raise ValueError('Invalid update budget or Cox risk set')
    # Independent streams derived from the one seed; evaluation never consumes them.
    children = np.random.SeedSequence(seed).spawn(2)
    order_rng, dropout_rng = (np.random.default_rng(child) for child in children)
    result, epoch = [], 0
    while len(result) < steps:
        epoch += 1
        order = order_rng.permutation(indices)
        start = 0
        for batch, size in enumerate(cox_batch_sizes(len(indices), cox_batch_size), start=1):
            result.append(dict(step=len(result)+1, epoch=epoch, epoch_batch=batch,
                               indices=order[start:start+size],
                               dropout_seed=int(dropout_rng.integers(0, 2**31))))
            start += size
            if len(result) == steps:
                break
    return result


def train_step(model, optimizer, bags, time_values, event, condition, dropout_seed, group):
    _, loss = joint_gradients(model, bags, time_values, event, condition, dropout_seed)
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    gradients = {name: float(p.grad.norm()) for name, p in model.named_parameters()}
    optimizer.step()
    rows = []
    for name, p in model.named_parameters():
        change = p.detach() - before[name]
        divisor = 40. if group != 'A' and name == 'compression.0.weight' else 1.
        rows.append(dict(parameter=name, data_gradient_norm=gradients[name],
                         parameter_norm=float(before[name].norm()), update_norm=float(change.norm()),
                         relative_update=float(change.norm() / (before[name].norm() + 1e-12)),
                         effective_raw_update_norm=float(change.norm()) / divisor,
                         learning_rate=4e-3 if group == 'D' and name == 'compression.0.weight' else 1e-4,
                         batch_cox_loss=float(loss)))
    if not all(torch.isfinite(p).all() for p in model.parameters()):
        raise RuntimeError('Non-finite parameters after Adam update')
    return rows


def run_fold(base, bags, samples, fold, directory, seed, max_steps=60, cox_batch_size=64):
    directory = Path(directory)
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError(f'Refusing to overwrite fold: {directory}')
    directory.mkdir(parents=True, exist_ok=True)
    train_indices = np.flatnonzero(samples.split.ne(fold))
    val_indices = np.flatnonzero(samples.split.eq(fold))
    if set(train_indices) & set(val_indices) or not len(val_indices):
        raise ValueError('Invalid saved outer folds')
    models, optimizers = create_groups(base)
    initial_hashes = {group: hashlib.sha256(b''.join(
        value.detach().cpu().numpy().tobytes() for value in model.state_dict().values())).hexdigest()
        for group, model in models.items()}
    schedule = batch_schedule(train_indices, cox_batch_size, max_steps, seed)
    checkpoint_steps = sorted({s for s in (0, 1, 5, 10, 20, 40, 60) if s <= max_steps} | {max_steps})
    schedule_rows = []
    for entry in schedule:
        batch = samples.iloc[entry['indices']]
        schedule_rows.append(dict(fold=fold, **{k:v for k,v in entry.items() if k != 'indices'},
                                  patients=len(batch), events=int(batch.event.sum()),
                                  patient_order_sha256=hashlib.sha256('|'.join(batch.pa_id).encode()).hexdigest()))
    pd.DataFrame(schedule_rows).to_csv(directory/'batch_schedule.csv', index=False)
    metrics, updates, differences = [], [], []
    start_time = time.monotonic()

    def checkpoint(step):
        outputs = {}
        for group, model in models.items():
            destination = directory/'private'/group/f'step_{step:03d}'
            condition = 'raw' if group == 'A' else 'div40'
            for split, indices in [('train', train_indices), ('val', val_indices)]:
                row, values = evaluate(model, bags, samples, indices, condition, destination/f'{split}_predictions.csv')
                metrics.append(dict(fold=fold, group=group, step=step, split=split, **row))
                outputs[(group, split)] = values
            torch.save(dict(model=model.state_dict(), optimizer=optimizers[group].state_dict(),
                            step=step, group=group, fold=fold, seed=seed), destination/'model.pth')
            a, b = metrics[-2], metrics[-1]
            print(f"Fold {fold} | Group {group} | Step {step}/{max_steps} | Train Loss: {a['cox_loss']:.6f} | "
                  f"Train C-index: {a['cindex']:.6f} | Val Loss: {b['cox_loss']:.6f} | "
                  f"Val C-index: {b['cindex']:.6f}", flush=True)
        current = []
        for split in ('train', 'val'):
            for group in ('B', 'C', 'D'):
                current.extend(compare(outputs[('A', split)], outputs[(group, split)],
                                       fold=fold, step=step, split=split, comparison=f'A_{group}'))
        if step == 0 and not all(row['within_tolerance'] for row in current
                                 if row['comparison'] in ('A_C', 'A_D')):
            raise RuntimeError('A/C/D initial function equivalence failed; no training allowed')
        differences.extend(current)
        pd.DataFrame(metrics).to_csv(directory/'metrics.csv', index=False)
        pd.DataFrame(differences).to_csv(directory/'function_differences.csv', index=False)
        if step == 0:
            print(f'Fold {fold}: A/C/D initial Train+Val eval equivalence passed; no labels used for selection', flush=True)

    checkpoint(0)
    for entry in schedule:
        indices = entry['indices']
        batch = [bags[i] for i in indices]
        group_frame = samples.iloc[indices]
        dtype = next(base.parameters()).dtype
        t = torch.tensor(group_frame.time.to_numpy(), dtype=dtype)
        e = torch.tensor(group_frame.event.to_numpy(), dtype=dtype)
        for group in ('A', 'B', 'C', 'D'):
            rows = train_step(models[group], optimizers[group], batch, t, e,
                              'raw' if group == 'A' else 'div40', entry['dropout_seed'], group)
            updates.extend(dict(fold=fold, group=group, step=entry['step'], **row) for row in rows)
        pd.DataFrame(updates).to_csv(directory/'gradient_update.csv', index=False)
        print(f"Fold {fold}: all4 groups completed step {entry['step']}/{max_steps}; "
              f"risk set={len(indices)}, events={int(e.sum())}, elapsed={time.monotonic()-start_time:.1f}s", flush=True)
        if entry['step'] in checkpoint_steps:
            checkpoint(entry['step'])
    completion = dict(fold=fold, groups=['A','B','C','D'], optimizer_steps_each=max_steps,
                      checkpoint_steps=checkpoint_steps, train_patients=len(train_indices),
                      val_patients=len(val_indices), initial_function_equivalence=True,
                      shared_schedule=True, weight_decay=0., early_stopping=False,
                      validation_used_for_selection=False, elapsed_seconds=time.monotonic()-start_time)
    completion['initial_parameter_sha256'] = initial_hashes
    (directory/'completed.json').write_text(json.dumps(completion, indent=2))
    return completion


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', type=Path, default=ROOT/'data/trident/features_uni_v1/all_label_roi64.csv')
    parser.add_argument('--output', type=Path, default=Path(__file__).parent/'results')
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError('Use a fresh --output: never overwrite earlier results')
    args.output.mkdir(parents=True, exist_ok=True)
    status = dict(status='running', folds=[2,4], groups=['A','B','C','D'], steps_per_fit=60, pid=os.getpid())
    (args.output/'status.json').write_text(json.dumps(status, indent=2))
    try:
        torch.set_num_threads(4)
        seed_everything(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        source_hash = sha256(args.csv)
        samples = pd.read_csv(args.csv, dtype={'pa_id':str, 'ct_id':str})
        if sha256(args.csv) != source_hash:
            raise RuntimeError('CSV changed during read')
        if len(samples) != 721 or samples.event.sum() != 251 or samples.pa_id.duplicated().any():
            raise ValueError('Expected saved 721-patient, 251-event UNI cohort')
        if not samples.split.isin(range(5)).all() or not samples.event.isin([0,1]).all():
            raise ValueError('Invalid folds/events')
        if not np.isfinite(samples.time).all() or (samples.time < 0).any():
            raise ValueError('Invalid survival times')
        print('Stage3 | complete UNI1024 WSI | folds2,4 | A/B/C/D | fixed60 steps | Adam WD0', flush=True)
        files = [Path(p) for p in samples.pa_path]
        hashes, bags = [], []
        for i, path in enumerate(files):
            hashes.append(sha256(path))
            bags.append(load_path_feature(path, 1024))
            if (i+1) % 100 == 0:
                print(f'Read-only CPU feature cache: {i+1}/{len(files)} WSI', flush=True)
        manifest = dict(csv=str(args.csv.resolve()), csv_sha256=source_hash,
                        features_sha256=hashlib.sha256(''.join(hashes).encode()).hexdigest(),
                        patches=sum(len(x) for x in bags), seed=args.seed,
                        torch_version=str(torch.__version__), device=args.device,
                        hardware=dict(cpu=platform.processor() or platform.machine(), cpu_count=os.cpu_count(),
                                      torch_cpu_threads=torch.get_num_threads(),
                                      gpu=torch.cuda.get_device_name(args.device) if str(args.device).startswith('cuda') else None,
                                      cuda_version=torch.version.cuda, cudnn_version=torch.backends.cudnn.version()),
                        settings=dict(dropout=.25, lr=1e-4, d_first_weight_lr=4e-3,
                                      adam_betas=[.9,.999], adam_eps=1e-8, weight_decay=0.,
                                      cox_batch_size=64, max_steps=60, early_stopping=False,
                                      checkpoint_selection=False, tf32=False),
                        code_sha256={str(p.relative_to(ROOT)):sha256(p) for p in [
                            Path(__file__), Path(__file__).with_name('scale_eval.py'),
                            Path(__file__).with_name('scale_analysis.py'),
                            Path(__file__).parent.parent/'pathology_stage2/diagnose.py',
                            ROOT/'model/abmil_cox.py', ROOT/'loss/survival_loss.py', ROOT/'dataset.py']})
        (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2))
        completed = []
        for fold in (2,4):
            seed_everything(args.seed)
            base = ABMIL(dropout=.25).to(args.device)
            manifest.setdefault('initial_parameter_sha256_by_fold', {})[str(fold)] = hashlib.sha256(b''.join(
                value.detach().cpu().numpy().tobytes() for value in base.state_dict().values())).hexdigest()
            (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2))
            print(f"Fold {fold}: Train {int(samples.split.ne(fold).sum())}; Val {int(samples.split.eq(fold).sum())}", flush=True)
            completed.append(run_fold(base, bags, samples, fold, args.output/f'fold_{fold}', args.seed))
        if hashes != [sha256(p) for p in files] or sha256(args.csv) != source_hash:
            raise RuntimeError('Input features or CSV changed')
        for path, digest in manifest['code_sha256'].items():
            if sha256(ROOT/path) != digest:
                raise RuntimeError(f'Code changed during experiment: {path}')
        manifest['source_sha256_unchanged'] = True
        (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2))
        from scale_analysis import analyze
        analyze(args.output)
        status.update(status='completed', completed=completed)
        print('Stage3 completed all8 fixed-budget fits; aggregate report and figures saved.', flush=True)
    except Exception as exc:
        status.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        (args.output/'status.json').write_text(json.dumps(status, indent=2))


if __name__ == '__main__':
    main()
