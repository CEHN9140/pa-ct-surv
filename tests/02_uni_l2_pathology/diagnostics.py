"""Cox-only training diagnostics isolated from the production trainer."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sksurv.metrics import concordance_index_censored

from cox_utils import cox_loss
from dataset import cox_batch_sizes


def collect_validation(model, loader, device):
    training = model.training
    model.eval()
    rows = []
    try:
        with torch.no_grad():
            for features, event, time, case_ids in loader:
                output = model(features.to(device, non_blocking=True))
                risk = output[0] if isinstance(output, tuple) else output
                rows.append(pd.DataFrame({
                    'case_id': list(case_ids), 'dfs.month': time.cpu().numpy().reshape(-1),
                    'dfs.status': event.cpu().numpy().reshape(-1),
                    'risk_score': risk.cpu().numpy().reshape(-1)}))
    finally:
        model.train(training)
    frame = pd.concat(rows, ignore_index=True)
    frame['case_id'] = frame['case_id'].astype(str)
    if not frame['case_id'].is_unique or not np.isfinite(frame['risk_score']).all():
        raise ValueError('Validation requires unique patient IDs and finite risk scores')
    return frame


def rank_correlation(current, reference):
    if reference is None:
        return np.nan
    left = current.set_index('case_id')
    right = reference.set_index('case_id').reindex(left.index)
    if set(current['case_id']) != set(reference['case_id']) or not np.array_equal(
            left[['dfs.month', 'dfs.status']].to_numpy(),
            right[['dfs.month', 'dfs.status']].to_numpy()):
        raise ValueError('Validation patients or labels changed between epochs')
    if left['risk_score'].nunique() < 2 or right['risk_score'].nunique() < 2:
        return np.nan
    return float(left['risk_score'].corr(right['risk_score'], method='spearman'))


def record_validation(frame, epoch, initial, first, previous, history, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    risk = torch.as_tensor(frame['risk_score'].to_numpy(), dtype=torch.float32)
    time = torch.as_tensor(frame['dfs.month'].to_numpy(), dtype=torch.float32)
    event = torch.as_tensor(frame['dfs.status'].to_numpy(), dtype=torch.long)
    result = concordance_index_censored(event.numpy().astype(bool), time.numpy(), risk.numpy())
    sd = float(np.std(risk.numpy().astype(np.float64), ddof=0))
    standardized = ((risk.double() - risk.double().mean()) / sd) if sd > 0 else None
    metrics = {
        'epoch': epoch, 'val_loss': float(cox_loss(risk, time, event)),
        'val_cindex': float(result[0]), 'risk_mean': float(risk.double().mean()),
        'risk_std': sd,
        'spearman_vs_epoch0': rank_correlation(frame, initial),
        'spearman_vs_epoch1': rank_correlation(frame, first),
        'spearman_vs_previous': rank_correlation(frame, previous),
        'val_loss_unit_std': float(cox_loss(standardized, time, event))
        if standardized is not None else np.nan,
        'n_val': len(frame), 'n_comparable_pairs': int(sum(result[1:4])),
        'n_tied_risk_pairs': int(result[3]),
    }
    frame.to_csv(directory / f'val_predictions_epoch_{epoch:03d}.csv', index=False)
    history.append(metrics)
    pd.DataFrame(history).to_csv(directory / 'epoch_metrics.csv', index=False)
    return metrics


def train_with_diagnostics(model, train_loader, val_loader, optimizer, args, device,
                           fold, checkpoint_dir, diagnostic_dir):
    checkpoint_dir = Path(checkpoint_dir)
    history = []
    # DataLoader iteration draws a base seed even with no random transforms.
    devices = [device.index if device.index is not None else torch.cuda.current_device()] \
        if device.type == 'cuda' else []
    with torch.random.fork_rng(devices=devices):
        initial = collect_validation(model, val_loader, device)
    metrics = record_validation(initial, 0, initial, None, None, history, diagnostic_dir)
    print(f"Epoch 0 (untrained) | Val Loss: {metrics['val_loss']:.4f} | "
          f"Val C-index: {metrics['val_cindex']:.4f}", flush=True)
    first, previous = None, initial
    best_cindex, best_state, best_epoch, wait = -np.inf, None, None, 0
    batch_ends = set(np.cumsum(cox_batch_sizes(len(train_loader), args.cox_batch_size)).tolist())
    for epoch in range(1, args.num_epochs + 1):
        model.train()
        optimizer.zero_grad()
        losses, risks, times, events = [], [], [], []
        all_risks, all_times, all_events = [], [], []
        for index, (features, event, time, _) in enumerate(train_loader, start=1):
            output = model(features.to(device, non_blocking=True))
            risk = output[0] if isinstance(output, tuple) else output
            all_risks.append(risk.detach().cpu())
            all_times.append(time.detach().cpu())
            all_events.append(event.detach().cpu())
            risks.append(risk)
            times.append(time.to(device))
            events.append(event.to(device))
            if index in batch_ends:
                loss = cox_loss(torch.cat(risks), torch.cat(times), torch.cat(events))
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                losses.append(float(loss.detach().cpu()))
                risks, times, events = [], [], []
        train_cindex = concordance_index_censored(
            torch.cat(all_events).numpy().astype(bool), torch.cat(all_times).numpy(),
            torch.cat(all_risks).numpy().reshape(-1))[0]
        model.eval()
        frame = collect_validation(model, val_loader, device)
        if epoch == 1:
            first = frame
        metrics = record_validation(frame, epoch, initial, first, previous, history, diagnostic_dir)
        previous = frame
        val_cindex = metrics['val_cindex']
        print(f'Epoch {epoch}/{args.num_epochs} | Train Loss: {np.mean(losses):.4f} | '
              f'Train C-index: {train_cindex:.4f} | Val Loss: {metrics["val_loss"]:.4f} | '
              f'Val C-index: {val_cindex:.4f}', flush=True)
        if val_cindex > best_cindex:
            best_cindex, best_epoch, wait = val_cindex, epoch, 0
            best_state = {key: value.clone() for key, value in model.state_dict().items()}
            torch.save(model.state_dict(), checkpoint_dir / 'best_model.pth')
        else:
            wait += 1
        if args.patience > 0 and wait >= args.patience:
            print(f'Early stopping at epoch {epoch}', flush=True)
            break
    torch.save(model.state_dict(), checkpoint_dir / 'last_model.pth')
    with (checkpoint_dir / 'checkpoint_metadata.yaml').open('w') as handle:
        yaml.safe_dump({'fold': int(fold), 'best_epoch': int(best_epoch),
                        'last_epoch': int(epoch), 'best_val_cindex': float(best_cindex)}, handle)
    model.load_state_dict(best_state)
    print(f'Fold {fold} best C-index: {best_cindex:.4f}', flush=True)
    return model
