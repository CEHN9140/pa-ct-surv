"""Experiment-local Cox + PAMT training, preserving the main risk-group protocol."""

import numpy as np
import torch
import yaml
from sksurv.metrics import concordance_index_censored

from dataset import cox_batch_sizes
from loss import cox_loss


def train_pact_alignment(model, train_loader, val_loader, optimizer, args, device, fold,
                         checkpoint_dir):
    ends = set(np.cumsum(cox_batch_sizes(len(train_loader), args.cox_batch_size)).tolist())
    best_cindex, best_state, best_epoch, wait = -np.inf, None, None, 0
    for epoch in range(1, args.num_epochs + 1):
        model.train()
        optimizer.zero_grad()
        group_risks, group_times, group_events, group_alignment = [], [], [], []
        all_risks, all_times, all_events = [], [], []
        totals, coxes, alignments = [], [], []
        for index, (ct, pa, event, time, _) in enumerate(train_loader, start=1):
            output = model(ct.to(device), pa.to(device))
            risk, alignment = output[0], output[7]
            group_risks.append(risk)
            group_times.append(time.to(device))
            group_events.append(event.to(device))
            group_alignment.append(alignment)
            all_risks.extend(risk.detach().cpu().reshape(-1).tolist())
            all_times.extend(time.reshape(-1).tolist())
            all_events.extend(event.reshape(-1).tolist())
            if index in ends:
                cox = cox_loss(torch.cat(group_risks), torch.cat(group_times), torch.cat(group_events))
                align = torch.stack(group_alignment).mean()
                total = cox + args.alignment_weight * align
                if not torch.isfinite(total):
                    raise FloatingPointError(f'Nonfinite training loss at fold {fold}, epoch {epoch}')
                total.backward()
                optimizer.step()
                optimizer.zero_grad()
                totals.append(total.item())
                coxes.append(cox.item())
                alignments.append(align.item())
                group_risks, group_times, group_events, group_alignment = [], [], [], []
        train_cindex = concordance_index_censored(np.asarray(all_events).astype(bool),
                                                  np.asarray(all_times, dtype=np.float32),
                                                  np.asarray(all_risks, dtype=np.float32))[0]
        model.eval()
        risks, times, events, val_alignment = [], [], [], []
        with torch.no_grad():
            for ct, pa, event, time, _ in val_loader:
                output = model(ct.to(device), pa.to(device))
                risks.append(output[0].detach().cpu().reshape(-1))
                times.append(time.reshape(-1))
                events.append(event.reshape(-1))
                val_alignment.append(output[7].item())
            risk, time, event = torch.cat(risks), torch.cat(times), torch.cat(events)
            val_cox = cox_loss(risk.to(device), time.to(device), event.to(device)).item()
        val_cindex = concordance_index_censored(event.numpy().astype(bool), time.numpy(), risk.numpy())[0]
        mean_alignment = float(np.mean(val_alignment))
        val_total = val_cox + args.alignment_weight * mean_alignment
        if not np.isfinite([val_cox, val_cindex, mean_alignment, val_total]).all():
            raise FloatingPointError(f'Nonfinite validation result at fold {fold}, epoch {epoch}')
        # Keep Val Loss = whole-fold fused Cox, matching every existing baseline.
        print(f'Epoch {epoch}/{args.num_epochs} | Train Loss: {np.mean(totals):.4f} '
              f'(Fused Cox: {np.mean(coxes):.4f}, PAMT Align: {np.mean(alignments):.4f}) | '
              f'Train C-index: {train_cindex:.4f} | Val Loss: {val_cox:.4f} | '
              f'Val C-index: {val_cindex:.4f} | Val Align: {mean_alignment:.4f} | '
              f'Val Total Loss: {val_total:.4f}')
        if val_cindex > best_cindex:
            best_cindex, best_epoch, wait = float(val_cindex), epoch, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            torch.save(best_state, checkpoint_dir / 'best_model.pth')
        else:
            wait += 1
        if args.patience > 0 and wait >= args.patience:
            print(f'Early stopping at epoch {epoch}')
            break
    model.load_state_dict(best_state)
    with (checkpoint_dir / 'checkpoint_metadata.yaml').open('w') as handle:
        yaml.safe_dump({'fold': fold, 'best_epoch': best_epoch, 'last_epoch': epoch,
                        'best_val_cindex': best_cindex, 'alignment_weight': args.alignment_weight,
                        'alignment_topk': args.alignment_topk}, handle)
    print(f'Fold {fold} best C-index: {best_cindex:.4f}')
    return model
