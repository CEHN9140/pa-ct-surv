from pathlib import Path

import pandas as pd
import torch
from sksurv.metrics import concordance_index_censored

from loss import cox_loss, survival_loss


def discrete_nll_loss(hazards, survival, disc_label, event, eps=1e-7):
    """Discrete-time survival NLL with event=1 and censoring=0 labels."""
    disc_label = disc_label.view(-1, 1).long()
    event = event.view(-1, 1).float()
    censoring = 1.0 - event
    survival_padded = torch.cat(
        [torch.ones_like(censoring), survival], dim=1
    )
    survival_before = torch.gather(survival_padded, 1, disc_label)
    survival_through = torch.gather(survival_padded, 1, disc_label + 1)
    event_hazard = torch.gather(hazards, 1, disc_label)
    event_loss = -(
        torch.log(survival_before.clamp_min(eps))
        + torch.log(event_hazard.clamp_min(eps))
    )
    censor_loss = -torch.log(survival_through.clamp_min(eps))
    return (event * event_loss + censoring * censor_loss).mean()


def evaluate_survival(model, train_loader, val_loader, device, save_dir=None,
                      loss_type="cox", rank_weight=1.0):
    """Evaluate with dropout disabled; optionally export patient predictions."""
    model.eval()
    metrics = {}
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
    is_nll = getattr(model, "time_bins", None) is not None
    if (is_nll and loss_type == "cox_rank") or (not is_nll and loss_type == "nll"):
        raise ValueError("The selected survival loss does not match the model output")
    with torch.no_grad():
        for split, loader in (("train", train_loader), ("val", val_loader)):
            risks, times, events, nll_losses, case_ids = [], [], [], [], []
            for batch in loader:
                if len(batch) == 4:
                    feat, event, time, case_id = batch
                    output = model(feat.to(device))
                else:
                    ct, pa, event, time, case_id = batch[:5]
                    output = model(ct.to(device)) if isinstance(pa, list) else model(ct.to(device), pa.to(device))
                case_ids.extend(case_id)
                risk = output[0] if isinstance(output, tuple) else output
                risks.append(risk.detach().cpu().reshape(-1))
                times.append(time.detach().cpu().reshape(-1))
                events.append(event.detach().cpu().reshape(-1))
                if is_nll:
                    hazards, survival = output[3:5]
                    labels = torch.bucketize(time.to(device), model.time_bins).clamp(max=hazards.size(1) - 1)
                    nll_losses.append(float(discrete_nll_loss(hazards, survival, labels, event.to(device))) * len(time))
            risk, time, event = torch.cat(risks), torch.cat(times), torch.cat(events)
            loss = sum(nll_losses) / len(time) if is_nll else float(survival_loss(
                risk.to(device), time.to(device), event.to(device), loss_type, rank_weight
            ))
            cindex = concordance_index_censored(event.numpy().astype(bool), time.numpy(), risk.numpy())[0]
            metrics.update({f"{split}_loss": loss, f"{split}_cindex": float(cindex)})
            if save_dir is not None:
                pd.DataFrame({"case_id": case_ids, "dfs.month": time.numpy(),
                              "dfs.status": event.numpy(), "risk_score": risk.numpy()
                              }).to_csv(save_dir / f"{split}_predictions.csv", index=False)
    if save_dir is not None:
        from visualization.survival_plots import analyze_fold, read_predictions

        extra = analyze_fold(read_predictions(save_dir / "train_predictions.csv"),
                             read_predictions(save_dir / "val_predictions.csv"), save_dir)
        metrics.update({key: value for key, value in extra.items() if key not in metrics})
    return metrics
