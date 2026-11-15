"""Fixed-checkpoint full-patient evaluation and paired function diagnostics."""

import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sksurv.metrics import concordance_index_censored

from loss.survival_loss import cox_loss


def evaluate(model, bags, samples, indices, condition, prediction_path):
    model.eval()
    device = next(model.parameters()).device
    pools, embeddings, risks, entropies, maxima = [], [], [], [], []
    captured = {}
    hook = model.post_compression.register_forward_pre_hook(
        lambda module, inputs: captured.update(pool=inputs[0]))
    try:
        with torch.no_grad():
            for index in indices:
                x = bags[index].to(device)
                if condition != 'raw':
                    x = x / 40.
                risk, embedding, weights = model(x)
                risks.append(risk.cpu())
                pools.append(captured['pool'].cpu())
                embeddings.append(embedding.cpu())
                weights = weights.reshape(-1)
                entropy = -(weights * weights.clamp_min(1e-30).log()).sum()
                entropies.append(float(entropy / math.log(len(weights))) if len(weights) > 1 else 0.)
                maxima.append(float(weights.max()))
    finally:
        hook.remove()
    r, pool, embedding = torch.cat(risks), torch.cat(pools), torch.cat(embeddings)
    group = samples.iloc[indices]
    time = torch.tensor(group.time.to_numpy(), dtype=r.dtype)
    event = torch.tensor(group.event.to_numpy(), dtype=r.dtype)
    if not torch.isfinite(r).all():
        raise RuntimeError('Non-finite evaluation risk')
    metrics = dict(patients=len(indices), events=int(event.sum()),
                   cox_loss=float(cox_loss(r, time, event)),
                   cindex=float(concordance_index_censored(event.bool().numpy(), time.numpy(), r.numpy())[0]),
                   risk_sd=float(r.std(unbiased=False)),
                   attention_normalized_entropy_mean=float(np.mean(entropies)),
                   attention_normalized_entropy_sd=float(np.std(entropies)),
                   attention_max_weight_mean=float(np.mean(maxima)),
                   pool512_patient_variance_mean=float(pool.var(dim=0, unbiased=False).mean()),
                   post256_patient_variance_mean=float(embedding.var(dim=0, unbiased=False).mean()))
    prediction_path = Path(prediction_path)
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(dict(case_id=group.pa_id.to_numpy(), event=group.event.to_numpy(),
                      time=group.time.to_numpy(), risk=r.numpy())).to_csv(prediction_path, index=False)
    return metrics, dict(risk=r, pool512=pool, post256=embedding)


def compare(left, right, **metadata):
    rows = []
    for feature, x in left.items():
        y = right[feature]
        diff = x - y
        row = dict(**metadata, feature=feature, rmse=float(diff.square().mean().sqrt()),
                   relative_l2=float(diff.norm() / (x.norm() + 1e-12)),
                   within_tolerance=bool(torch.allclose(x, y, atol=2e-6, rtol=2e-4)))
        if feature == 'risk':
            defined = bool(x.std(unbiased=False) > 0 and y.std(unbiased=False) > 0)
            row.update(spearman=float(spearmanr(x.numpy(), y.numpy()).statistic) if defined else 0.,
                       spearman_defined=defined,
                       centered_rmse=float((diff - diff.mean()).square().mean().sqrt()))
        rows.append(row)
    return rows
