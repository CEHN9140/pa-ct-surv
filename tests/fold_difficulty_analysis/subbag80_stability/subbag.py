"""Exact eval-mode patch subsetting; projection/attention logits are patch independent."""
import hashlib
import numpy as np
import torch


def subset_indices(n,patient,repeat):
    seed=int.from_bytes(hashlib.sha256(f'subbag80:123:{patient}:{repeat}'.encode()).digest()[:8],'little')
    return np.sort(np.random.default_rng(seed).choice(n,int(np.ceil(.8*n)),replace=False))


@torch.no_grad()
def cached_predictions(model,x,patient,repeats=30):
    if any(m.training for m in model.modules()) or model.l2_normalize:raise ValueError('Raw UNI and eval mode required')
    h=model.mil.compression(x);logits=model.mil.attention(h).reshape(-1)
    pools=[logits.softmax(0)@h]
    for repeat in range(repeats):
        idx=torch.as_tensor(subset_indices(len(x),patient,repeat),device=x.device)
        pools.append(logits[idx].softmax(0)@h[idx])
    return model.mil.risk_head(model.mil.post_compression(torch.stack(pools))).reshape(-1).cpu().numpy()
