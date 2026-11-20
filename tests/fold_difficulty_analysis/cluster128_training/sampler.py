"""Capacity-balanced no-replacement sampling with a private stable RNG."""
import hashlib
import json
import numpy as np


def allocate(capacities,total=128):
    caps=np.asarray(capacities)
    if caps.ndim!=1 or not np.issubdtype(caps.dtype,np.integer) or (caps<0).any() or total<1:
        raise ValueError('Require nonnegative integer capacities and positive budget')
    quotas=np.zeros(len(caps),dtype=np.int64);remaining=min(int(total),int(caps.sum()))
    while remaining:
        active=np.flatnonzero(quotas<caps)
        share,remainder=divmod(remaining,len(active))
        if share:
            add=np.minimum(share,caps[active]-quotas[active]);quotas[active]+=add;remaining-=int(add.sum())
        else:quotas[active[:remainder]]+=1;remaining=0
    return quotas


def select_indices(labels,k,seed,role,step,patient_id):
    labels=np.asarray(labels)
    if labels.ndim!=1 or not len(labels) or not np.issubdtype(labels.dtype,np.integer) or (labels<0).any() or k<1:
        raise ValueError('Require nonempty nonnegative integer cluster labels and k>0')
    if len(labels)<=k:return np.arange(len(labels),dtype=np.int64)
    groups,counts=np.unique(labels,return_counts=True);quotas=allocate(counts,k)
    payload=json.dumps([int(seed),str(role),int(step),str(patient_id)],separators=(',',':')).encode()
    rng=np.random.default_rng(int.from_bytes(hashlib.sha256(payload).digest()[:16],'little'))
    selected=np.concatenate([rng.choice(np.flatnonzero(labels==c),int(n),replace=False) for c,n in zip(groups,quotas)])
    return np.sort(selected)
