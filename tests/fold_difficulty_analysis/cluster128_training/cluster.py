"""Independent per-WSI clustering; raw model inputs are never normalized."""
import hashlib
import json
from pathlib import Path
import os
import numpy as np
from sklearn.cluster import MiniBatchKMeans
import sklearn


def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def cluster_labels(raw):
    raw=np.asarray(raw)
    if raw.ndim!=2 or not len(raw) or raw.shape[1]!=1024 or not np.isfinite(raw).all():
        raise ValueError('Expected finite raw UNI [N,1024]')
    model=MiniBatchKMeans(n_clusters=min(8,len(raw)),random_state=123,batch_size=1024,n_init=3)
    # Allocate a distinct array: this L2 is exclusively for clustering geometry.
    normalized=raw.astype(np.float32,copy=True)
    normalized=normalized/np.maximum(np.linalg.norm(normalized,axis=1,keepdims=True),1e-12)
    labels=model.fit_predict(normalized).astype(np.int32)
    return labels,dict(parameters=model.get_params(),inertia=float(model.inertia_),n_iter=int(model.n_iter_),n_steps=int(model.n_steps_))


def load_cached_clusters(source,patient_id,raw,cache_dir):
    source=Path(source).resolve();cache_dir=Path(cache_dir)
    if Path(str(patient_id)).name!=str(patient_id) or str(patient_id) in ('','.','..'):raise ValueError('Invalid patient ID')
    raw=np.asarray(raw)
    if raw.ndim!=2 or not len(raw) or raw.shape[1]!=1024 or not np.isfinite(raw).all():raise ValueError('Invalid UNI features')
    expected=dict(patient_id=str(patient_id),source=str(source),source_sha256=sha256(source),n_patches=len(raw),dimension=1024,
                  n_clusters_requested=8,cluster_input_l2=True,sklearn_version=sklearn.__version__,schema=1,
                  parameters=MiniBatchKMeans(n_clusters=min(8,len(raw)),random_state=123,batch_size=1024,n_init=3).get_params())
    path=cache_dir/f'{patient_id}.npz'
    if path.exists():
        with np.load(path,allow_pickle=False) as saved:
            labels=saved['labels'];meta=json.loads(str(saved['metadata']))
        if any(meta.get(k)!=v for k,v in expected.items()):raise ValueError(f'Cluster cache source/config mismatch: {path}')
    else:
        labels,fit=cluster_labels(raw);groups,counts=np.unique(labels,return_counts=True)
        meta=dict(**expected,**{k:v for k,v in fit.items() if k!='parameters'},nonempty_clusters=len(groups),
                  cluster_ids=groups.tolist(),counts=counts.tolist(),labels_sha256=hashlib.sha256(labels.tobytes()).hexdigest())
        cache_dir.mkdir(parents=True,exist_ok=True);temporary=path.with_suffix('.tmp')
        with temporary.open('wb') as f:np.savez_compressed(f,labels=labels,metadata=json.dumps(meta,sort_keys=True))
        os.replace(temporary,path)
    if (labels.shape!=(len(raw),) or labels.dtype!=np.int32 or (labels<0).any() or (labels>=min(8,len(raw))).any()
            or hashlib.sha256(labels.tobytes()).hexdigest()!=meta['labels_sha256']):raise ValueError(f'Invalid cached labels: {path}')
    ids,counts=np.unique(labels,return_counts=True)
    if ids.tolist()!=meta['cluster_ids'] or counts.tolist()!=meta['counts']:raise ValueError(f'Invalid cached counts: {path}')
    return labels,meta
