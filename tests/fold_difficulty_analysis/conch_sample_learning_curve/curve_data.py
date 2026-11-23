"""Deterministic train-only nested patient subsets; no feature fitting here."""
import hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
REFERENCE=HERE.parent/'titan_aggregation_comparison'

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def dump(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(obj,indent=2));q.replace(p)

def make_jobs(c,sizes,seeds):
    jobs=[]
    for fold in range(5):
        train=np.flatnonzero(c.split.ne(fold));fraction=c.iloc[train].event.mean()
        for repeat,seed in enumerate(seeds):
            pools=[np.random.default_rng(np.random.SeedSequence([seed,fold,e])).permutation(train[c.iloc[train].event.to_numpy()==e]) for e in [0,1]]
            for n in sizes:
                ne=int(round(n*fraction));indices=sorted(np.r_[pools[0][:n-ne],pools[1][:ne]].tolist())
                if len(indices)!=n or set(indices)&set(np.flatnonzero(c.split.eq(fold))):raise ValueError('Invalid subset')
                jobs.append(dict(fold=fold,repeat=repeat,sampling_seed=seed,size_label=str(n),indices=indices))
        jobs.append(dict(fold=fold,repeat=0,sampling_seed=None,size_label='full',indices=train.tolist()))
    for j in jobs:j['key']=f"fold{j['fold']}_n{j['size_label']}_repeat{j['repeat']}"
    return jobs

def prepare():
    config=json.loads((HERE/'config.json').read_text())
    assert (config['cox_batch_size'],config['dropout'],config['lr'],config['weight_decay'])==(64,.25,1e-4,5e-4)
    assert config['training_seed']==config['inner_seed']==123
    c=pd.read_csv(ROOT/'data/trident/features_uni_v1/all_label_roi64.csv',dtype={'pa_id':str,'ct_id':str})
    assert len(c)==721 and c.event.sum()==251 and c.pa_id.is_unique and set(c.split)==set(range(5))
    c['time']=c.time.to_numpy(np.float32).astype(float)
    source=Path(json.loads((REFERENCE/'config.json').read_text())['source'])
    c['pa_path']=[str(source/'features_conch_v15'/f'{i}.h5') for i in c.pa_id]
    jobs=make_jobs(c,config['sample_sizes'],config['sampling_seeds'])
    return config,c,jobs

def fingerprint():
    paths=list(HERE.glob('*.py'))+[HERE/'config.json',ROOT/'model/abmil_cox.py',ROOT/'dataset.py',ROOT/'loss/survival_loss.py']
    for stage,file in [('pathology_stage6','nested_train.py'),('pathology_stage3','train_scale.py'),('pathology_stage3','scale_eval.py'),('pathology_stage2','diagnose.py'),('pathology_stage5','ridge.py'),('pathology_stage5','run_stage5.py')]:paths.append(HERE.parent/stage/file)
    return {str(p.relative_to(ROOT)):sha(p) for p in paths}

def subgroup(c,r):
    t=c.time.to_numpy();e=c.event.to_numpy();r=np.asarray(r)
    i,j=np.where((e[:,None]==1)&((t[:,None]<t[None,:])|((t[:,None]==t[None,:])&(e[None,:]==0))))
    margin=r[i]-r[j];credit=(margin>1e-8).astype(float)+.5*(np.abs(margin)<=1e-8)
    return {name:float(credit[mask].mean()) if mask.any() else np.nan for name,mask in [('ee',e[j]==1),('ec',e[j]==0),('ee_gap24',(e[j]==1)&((t[j]-t[i])>24))]}
