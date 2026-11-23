"""Scratch CONCH ABMIL with unchanged Stage6 3-inner/5-outer training helpers."""
import hashlib
import json
from pathlib import Path
import sys
import h5py
import numpy as np
import pandas as pd
import torch

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(HERE.parent/'pathology_stage6'))
from nested_train import run_outer,inner_roles,fresh_model,select_step

REFERENCE=HERE.parent/'titan_aggregation_comparison'
CSV=ROOT/'data/trident/features_uni_v1/all_label_roi64.csv'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def dump(path,obj):
    p=Path(path);q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(obj,indent=2));q.replace(p)


def load_aligned(path,expected):
    st=Path(path).stat()
    with h5py.File(path,'r') as f:
        x=f['features'][:];coords=f['coords'][:];attrs=dict(f['coords'].attrs)
    if x.shape!=(expected['patches'],768) or not np.isfinite(x).all():raise ValueError('Invalid CONCH bag')
    if hashlib.sha256(x.tobytes()).hexdigest()!=expected['feature_values_sha256']:raise ValueError('CONCH values differ from TITAN reference')
    if hashlib.sha256(coords.tobytes()).hexdigest()!=expected['coords_sha256']:raise ValueError('CONCH coords differ from TITAN reference')
    if int(attrs.get('patch_size',0))!=512 or float(attrs.get('patch_size_level0',0))!=expected['patch_size_lv0']:
        raise ValueError('CONCH patch geometry differs')
    now=Path(path).stat()
    if (now.st_size,now.st_mtime_ns)!=(st.st_size,st.st_mtime_ns):raise RuntimeError('Source changed during read')
    return torch.from_numpy(x.astype(np.float32,copy=False)),dict(size=st.st_size,mtime_ns=st.st_mtime_ns)


def main():
    config=json.loads((HERE/'config.json').read_text());out=HERE/'results'
    if not torch.cuda.is_available():raise RuntimeError('CUDA required for scientific ABMIL training')
    if out.exists() and any(out.iterdir()):raise FileExistsError('Use a fresh output; refuse overwrite')
    out.mkdir(exist_ok=True);private=out/'private';private.mkdir()
    assert (config['cox_batch_size'],config['dropout'],config['lr'],config['weight_decay'])==(64,.25,1e-4,5e-4)
    c=pd.read_csv(CSV,dtype={'pa_id':str,'ct_id':str})
    assert len(c)==721 and c.event.sum()==251 and c.pa_id.is_unique and set(c.split)==set(range(5))
    c['time']=c.time.to_numpy(np.float32).astype(float)
    previous=json.loads((REFERENCE/'config.json').read_text());source=Path(previous['source'])
    records=json.loads((REFERENCE/'private/input_provenance.json').read_text())
    assert set(records)==set(c.pa_id)
    c['pa_path']=[str(source/'features_conch_v15'/f'{case}.h5') for case in c.pa_id]
    c['coord_path']=[str(source/'patches'/f'{case}_patches.h5') for case in c.pa_id]
    c.to_csv(private/'cohort.csv',index=False)
    code_paths=[Path(__file__),HERE/'compare.py',HERE/'report.py',HERE/'config.json',
                HERE.parent/'pathology_stage6/nested_train.py',HERE.parent/'pathology_stage2/diagnose.py',
                HERE.parent/'pathology_stage3/train_scale.py',HERE.parent/'pathology_stage3/scale_eval.py',
                ROOT/'model/abmil_cox.py',ROOT/'loss/survival_loss.py',ROOT/'dataset.py',ROOT/'utils.py',HERE.parent/'pathology_stage5/run_stage5.py']
    hashes={str(p.relative_to(ROOT)):sha(p) for p in code_paths};cohort_hash=sha(CSV)
    for arm in ['conch_mean','titan']:
        prior=json.loads((REFERENCE/f'results/{arm}/validation.json').read_text())
        assert prior['cohort_sha256']==cohort_hash
    bags=[];stats={};status=dict(stage='loading',completed_folds=0)
    for i,row in enumerate(c.itertuples()):
        bag,stat=load_aligned(row.pa_path,records[row.pa_id]);bags.append(bag);stats[row.pa_path]=stat
        if (i+1)%100==0 or i==720:
            dump(out/'status.json',dict(**status,loaded_patients=i+1));print(f'Exact same CONCH bags cached {i+1}/721',flush=True)
    torch.set_num_threads(4)
    rows=[]
    for fold in range(5):
        train=np.flatnonzero(c.split.ne(fold));roles=inner_roles(c,train,config['seed'])
        for arm in ['conch_mean','titan']:
            old=pd.read_csv(REFERENCE/f'results/{arm}/private/fold_{fold}/inner_roles.csv',dtype={'case_id':str}).set_index('case_id')
            assert np.array_equal(roles,old.loc[c.iloc[train].pa_id,'inner_fold'])
        status.update(stage='training',active_fold=fold,completed_folds=fold);dump(out/'status.json',status)
        row=run_outer(bags,c,fold,private/f'fold_{fold}',config['seed'],'cuda:0',config['candidate_steps'],config['cox_batch_size'])
        rows.append(row);pd.DataFrame(rows).to_csv(out/'metrics.csv',index=False)
        print('Completed outer fold',fold,flush=True)
    for path,stat in stats.items():
        now=Path(path).stat();assert (now.st_size,now.st_mtime_ns)==(stat['size'],stat['mtime_ns'])
    assert sha(CSV)==cohort_hash and all(sha(ROOT/path)==value for path,value in hashes.items())
    from report import build_report
    build_report()
    dump(out/'source_validation.json',dict(status='completed',patients=721,events=251,patches=sum(r['patches'] for r in records.values()),
         exact_values_and_coordinates_match_titan=True,cohort_sha256=cohort_hash,code_sha256=hashes,
         feature_bundle_sha256=hashlib.sha256(json.dumps(records,sort_keys=True).encode()).hexdigest()))
    dump(out/'status.json',dict(stage='completed',completed_folds=5,neural_fits=20))


if __name__=='__main__':main()
