"""Matched frozen patch encoders -> all-patch mean -> nested Ridge-Cox."""
import hashlib
import json
import os
from pathlib import Path
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import h5py
import joblib
import numpy as np
import pandas as pd
from sksurv.util import Surv
from threadpoolctl import threadpool_limits

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(HERE.parent/'pathology_stage5'))
from ridge import fit_outer
from run_stage5 import evaluate_predictions

FEATURE_ROOT=Path('/home/gly001/cqj/data/lung_cancer/pathology/trident/20x_256px_0px_overlap')
CSV=ROOT/'data/trident/features_uni_v1/all_label_roi64.csv'
GEOMETRY=['patch_size','patch_size_level0','patch_level','level0_magnification','target_magnification']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path,value):
    path=Path(path);temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,ensure_ascii=False));temp.replace(path)


def signature(path):
    st=Path(path).stat()
    return dict(size=st.st_size,mtime_ns=st.st_mtime_ns)


def pool_checked(path,coords,attrs,dim):
    before=signature(path);digest=hashlib.sha256();total=np.zeros(dim,np.float64);zeros=0
    with h5py.File(path,'r') as f:
        ds=f['features'];n=len(ds)
        if ds.ndim!=2 or n==0 or ds.shape!=(len(coords),dim):raise ValueError('Invalid feature dimensions')
        if not np.array_equal(f['coords'][:],coords):raise ValueError('Patch coordinate order differs')
        for key in GEOMETRY:
            if key in attrs or key in f['coords'].attrs:
                if str(attrs.get(key))!=str(f['coords'].attrs.get(key)):raise ValueError('Patch geometry differs')
        for start in range(0,n,512):
            raw=np.asarray(ds[start:start+512])
            if not np.isfinite(raw).all():raise ValueError('Nonfinite patch features')
            zeros+=int((raw==0).all(axis=1).sum())
            digest.update(raw.tobytes());total+=raw.astype(np.float64).sum(axis=0)
        dtype=str(ds.dtype)
    if signature(path)!=before:raise RuntimeError('Source changed while pooling')
    return total/n,n,dict(**before,feature_values_sha256=digest.hexdigest(),dtype=dtype,zero_patch_vectors=zeros,
                          coords_sha256=hashlib.sha256(np.asarray(coords).tobytes()).hexdigest())


def worker(name,dim,config):
    out=HERE/'results'/name;private=out/'private';private.mkdir(parents=True,exist_ok=True)
    status=out/'status.json'
    cohort=pd.read_csv(CSV,dtype={'pa_id':str,'ct_id':str})
    assert len(cohort)==721 and cohort.event.sum()==251 and cohort.pa_id.is_unique and set(cohort.split)==set(range(5))
    cohort['time']=cohort.time.to_numpy(np.float32).astype(float)
    configuration=sha(HERE/'config.json');code=sha(__file__);cohort_hash=sha(CSV)
    solver_hash=sha(HERE.parent/'pathology_stage5/ridge.py')
    source=FEATURE_ROOT/('features_'+name)
    cache=private/'patient_features.npz';provenance_file=private/'feature_provenance.json'
    if cache.exists():
        provenance=json.loads(provenance_file.read_text())
        if provenance['cohort_sha256']!=cohort_hash or provenance['config_sha256']!=configuration or provenance['code_sha256']!=code:
            raise RuntimeError('Existing cache source/config/code differs')
        if sha(cache)!=provenance['cache_sha256']:raise RuntimeError('MeanPool cache changed')
        for case,record in provenance['files'].items():
            if signature(source/f'{case}.h5')!={k:record[k] for k in ['size','mtime_ns']}:raise RuntimeError('Cached feature source changed')
        with np.load(cache,allow_pickle=False) as z:
            assert np.array_equal(z['case_ids'],cohort.pa_id.to_numpy(dtype=str));x=z['features'];counts=z['patch_counts']
        print(f'{name}: verified MeanPool cache reused',flush=True)
    else:
        means=[];counts=[];files={}
        for i,row in enumerate(cohort.itertuples()):
            with h5py.File(row.coord_path,'r') as ref:
                coords=ref['coords'][:];attrs=dict(ref['coords'].attrs)
            mean,n,record=pool_checked(source/f'{row.pa_id}.h5',coords,attrs,dim)
            means.append(mean);counts.append(n);files[row.pa_id]=record
            if (i+1)%50==0 or i==len(cohort)-1:
                dump(status,dict(stage='pooling',encoder=name,completed_patients=i+1,total_patients=len(cohort)))
                print(f'{name}: checked/aligned/mean pooled {i+1}/721',flush=True)
        x=np.stack(means);counts=np.asarray(counts)
        np.savez_compressed(cache,features=x,case_ids=cohort.pa_id.to_numpy(dtype=str),patch_counts=counts)
        provenance=dict(files=files,cache_sha256=sha(cache),config_sha256=configuration,code_sha256=code,cohort_sha256=cohort_hash)
        dump(provenance_file,provenance)
    if name=='uni_v1':
        with np.load(HERE.parent/'pathology_stage5/results/private/patient_features.npz',allow_pickle=False) as z:
            assert np.array_equal(z['case_ids'],cohort.pa_id.to_numpy(dtype=str))
            np.testing.assert_allclose(x,z['features'],rtol=0,atol=1e-12)
            assert np.array_equal(counts,z['patch_counts'])
    rows=[];cv_rows=[]
    with threadpool_limits(limits=1):
        for fold in range(5):
            dump(status,dict(stage='fitting',encoder=name,fold=fold,completed_folds=fold))
            train=np.flatnonzero(cohort.split.ne(fold));val=np.flatnonzero(cohort.split.eq(fold))
            saved=private/f'fold_{fold}';saved.mkdir(exist_ok=True)
            print(f'{name}: Fold {fold} | 30 inner fits + outer refit',flush=True)
            y=Surv.from_arrays(cohort.iloc[train].event.astype(bool),cohort.iloc[train].time)
            model,alpha,cv,roles=fit_outer(x[train],y,seed=config['seed'],alphas=config['alphas'])
            old=pd.read_csv(HERE.parent/f'pathology_stage5/results/private/fold_{fold}/inner_roles.csv',dtype={'case_id':str}).set_index('case_id')
            assert np.array_equal(roles,old.loc[cohort.iloc[train].pa_id,'inner_fold'].to_numpy())
            pd.DataFrame(dict(case_id=cohort.iloc[train].pa_id.to_numpy(),inner_fold=roles)).to_csv(saved/'inner_roles.csv',index=False)
            joblib.dump(model,saved/'model.joblib');risk_train=model.predict(x[train]);risk_val=model.predict(x[val])
            assert np.array_equal(risk_val,joblib.load(saved/'model.joblib').predict(x[val]))
            score=evaluate_predictions(cohort.iloc[train],cohort.iloc[val],risk_train,risk_val)
            rows.append(dict(encoder=name,fold=fold,alpha=alpha,n_train=len(train),n_val=len(val),n_event_val=int(cohort.iloc[val].event.sum()),**score))
            cv_rows.extend(dict(encoder=name,fold=fold,**r) for r in cv)
            for role,ix,risk in [('train',train,risk_train),('val',val,risk_val)]:
                g=cohort.iloc[ix]
                pd.DataFrame({'case_id':g.pa_id,'dfs.month':g.time,'dfs.status':g.event,'risk_score':risk}).to_csv(saved/f'{role}_predictions.csv',index=False)
            pd.DataFrame(rows).to_csv(out/'metrics.csv',index=False)
            pd.DataFrame(cv_rows).to_csv(out/'inner_cv.csv',index=False)
            print(f'{name}: Fold {fold} DONE | alpha={alpha:g} | Train={score["train_cindex"]:.4f} Val={score["val_cindex"]:.4f}',flush=True)
    for case,record in provenance['files'].items():
        if signature(source/f'{case}.h5')!={k:record[k] for k in ['size','mtime_ns']}:raise RuntimeError('Input changed after fitting')
    if sha(CSV)!=cohort_hash or sha(__file__)!=code or sha(HERE/'config.json')!=configuration or sha(HERE.parent/'pathology_stage5/ridge.py')!=solver_hash:raise RuntimeError('Code/config/cohort changed')
    dump(out/'validation.json',dict(status='completed',patients=721,events=251,patches=int(np.sum(counts)),feature_dim=dim,
        config_sha256=configuration,cohort_sha256=cohort_hash,code_sha256=code,solver_sha256=solver_hash,mean_matrix_sha256=hashlib.sha256(x.tobytes()).hexdigest(),
        source_bundle_sha256=hashlib.sha256(json.dumps(provenance['files'],sort_keys=True).encode()).hexdigest(),
        inner_attempts=len(cv_rows),inner_failures=sum(not r['converged'] for r in cv_rows),outer_fits=5,
        inner_roles_match_stage5=True,roundtrip_exact=True,patch_geometry_and_all_values_checked=True))
    dump(status,dict(stage='completed',encoder=name,completed_folds=5))
    return name


def main():
    config=json.loads((HERE/'config.json').read_text());out=HERE/'results';out.mkdir(exist_ok=True)
    assert (config['inner_folds'],config['ties'],config['n_iter'],config['tol'])==(3,'breslow',1000,1e-9)
    dump(out/'run_config.json',config)
    print('Matched five encoders | 721 cases | no new encoder/neural training | CPU workers=5 BLAS=1',flush=True)
    try:
        with ProcessPoolExecutor(max_workers=config['workers']) as pool:
            futures=[pool.submit(worker,name,dim,config) for name,dim in config['encoders'].items()]
            for f in as_completed(futures):print('ENCODER COMPLETED:',f.result(),flush=True)
        from report import build_report
        build_report(config)
        dump(out/'status.json',dict(stage='completed',encoders=5,outer_fits=25))
    except BaseException as error:
        dump(out/'status.json',dict(stage='failed',error=f'{type(error).__name__}: {error}'))
        raise


if __name__=='__main__':main()
