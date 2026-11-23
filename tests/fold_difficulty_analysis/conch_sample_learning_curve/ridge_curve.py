"""Matched subset TITAN Ridge-Cox; full point reuses verified unchanged fits."""
import concurrent.futures,json,sys,time
import joblib
import numpy as np
import pandas as pd
from curve_data import HERE,ROOT,REFERENCE,prepare,dump,fingerprint,sha,subgroup
sys.path.insert(0,str(HERE.parent/'pathology_stage6'))
from nested_train import inner_roles,evaluate_predictions
from ridge import fit_outer
from sksurv.util import Surv

def fit_job(j):
    cfg,c,_=prepare();start=time.monotonic()
    matrix=REFERENCE/'private/titan.npz';z=np.load(matrix)
    assert np.array_equal(z['case_ids'].astype(str),c.pa_id.to_numpy())
    x=z['features'].astype(np.float64);train=np.array(j['indices']);val=np.flatnonzero(c.split.eq(j['fold']))
    assert not set(train)&set(val)
    out=HERE/'results/private/titan'/j['key'];out.mkdir(parents=True,exist_ok=False)
    roles=inner_roles(c,train,cfg['inner_seed'])
    if j['size_label']=='full':
        reference=REFERENCE/f"results/titan/private/fold_{j['fold']}"
        check=json.loads((REFERENCE/'results/titan/validation.json').read_text())
        assert check['cohort_sha256']==sha(ROOT/'data/trident/features_uni_v1/all_label_roi64.csv')
        prior_cfg=json.loads((REFERENCE/'config.json').read_text());assert cfg['alphas']==prior_cfg['alphas'] and cfg['inner_seed']==prior_cfg['seed']
        assert check['matrix_sha256']==sha(matrix)
        assert check['solver_sha256']==sha(HERE.parent/'pathology_stage5/ridge.py')
        model=joblib.load(reference/'model.joblib')
        old=pd.read_csv(reference/'inner_roles.csv',dtype={'case_id':str}).set_index('case_id')
        assert np.array_equal(roles,old.loc[c.iloc[train].pa_id,'inner_fold'])
        cv=pd.read_csv(REFERENCE/'results/titan/inner_cv.csv');cv=cv[cv.fold.eq(j['fold'])].drop(columns='fold').to_dict('records')
        alpha=float(model.named_steps['cox'].alpha)
        # Stored reference predictions must match both present matrix and labels.
        for name,ix in [('train',train),('val',val)]:
            old=pd.read_csv(reference/f'{name}_predictions.csv',dtype={'case_id':str},float_precision='round_trip').set_index('case_id').loc[c.iloc[ix].pa_id]
            assert np.allclose(old.risk_score,model.predict(x[ix]),rtol=1e-10,atol=1e-10)
            assert np.array_equal(old['dfs.status'],c.iloc[ix].event) and np.array_equal(old['dfs.month'],c.iloc[ix].time)
    else:
        y=Surv.from_arrays(c.iloc[train].event.astype(bool),c.iloc[train].time)
        model,alpha,cv,fitted_roles=fit_outer(x[train],y,cfg['inner_seed'],cfg['alphas'])
        assert np.array_equal(roles,fitted_roles)
    joblib.dump(model,out/'model.joblib');pd.DataFrame(cv).to_csv(out/'inner_cv.csv',index=False)
    pd.DataFrame(dict(case_id=c.iloc[train].pa_id,inner_fold=roles)).to_csv(out/'inner_roles.csv',index=False)
    risks={};diagnostics={}
    for name,ix in [('train',train),('val',val)]:
        risks[name]=model.predict(x[ix]);diagnostics.update({f'{name}_{k}':v for k,v in subgroup(c.iloc[ix],risks[name]).items()})
        pd.DataFrame(dict(case_id=c.iloc[ix].pa_id,time=c.iloc[ix].time,event=c.iloc[ix].event,risk=risks[name])).to_csv(out/f'{name}_predictions.csv',index=False)
    reloaded=joblib.load(out/'model.joblib');assert np.array_equal(reloaded.predict(x[train]),risks['train'])
    row={k:v for k,v in j.items() if k!='indices'}
    row.update(model='titan',n_train=len(train),n_events=int(c.iloc[train].event.sum()),alpha=alpha,reused_full=j['size_label']=='full',elapsed_seconds=time.monotonic()-start,**evaluate_predictions(c.iloc[train],c.iloc[val],risks['train'],risks['val']),**diagnostics)
    dump(out/'completed.json',row);return row

def main():
    cfg,c,jobs=prepare();out=HERE/'results';out.mkdir(exist_ok=True);hashes=fingerprint();matrix_hash=sha(REFERENCE/'private/titan.npz');rows=[]
    with concurrent.futures.ProcessPoolExecutor(max_workers=cfg['ridge_workers']) as pool:
        futures={pool.submit(fit_job,j):j for j in jobs}
        for future in concurrent.futures.as_completed(futures):
            rows.append(future.result());pd.DataFrame(rows).sort_values(['fold','size_label','repeat']).to_csv(out/'titan_metrics.csv',index=False)
            dump(out/'titan_status.json',dict(status='running',completed=len(rows),total=50));print('Completed TITAN',len(rows),'/50',flush=True)
    assert fingerprint()==hashes and sha(REFERENCE/'private/titan.npz')==matrix_hash
    dump(out/'titan_validation.json',dict(status='completed',jobs=50,new_outer_fits=45,new_inner_attempts=1350,reused_full_fits=5,code_sha256=hashes,matrix_sha256=matrix_hash))
    dump(out/'titan_status.json',dict(status='completed',completed=50,total=50))

if __name__=='__main__':main()
