"""Common nested Ridge protocol for same-bag CONCH MeanPool and TITAN."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import hashlib
import json
from pathlib import Path
import sys
import joblib
import numpy as np
import pandas as pd
from sksurv.util import Surv
from threadpoolctl import threadpool_limits

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
sys.path.insert(0,str(HERE.parent/'pathology_stage5'))
from ridge import fit_outer
from run_stage5 import evaluate_predictions


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path,obj):
    p=Path(path);q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(obj,indent=2));q.replace(p)


def load_matrix(path,case_ids):
    with np.load(path,allow_pickle=False) as z:
        assert np.array_equal(z['case_ids'],np.asarray(case_ids,dtype=str))
        x=np.asarray(z['features'],dtype=np.float64)
    assert x.shape==(len(case_ids),768) and np.isfinite(x).all()
    return x


def fit_arm(arm):
    config=json.loads((HERE/'config.json').read_text());root=HERE/'results'/arm;private=root/'private';private.mkdir(parents=True,exist_ok=True)
    c=pd.read_csv(ROOT/'data/trident/features_uni_v1/all_label_roi64.csv',dtype={'pa_id':str});c['time']=c.time.to_numpy(np.float32).astype(float)
    assert len(c)==721 and c.pa_id.is_unique and c.event.sum()==251 and c.event.isin([0,1]).all()
    matrix=HERE/'private'/f'{arm}.npz';matrix_hash=sha(matrix);code=sha(__file__);config_hash=sha(HERE/'config.json')
    cohort_hash=sha(ROOT/'data/trident/features_uni_v1/all_label_roi64.csv');solver_hash=sha(HERE.parent/'pathology_stage5/ridge.py')
    extraction=json.loads((HERE/'results/extraction_validation.json').read_text())
    assert matrix_hash==extraction['mean_matrix_sha256' if arm=='conch_mean' else 'titan_matrix_sha256']
    x=load_matrix(matrix,c.pa_id.to_numpy(dtype=str))
    rows=[];inner=[]
    with threadpool_limits(limits=1):
        for fold in range(5):
            dump(root/'status.json',dict(stage='fitting',arm=arm,completed_folds=fold,fold=fold))
            train=np.flatnonzero(c.split.ne(fold));val=np.flatnonzero(c.split.eq(fold));saved=private/f'fold_{fold}';saved.mkdir(exist_ok=True)
            y=Surv.from_arrays(c.iloc[train].event.astype(bool),c.iloc[train].time)
            print(f'{arm}: Fold {fold} | 30 inner attempts + outer refit',flush=True)
            model,alpha,cv,roles=fit_outer(x[train],y,seed=config['seed'],alphas=config['alphas'])
            previous=pd.read_csv(HERE.parent/f'pathology_stage5/results/private/fold_{fold}/inner_roles.csv',dtype={'case_id':str}).set_index('case_id')
            assert np.array_equal(roles,previous.loc[c.iloc[train].pa_id,'inner_fold'].to_numpy())
            pd.DataFrame(dict(case_id=c.iloc[train].pa_id.to_numpy(),inner_fold=roles)).to_csv(saved/'inner_roles.csv',index=False)
            joblib.dump(model,saved/'model.joblib');a=model.predict(x[train]);b=model.predict(x[val])
            assert np.array_equal(b,joblib.load(saved/'model.joblib').predict(x[val]))
            scores=evaluate_predictions(c.iloc[train],c.iloc[val],a,b)
            rows.append(dict(arm=arm,fold=fold,alpha=alpha,n_train=len(train),n_val=len(val),n_event_val=int(c.iloc[val].event.sum()),**scores))
            inner.extend(dict(arm=arm,fold=fold,**r) for r in cv)
            for role,ix,risk in [('train',train,a),('val',val,b)]:
                g=c.iloc[ix];pd.DataFrame({'case_id':g.pa_id,'dfs.month':g.time,'dfs.status':g.event,'risk_score':risk}).to_csv(saved/f'{role}_predictions.csv',index=False)
            pd.DataFrame(rows).to_csv(root/'metrics.csv',index=False);pd.DataFrame(inner).to_csv(root/'inner_cv.csv',index=False)
            print(f'{arm}: Fold {fold} DONE alpha={alpha:g} | Train={scores["train_cindex"]:.4f} Val={scores["val_cindex"]:.4f}',flush=True)
    assert sha(matrix)==matrix_hash and sha(__file__)==code and sha(HERE/'config.json')==config_hash
    assert sha(ROOT/'data/trident/features_uni_v1/all_label_roi64.csv')==cohort_hash and sha(HERE.parent/'pathology_stage5/ridge.py')==solver_hash
    dump(root/'validation.json',dict(status='completed',arm=arm,patients=721,events=251,inner_attempts=len(inner),
          inner_failures=sum(not r['converged'] for r in inner),outer_fits=5,matrix_sha256=matrix_hash,code_sha256=code,
          config_sha256=config_hash,cohort_sha256=cohort_hash,solver_sha256=solver_hash,inner_roles_match_stage5=True,model_roundtrip_exact=True))
    dump(root/'status.json',dict(stage='completed',arm=arm,completed_folds=5))
    return arm


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--arm',choices=['conch_mean','titan','both'],default='both');args=parser.parse_args()
    if args.arm=='both':
        with ProcessPoolExecutor(max_workers=2) as pool:
            for f in as_completed([pool.submit(fit_arm,arm) for arm in ['conch_mean','titan']]):print('COMPLETED',f.result(),flush=True)
        from report import build_report
        build_report()
    else:fit_arm(args.arm)


if __name__=='__main__':main()
