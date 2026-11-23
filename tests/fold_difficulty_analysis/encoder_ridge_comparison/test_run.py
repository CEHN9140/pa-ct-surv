from pathlib import Path
import sys
import h5py
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).parent))
from run import pool_checked


def h5(path,x,coords):
    with h5py.File(path,'w') as f:
        f['features']=x;d=f.create_dataset('coords',data=coords);d.attrs['patch_size']=256


def test_pool_uses_all_raw_vectors_without_l2_and_checks_alignment(tmp_path):
    x=np.array([[1.,2.],[3.,8.],[11.,4.]],np.float32);coords=np.array([[0,0],[5,0],[10,0]])
    p=tmp_path/'x.h5';h5(p,x,coords)
    mean,n,provenance=pool_checked(p,coords,{'patch_size':256},2)
    np.testing.assert_allclose(mean,x.mean(axis=0,dtype=np.float64),atol=1e-15)
    assert n==3 and len(provenance['feature_values_sha256'])==64
    with pytest.raises(ValueError,match='coordinate'):pool_checked(p,coords[::-1],{'patch_size':256},2)
    with pytest.raises(ValueError,match='geometry'):pool_checked(p,coords,{'patch_size':512},2)


def test_nonfinite_patch_cannot_silently_enter_mean(tmp_path):
    p=tmp_path/'bad.h5';coords=np.array([[0,0],[5,0]]);h5(p,np.array([[1.,2.],[np.nan,3.]]),coords)
    with pytest.raises(ValueError,match='Nonfinite'):pool_checked(p,coords,{'patch_size':256},2)


def test_shared_ridge_solver_uses_fit_statistics_and_deterministic_inner_roles():
    from run import fit_outer
    from sklearn.model_selection import StratifiedKFold
    from sksurv.util import Surv
    rng=np.random.default_rng(123);x=rng.normal(size=(36,3));e=np.arange(36)%2==0
    y=Surv.from_arrays(e,np.arange(1,37,dtype=float))
    model,alpha,rows,roles=fit_outer(x,y,seed=123,alphas=[1000.,100000.])
    np.testing.assert_allclose(model.named_steps['scale'].mean_,x.mean(axis=0),atol=1e-15)
    expected=np.full(len(x),-1)
    for fold,(_,held) in enumerate(StratifiedKFold(3,shuffle=True,random_state=123).split(x,e)):expected[held]=fold
    np.testing.assert_array_equal(roles,expected)
    assert len(rows)==6 and alpha in [1000.,100000.] and np.isfinite(model.predict(x)).all()


def test_report_replays_predictions_and_keeps_aggregate_output(tmp_path,monkeypatch):
    import json
    import pandas as pd
    from sksurv.metrics import concordance_index_censored
    import report
    monkeypatch.setattr(report,'HERE',tmp_path)
    names=['uni_v1','keep','gpfm','phikon_v2','kaiko-vitb16']
    config={'encoders':{n:3 for n in names},'alphas':[1000.], 'bootstrap_repeats':30,'bootstrap_seed':123}
    for index,name in enumerate(names):
        out=tmp_path/'results'/name;out.mkdir(parents=True)
        rows=[]
        for fold in range(5):
            saved=out/f'private/fold_{fold}';saved.mkdir(parents=True)
            t=np.array([1.,10.,35.,60.,90.]);e=np.array([1,1,1,0,0],bool)
            risk=np.roll(np.arange(5.,0.,-1),index%2)
            score=concordance_index_censored(e,t,risk)[0]
            p=pd.DataFrame({'case_id':[f'{fold}_{i}' for i in range(5)],'dfs.month':t,'dfs.status':e.astype(int),'risk_score':risk})
            for role in ['train','val']:p.to_csv(saved/f'{role}_predictions.csv',index=False)
            rows.append(dict(encoder=name,fold=fold,alpha=1000.,train_cindex=score,val_cindex=score,gap=0.,auc_36m=.7,auc_60m=.7))
        pd.DataFrame(rows).to_csv(out/'metrics.csv',index=False)
        pd.DataFrame({'converged':[True]*150}).to_csv(out/'inner_cv.csv',index=False)
        (out/'validation.json').write_text('{}')
    report.build_report(config)
    root=tmp_path/'results'
    assert len(pd.read_csv(root/'summary.csv'))==5
    assert len(pd.read_csv(root/'paired_vs_uni.csv'))==4
    assert json.loads((root/'validation.json').read_text())['cindex_prediction_replays']==50
    for path in root.glob('*.csv'):
        assert 'case_id' not in pd.read_csv(path,nrows=0).columns
