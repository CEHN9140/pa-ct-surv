import sys
import warnings
from pathlib import Path

import h5py
import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import ConvergenceWarning
from sksurv.util import Surv

sys.path.insert(0,str(Path(__file__).parent))


def example():
    rng=np.random.default_rng(123)
    x=rng.normal(size=(60,5))
    t=np.arange(1,61,dtype=float)
    e=np.arange(60)%3!=0
    return x,Surv.from_arrays(e,t)


def test_pooling_uses_every_raw_patch_without_normalization(tmp_path):
    from ridge import pool_feature
    x=np.arange(25,dtype=np.float32).reshape(5,5)
    path=tmp_path/'features.h5'
    with h5py.File(path,'w') as f: f['features']=x
    mean,n=pool_feature(path,feature_dim=5,block_size=2)
    np.testing.assert_array_equal(mean,x.astype(float).mean(axis=0))
    assert n==5
    with pytest.raises(ValueError): pool_feature(path,feature_dim=1024)


def test_ridge_scaler_is_fit_only_on_fit_patients_and_roundtrip(tmp_path):
    from ridge import fit_ridge
    x,y=example()
    model=fit_ridge(x[:40],y[:40],10.)
    np.testing.assert_allclose(model.named_steps['scale'].mean_,x[:40].mean(axis=0))
    assert model.named_steps['cox'].ties=='breslow'
    path=tmp_path/'model.joblib';joblib.dump(model,path)
    np.testing.assert_array_equal(joblib.load(path).predict(x[40:]),model.predict(x[40:]))


def test_inner_selection_is_stratified_reproducible_and_ties_favor_stronger_ridge():
    from ridge import select_alpha
    x,y=example()
    x[:]=0
    alpha,rows,roles=select_alpha(x,y,seed=123,alphas=(.1,1.,10.))
    assert alpha==10.
    assert len(rows)==9 and len(roles)==len(x)
    assert set(roles)=={0,1,2}
    assert np.all(pd.DataFrame(rows).cindex==.5)
    other,_,other_roles=select_alpha(x,y,seed=123,alphas=(.1,1.,10.))
    np.testing.assert_array_equal(roles,other_roles)


def test_solver_warning_is_failure(monkeypatch):
    from ridge import fit_ridge, CoxPHSurvivalAnalysis
    x,y=example()
    def failed(self,*args,**kwargs):
        warnings.warn('No convergence',ConvergenceWarning)
    monkeypatch.setattr(CoxPHSurvivalAnalysis,'fit',failed)
    with pytest.raises(ConvergenceWarning): fit_ridge(x,y,1.)


def test_outer_validation_labels_cannot_change_selected_model():
    from ridge import fit_outer
    x,y=example()
    a,alpha,_,_=fit_outer(x[:45],y[:45],123,alphas=(1.,10.))
    # The interface accepts training patients only: held outcomes are never passed.
    held=y[45:].copy();held['time']*=2
    b,other,_,_=fit_outer(x[:45],y[:45],123,alphas=(1.,10.))
    assert alpha==other
    np.testing.assert_array_equal(a.named_steps['cox'].coef_,b.named_steps['cox'].coef_)


def test_pair_error_overlap_distinguishes_shared_wrong_from_ties():
    from run_stage5 import compare_predictions
    patients=pd.DataFrame({'pa_id':['a','b','c','d'], 'time':[1.,2.,3.,4.], 'event':[1,1,1,0]})
    ridge=np.array([1.,2.,3.,4.])
    reference=np.array([1.,2.,3.,4.])
    row,strata,private=compare_predictions(patients,ridge,reference)
    assert row['both_wrong']==6 and row['wrong_jaccard']==1.
    assert row['delta_cindex']==0
    assert private.shared_wrong_pairs.eq(3).all()
    row,_,_=compare_predictions(patients,ridge,reference[::-1])
    assert row['both_wrong']==0 and row['wrong_jaccard']==0.


def test_shared_patient_errors_exclude_half_credit_ties():
    from run_stage5 import compare_predictions
    patients=pd.DataFrame({'pa_id':['a','b','c','d'], 'time':[1.,2.,3.,4.], 'event':[1,1,1,0]})
    row,_,errors=compare_predictions(patients,np.array([1.,1.,3.,4.]),np.array([1.,2.,3.,4.]))
    assert row['both_wrong']==5
    assert row['wrong_jaccard']==5/6
    np.testing.assert_array_equal(errors.shared_wrong_pairs,[2,2,3,3])
    assert errors.shared_wrong_pairs.sum()==2*row['both_wrong']
    np.testing.assert_array_equal(errors.ridge_tied_pairs,[1,1,0,0])


def test_high_dimensional_ridge_does_not_require_p_less_than_n():
    from ridge import fit_ridge
    rng=np.random.default_rng(123)
    x=rng.normal(size=(24,64))
    y=Surv.from_arrays(np.arange(24)%3!=0,np.arange(1,25,dtype=float))
    model=fit_ridge(x,y,.1)
    assert len(model.named_steps['cox'].coef_)==64
    assert np.isfinite(model.predict(x)).all()


def test_one_failed_inner_fit_invalidates_whole_alpha_and_preserves_all_attempts(monkeypatch):
    import ridge
    x,y=example()
    actual=ridge.fit_ridge
    calls=[]
    def numerical_failure(x,y,alpha):
        calls.append(alpha)
        if alpha==.1 and calls.count(.1)==1: raise RuntimeWarning('overflow encountered in exp')
        return actual(x,y,alpha)
    monkeypatch.setattr(ridge,'fit_ridge',numerical_failure)
    alpha,rows,roles=ridge.select_alpha(x,y,123,alphas=(.1,1.))
    table=pd.DataFrame(rows)
    assert alpha==1. and len(calls)==6
    assert table.loc[table.alpha.eq(.1),'cindex'].isna().sum()==1
    assert not table.loc[table.alpha.eq(.1),'candidate_eligible'].any()
    assert table.loc[table.alpha.eq(1.),'candidate_eligible'].all()
    assert 'RuntimeWarning' in table.loc[table.cindex.isna(),'error'].iloc[0]


def test_all_invalid_candidates_abort_and_data_errors_are_not_hidden(monkeypatch):
    import ridge
    x,y=example()
    calls=[]
    def all_failed(*args):
        calls.append(1)
        raise RuntimeWarning('overflow')
    monkeypatch.setattr(ridge,'fit_ridge',all_failed)
    with pytest.raises(RuntimeError,match='No complete'):
        ridge.select_alpha(x,y,123,alphas=(.1,1.))
    assert len(calls)==6
    def bad_data(*args): raise ValueError('broken data shape')
    monkeypatch.setattr(ridge,'fit_ridge',bad_data)
    with pytest.raises(ValueError,match='broken data'):
        ridge.select_alpha(x,y,123,alphas=(1.,))


def test_time_auc_uses_training_censor_distribution():
    from run_stage5 import evaluate_predictions
    from sksurv.metrics import cumulative_dynamic_auc
    train=pd.DataFrame({'time':[5.,20.,30.,50.,70.,100.], 'event':[1,0,1,1,0,1]})
    val=pd.DataFrame({'time':[10.,25.,40.,60.,80.], 'event':[1,0,1,1,0]})
    a=np.arange(6,dtype=float)[::-1];b=np.arange(5,dtype=float)[::-1]
    row=evaluate_predictions(train,val,a,b)
    expected=cumulative_dynamic_auc(Surv.from_arrays(train.event.astype(bool),train.time),
                                    Surv.from_arrays(val.event.astype(bool),val.time),b,[36])[0][0]
    assert row['auc_36m']==expected
