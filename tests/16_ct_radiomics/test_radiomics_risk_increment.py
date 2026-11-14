import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parent))
from radiomics_risk_increment import checked_table, checked_roles, radiomics_risks, fit_heads


def cohort(n=90):
    return pd.DataFrame(dict(case_id=[f'p{i}|c{i}' for i in range(n)],
        time=np.arange(n,dtype=float)+1,event=np.tile([1,0],n//2),split=np.arange(n)%5))


def role_table(samples, fold=0):
    train=np.flatnonzero(samples.split.to_numpy()!=fold)
    rows=[]
    for inner in range(3):
        held=train[inner::3];remaining=np.array([i for i in train if i not in set(held)])
        stop=remaining[::7];fit=np.array([i for i in remaining if i not in set(stop)])
        for role,indices in [('fit',fit),('stop',stop),('oof',held)]:
            rows.extend(dict(inner_fold=inner,role=role,case_id=samples.iloc[i].case_id) for i in indices)
    return pd.DataFrame(rows)


def test_risk_table_aligns_identity_and_rejects_wrong_labels():
    samples=cohort();indices=np.flatnonzero(samples.split.to_numpy()!=0)
    rows=samples.iloc[indices].rename(columns={'time':'dfs.month','event':'dfs.status'})
    rows['risk_score']=np.arange(len(rows),dtype=float)
    result=checked_table(rows.iloc[::-1],samples,indices)
    assert result.index.tolist()==samples.iloc[indices].case_id.tolist()
    rows.loc[rows.index[0],'dfs.status']=1-rows.iloc[0]['dfs.status']
    with pytest.raises(ValueError,match='labels'):
        checked_table(rows,samples,indices)


def test_roles_exclude_outer_validation_and_cover_each_oof_patient_once():
    samples=cohort();roles=role_table(samples)
    partitions=checked_roles(samples,roles,0)
    assert len(partitions)==3
    assert sorted(i for _,_,_,held in partitions for i in held)==np.flatnonzero(samples.split.to_numpy()!=0).tolist()
    roles.loc[0,'case_id']=samples.loc[0,'case_id']
    with pytest.raises(ValueError,match='partition|outer|roles'):
        checked_roles(samples,roles,0)


def test_radiomics_oof_fit_excludes_held_and_outer_labels(tmp_path,monkeypatch):
    import radiomics_risk_increment as module
    samples=cohort();parts=checked_roles(samples,role_table(samples),0)
    rng=np.random.default_rng(9)
    x=pd.DataFrame(rng.normal(size=(90,4)),index=samples.case_id,
        columns=['original_shape_V','original_firstorder_Mean','original_glcm_Contrast','original_ngtdm_Strength'])
    observed=[]
    def selection(frame,time,event):
        observed.append(frame.index.tolist())
        return 1.,pd.DataFrame([dict(penalty=1.,inner_fold=0,cindex=.5)])
    monkeypatch.setattr(module,'select_penalty',selection)
    first=radiomics_risks(x,samples,0,parts,tmp_path/'first')
    train=np.flatnonzero(samples.split.to_numpy()!=0)
    assert first['oof'].index.tolist()==samples.iloc[train].case_id.tolist()
    for (_,fit,stop,held),used in zip(parts,observed):
        assert set(used)==set(samples.iloc[np.r_[fit,stop]].case_id)
        assert not set(used)&set(samples.iloc[held].case_id)
    assert set(observed[-1])==set(samples.iloc[train].case_id)
    changed=samples.copy();changed.loc[changed.split.eq(0),'event']=1-changed.loc[changed.split.eq(0),'event']
    changed.loc[changed.split.eq(0),'time']+=1000
    second=radiomics_risks(x,changed,0,parts,tmp_path/'second')
    for split in ['oof','train','val']:
        np.testing.assert_array_equal(first[split].radiomics_risk,second[split].radiomics_risk)
    import joblib
    model=joblib.load(tmp_path/'first/outer/ct_cox.joblib')
    np.testing.assert_allclose(first['val'].raw_radiomics_risk,
        model.predict(x.iloc[np.flatnonzero(samples.split.to_numpy()==0)]))


def test_fixed_fusion_head_penalizes_only_ct():
    rng=np.random.default_rng(8);n=80
    oof=pd.DataFrame(dict(pathology_risk=rng.normal(size=n),ct_A=rng.normal(size=n),
        ct_C=rng.normal(size=n),radiomics_risk=rng.normal(size=n),
        **{'dfs.month':rng.uniform(1,100,n),'dfs.status':np.tile([0,1],n//2)}))
    heads=fit_heads(oof,1.)
    assert list(heads)==['M0','M1','M2','M3']
    for name,head in heads.items():
        np.testing.assert_array_equal(head.model.alpha,[0.] if name=='M0' else [0.,1.])
        assert head.pa_scaler.n_samples_seen_==n
