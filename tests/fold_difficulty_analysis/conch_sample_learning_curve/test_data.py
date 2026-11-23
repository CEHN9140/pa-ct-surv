from pathlib import Path
import sys
import numpy as np,pandas as pd
sys.path.insert(0,str(Path(__file__).parent))
from curve_data import make_jobs


def test_subsets_nested_balanced_and_never_touch_outer_val():
    c=pd.DataFrame(dict(pa_id=[str(i) for i in range(100)],split=np.arange(100)%5,event=(np.arange(100)%3==0).astype(int),time=np.arange(1,101)))
    jobs=make_jobs(c,[12,24,48],[123,7,42]);assert len(jobs)==50
    assert sum(j['size_label']=='full' for j in jobs)==5
    assert jobs==make_jobs(c,[12,24,48],[123,7,42])
    for fold in range(5):
        for repeat in range(3):
            previous=set()
            for n in [12,24,48]:
                j=next(x for x in jobs if x['fold']==fold and x['repeat']==repeat and x['size_label']==str(n))
                ix=j['indices'];assert len(ix)==n and not c.iloc[ix].split.eq(fold).any() and previous<=set(ix)
                assert abs(c.iloc[ix].event.sum()-n*c.loc[c.split!=fold,'event'].mean())<=.5+1e-12
                previous=set(ix)


def test_pair_metrics_include_same_time_censor_and_tolerate_risk_ties():
    from curve_data import subgroup
    c=pd.DataFrame(dict(time=[1,1,2],event=[1,0,1]))
    r=subgroup(c,[0,0,1e-9])
    assert r['ee']==.5 and r['ec']==.5 and np.isnan(r['ee_gap24'])
