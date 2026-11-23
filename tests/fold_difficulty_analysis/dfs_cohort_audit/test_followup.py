import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from pair_followup import classified_pairs, uno_sensitivity
from sksurv.metrics import concordance_index_ipcw
from sksurv.util import Surv


def test_ee_gap_boundaries_and_oriented_batches():
    p = classified_pairs(np.array([1.,7.,13.,25.,50.]), np.array([1,1,1,1,0]),
                         np.array([5.,4.,3.,2.,1.]), np.array(['A','A','B','B','C']))
    assert p.loc[(p.anchor==0)&(p.other==1), 'gap_bin'].item() == '0-6'
    assert p.loc[(p.anchor==0)&(p.other==2), 'gap_bin'].item() == '6-12'
    assert p.loc[(p.anchor==0)&(p.other==3), 'gap_bin'].item() == '12-24'
    assert p.loc[(p.anchor==0)&(p.other==2), 'batch_pair'].item() == 'A->B'
    assert p.loc[(p.anchor==0)&(p.other==4), 'pair_type'].item() == 'EC'
    assert p.loc[(p.anchor==0)&(p.other==4), 'gap_bin'].isna().all()


def test_uno_matches_reference_and_keeps_later_comparators():
    time=np.array([2.,4.,6.,10.,15.]);event=np.array([1,0,1,0,1],bool);risk=np.array([1.,2.,1.,0.,-1.])
    y=Surv.from_arrays(event,time)
    result=uno_sensitivity(y,time,event,risk,8.)
    assert abs(result['uno_cindex']-concordance_index_ipcw(y,y,risk,tau=8.)[0])<1e-12
    assert result['n_pairs']==6  # event2 vs4,6,10,15; event6 vs10,15
    assert abs(result['harrell_same_pairs']-4.5/6)<1e-12  # one discordance and one risk tie -> 4.5/6


def test_uno_without_censoring_equals_harrell_same_pairs():
    t=np.array([1.,2.,3.,5.]);e=np.ones(4,bool);r=np.array([4.,2.,3.,1.])
    d=uno_sensitivity(Surv.from_arrays(e,t),t,e,r,3.)
    assert abs(d['uno_cindex']-d['harrell_same_pairs'])<1e-12
