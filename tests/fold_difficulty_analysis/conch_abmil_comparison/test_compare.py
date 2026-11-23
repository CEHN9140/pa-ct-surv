from pathlib import Path
import sys
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).parent))
from compare import pair_credit_matrix,paired_group_interval


def frame():
    return pd.DataFrame({'dfs.month':[1.,5.,40.,50.], 'dfs.status':[1,1,1,0],
                         'risk_score':[4.,2.,3.,1.], 'reference_risk':[4.,3.,2.,1.]})


def test_subgroup_masks_exclude_censored_and_keep_large_event_gaps():
    f=frame();ee,change=pair_credit_matrix(f,'EE')
    assert ee.sum()==3 and not ee[:,3].any() and not ee[3].any()
    large,d=pair_credit_matrix(f,'EE_gap>24')
    assert large.sum()==2 and large[0,2] and large[1,2]
    assert not large[0,1]
    ec,_=pair_credit_matrix(f,'EC');assert ec.sum()==3


def test_risk_ties_keep_half_credit_in_delta():
    f=frame();f['risk_score']=[4.,2.,2.,1.]
    mask,diff=pair_credit_matrix(f,'EE')
    assert diff[1,2]==-.5 and abs(diff.sum()/mask.sum()+1/6)<1e-12


def test_paired_patient_interval_reproduces_group_delta_without_pair_resampling():
    f=frame();result=paired_group_interval([f],'EE',repeats=100,seed=123)
    assert abs(result['delta_cindex']+1/3)<1e-12
    assert result['lower']<=result['delta_cindex']<=result['upper']
    assert result['valid_repeats']>50
    f['risk_score']=f.reference_risk
    z=paired_group_interval([f]*5,'all',repeats=100,seed=123)
    assert z['delta_cindex']==z['lower']==z['upper']==0


def test_all_pair_interval_matches_previously_verified_bootstrap():
    sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'15_pathology_query_ct_tokens'))
    from risk_calibration import paired_comparison
    f=frame();old=paired_comparison([f]*5,repeats=100,seed=123,reference_columns={'ref':'reference_risk'})['fusion_minus_ref']
    new=paired_group_interval([f]*5,'all',repeats=100,seed=123)
    for key in ['delta_cindex','lower','upper','valid_repeats']:assert abs(old[key]-new[key])<1e-12
