from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from ridge_overlap import transition_counts, matched_pairs
from sksurv.metrics import concordance_index_censored


def test_all_nine_transitions_keep_risk_ties_separate():
    r=np.repeat([0.,.5,1.],3);a=np.tile([0.,.5,1.],3)
    x=transition_counts(r,a)
    assert x['n_pairs']==9
    for i in ['wrong','tie','correct']:
        for j in ['wrong','tie','correct']:
            assert x[f'ridge_{i}__abmil_{j}']==1
    assert x['tie_involved']==5 and x['delta_cindex']==0


def test_ee_correction_direction_and_ec_preservation():
    p=matched_pairs(np.array([1.,2.,4.]),np.array([1,1,0],bool),np.array([1.,3.,0.]),np.array([3.,1.,0.]),1.,1.)
    ee=p[p.pair_type=='EE'].iloc[0]
    assert ee.ridge_credit==0 and ee.abmil_credit==1
    x=transition_counts(p.ridge_credit,p.abmil_credit)
    assert x['ridge_wrong__abmil_correct']==1
    assert abs(x['delta_cindex']-1/3)<1e-12


def test_net_credit_matches_reference_including_ties():
    rng=np.random.default_rng(67)
    for _ in range(20):
        t=rng.integers(1,70,30).astype(float);e=rng.random(30)<.4;e[0]=True
        r=rng.integers(-3,4,30).astype(float);a=rng.integers(-3,4,30).astype(float)
        p=matched_pairs(t,e,r,a,1.,1.)
        z=transition_counts(p.ridge_credit,p.abmil_credit)
        expected=concordance_index_censored(e,t,a)[0]-concordance_index_censored(e,t,r)[0]
        assert abs(z['delta_cindex']-expected)<1e-12


def test_margin_units_and_strong_error_flags_are_scale_invariant():
    t=np.array([1.,35.,60.]);e=np.ones(3,bool)
    r=np.array([0.,2.,4.]);a=np.array([0.,3.,6.])
    p=matched_pairs(t,e,r,a,2.,3.)
    q=matched_pairs(t,e,r*7,a*5,14.,15.)
    assert np.allclose(p.ridge_margin_z,q.ridge_margin_z)
    assert np.allclose(p.abmil_margin_z,q.abmil_margin_z)
    assert np.array_equal(p.both_strong_wrong,q.both_strong_wrong)
    assert p.both_strong_wrong.all()
