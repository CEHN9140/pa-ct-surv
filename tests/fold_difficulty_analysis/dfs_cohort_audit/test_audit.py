import sys
from pathlib import Path
import numpy as np
sys.path[:0]=[str(Path(__file__).parent),str(Path(__file__).parents[1]/'attention_risk_attribution')]
from audit import pairs_for
from sksurv.metrics import concordance_index_censored


def test_pair_strata_credit_reconstruct_cindex_including_ties():
    t=np.array([1.,12.,12.,24.,37.]);e=np.array([1,1,0,0,1],bool);r=np.array([3.,2.,2.,1.,0.])
    pairs=pairs_for(t,e,r)
    assert abs(pairs.credit.mean()-concordance_index_censored(e,t,r)[0])<1e-12
    assert ((pairs.anchor==1)&(pairs.other==2)).any()
    assert not ((pairs.anchor==2)&(pairs.other==1)).any()
    summary=pairs.groupby('stratum').credit.agg(['size','mean'])
    assert abs((summary['size']*summary['mean']).sum()/summary['size'].sum()-pairs.credit.mean())<1e-12


def test_pair_credits_match_reference_with_permutations_and_risk_ties():
    rng = np.random.default_rng(45)
    for _ in range(30):
        time = rng.integers(1, 65, 25).astype(float)
        event = rng.random(25) < .4
        event[0] = True
        risk = rng.integers(-3, 4, 25).astype(float)
        assert abs(pairs_for(time, event, risk).credit.mean() - concordance_index_censored(event, time, risk)[0]) < 1e-12


def test_anchor_time_bin_boundaries_and_event_pair_type():
    p = pairs_for([12.,24.,36.,40.,80.], [1,1,1,1,0], [5.,4.,3.,2.,1.])
    assert p.groupby('anchor').time_bin.first().to_dict() == {0:'0-12',1:'12-24',2:'24-36',3:'>36'}
    assert set(p.loc[p.other == 4, 'pair_type']) == {'EC'}
    assert set(p.loc[p.other < 4, 'pair_type']) == {'EE'}
