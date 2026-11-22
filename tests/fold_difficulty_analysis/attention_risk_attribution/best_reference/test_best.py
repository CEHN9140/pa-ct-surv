from pathlib import Path
import sys
import numpy as np
import pandas as pd
sys.path[:0]=[str(Path(__file__).parent),str(Path(__file__).parents[1])]
from best_tools import contextual_counts,context_scores,matched_controls,white_fraction
from attribution import pair_changes


def test_contextual_scores_consider_both_patient_roles_and_ties():
    pairs=pd.DataFrame(dict(event_index=[0,1,0],other_index=[1,0,2],credit_change=[-1.,-.5,1.]))
    counts=contextual_counts(pairs,3)
    c=np.array([2.,-3.,0.]);scores=context_scores(c,counts.iloc[0])
    # Patient0 harmed as event with weight1 and as comparator with weight.5.
    np.testing.assert_allclose(scores['harm_support'],[1.,3.,0.])
    np.testing.assert_allclose(scores['harm_oppose'],[2.,1.5,0.])
    np.testing.assert_allclose(scores['repair_support'],[2.,0.,0.])
    np.testing.assert_allclose(scores['repair_oppose'],[0.,3.,0.])
    # c=+2 both harms a pair as comparator and repairs a different pair as event.
    assert scores['harm_support'][0]>0 and scores['repair_support'][0]>0


def test_matches_disjoint_low_score_with_coverage_caliper():
    features=np.array([[1,0],[.9,.1],[0,1],[.1,.9],[1,1],[1,-1]],float)
    score=np.array([10,8,0,0,0,0],float);blank=np.array([.1,.9,.1,.9,.5,.5])
    matches=matched_controls(score,features,blank,count=2)
    assert list(matches.high_index)==[0,1] and list(matches.low_index)==[2,3]
    assert not set(matches.high_index)&set(matches.low_index)
    assert np.all(matches.white_difference<=.1)
    assert matched_controls([1,0],features[:2],[0.,1.],count=1).empty


def test_thumbnail_white_fraction_coordinates():
    image=np.full((10,20,3),255,np.uint8);image[:,10:]=[100,20,50]
    coords=np.array([[0,0],[100,0]])
    values=white_fraction(image,coords,100,(200,100))
    np.testing.assert_allclose(values,[1.,0.])


def test_contextual_counts_no_changed_pairs():
    pairs,_=pair_changes([2.,1.],[3.,2.],[1.,2.],[1,0])
    counts=contextual_counts(pairs,2)
    assert counts.to_numpy().sum()==0


def test_review_selection_keeps_train_val_directions_distinct():
    from inspect_patches import review_cases
    frame=pd.DataFrame(dict(fold=[0]*12,split=['val']*6+['train']*6,pa_id=[str(i) for i in range(12)],
        actual_mean_pair_credit_change=[-.3,-.2,-.1,0,.1,.2]*2,
        actual_changed_pair_fraction=[.8,.7,.6,0,.5,.9]*2))
    selected=review_cases(frame)
    assert set(selected[selected.split.eq('val')].pa_id)=={'0','1','3'}
    assert set(selected[selected.split.eq('train')].pa_id)=={'10','11','9'}
    assert selected.review_case.nunique()==6
