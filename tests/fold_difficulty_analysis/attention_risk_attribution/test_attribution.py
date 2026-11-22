import sys
from pathlib import Path
import numpy as np
import torch
from torch import nn
sys.path.insert(0,str(Path(__file__).parent))
from attribution import attention_path,comparable_pairs,pair_changes,technical_qc,select_controls


def test_exact_ig_nonlinear_crossings_and_completeness():
    torch.manual_seed(3)
    h=torch.randn(17,5,dtype=torch.float64);early=torch.softmax(torch.randn(17,dtype=torch.float64),0);late=early.flip(0)
    post=nn.Sequential(nn.Linear(5,7),nn.ReLU(),nn.Dropout(.25)).double().eval();head=nn.Linear(7,1).double().eval()
    result=attention_path(h,early,late,post,head)
    def risk(a):return head(post((a@h).unsqueeze(0))).squeeze()
    torch.testing.assert_close(result['contributions'].sum(),risk(late)-risk(early),atol=1e-10,rtol=1e-10)
    # Independent dense midpoint integration, including ReLU crossings.
    t=(torch.arange(20000,dtype=torch.float64)+.5)/20000
    z=((early@h)[None,:]+t[:,None]*((late-early)@h)[None,:]).requires_grad_()
    gradient=torch.autograd.grad(head(post(z)).sum(),z)[0].mean(0)
    estimate=(late-early)*(h@gradient)
    torch.testing.assert_close(result['contributions'],estimate,atol=2e-5,rtol=2e-4)
    assert result['completeness_error']<1e-10


def test_zero_displacement_and_singleton():
    post=nn.Sequential(nn.Linear(2,1),nn.ReLU(),nn.Dropout(0)).double().eval();head=nn.Linear(1,1).double().eval()
    with torch.no_grad():post[0].weight[:]=torch.tensor([[1.,0.]]);post[0].bias[:]=1;head.weight[:]=1
    h=torch.tensor([[1.,2.],[1.,-2.]],dtype=torch.float64)
    result=attention_path(h,torch.tensor([.2,.8]),torch.tensor([.8,.2]),post,head)
    assert abs(float(result['contributions'].sum()))<1e-12
    single=attention_path(h[:1],torch.ones(1),torch.ones(1),post,head)
    assert float(single['contributions'].abs().sum())==0


def test_comparable_pairs_credit_and_two_endpoint_accounting():
    from sksurv.metrics import concordance_index_censored
    t=np.array([1.,2.,2.,3.,4.]);e=np.array([1,1,0,0,1],bool)
    a,b=comparable_pairs(t,e)
    assert (1,2) in set(zip(a,b)) and (2,1) not in set(zip(a,b))
    old=np.array([1.,2.,2.,0.,-1.]);new=np.array([3.,1.,2.,0.,-1.])
    pairs,patients=pair_changes(old,new,t,e)
    expected=concordance_index_censored(e,t,new)[0]-concordance_index_censored(e,t,old)[0]
    assert abs(pairs.credit_change.mean()-expected)<1e-12
    np.testing.assert_allclose(pairs.margin_delta,pairs.event_delta-pairs.other_delta)
    assert abs(patients.pair_credit_change.sum()/2-pairs.credit_change.sum())<1e-12


def test_qc_controls_and_background():
    white=np.full((256,256,3),255,np.uint8);black=np.zeros_like(white)
    assert technical_qc(white)['white_fraction']==1
    assert technical_qc(black)['black_fraction']==1
    rng=np.random.default_rng(7);image=rng.integers(30,210,(256,256,3),dtype=np.uint8)
    q=technical_qc(image);assert np.isfinite(q['laplacian_variance']) and q['laplacian_variance']>0
    for n in [1,2,7,33]:
        high,low=select_controls(np.linspace(-1,1,n),8)
        assert len(high)==len(low)==min(8,n//2) and not set(high)&set(low)


def test_no_comparable_pairs_is_undefined_not_zero():
    pairs,patients=pair_changes([0.,1.],[1.,0.],[1.,2.],[0,0])
    assert pairs.empty and patients.mean_pair_credit_change.isna().all()
    pairs,patients=pair_changes([0.,1.],[1.,0.],[2.,2.],[1,1])
    assert pairs.empty and patients.mean_pair_credit_change.isna().all()


def test_review_uses_temporal_deterioration_and_stable_control():
    import pandas as pd
    from quality import representative_cases
    frame=pd.DataFrame(dict(pa_id=list('abcde'),fold=[0]*5,
        temporal_mean_pair_credit_change=[-.3,-.2,.1,0.,np.nan],
        temporal_changed_pair_fraction=[.5,.4,.2,0.,np.nan]))
    chosen=representative_cases(frame)
    assert chosen.pa_id.tolist()==['a','b','d']
    assert chosen.selection.iloc[-1]=='temporal_stable'
