from pathlib import Path
import sys
import numpy as np
import pandas as pd
sys.path[:0]=[str(Path(__file__).parent),str(Path(__file__).parents[1]/'attention_risk_attribution')]
from cluster_tools import fit_dictionary,sample_indices,aggregate_bag,changed_margin,context_coefficients
from attribution import pair_changes


def test_dictionary_uses_train_only_and_equal_patient_weights():
    frame=pd.DataFrame(dict(pa_id=['a','b','c','d'],split=[0,1,1,2]))
    bags={'a':np.full((9,4),1000,np.float32),'b':np.arange(20,dtype=np.float32).reshape(5,4),'c':np.ones((11,4),np.float32),'d':np.full((7,4),-2,np.float32)}
    seen=[]
    def load(row):
        assert row.split!=0;seen.append(row.pa_id);return bags[row.pa_id]
    model,fit,prototypes=fit_dictionary(frame,0,load,n_clusters=2,cap=8)
    assert set(seen)=={'b','c','d'} and set(fit.pa_id)=={'b','c','d'}
    np.testing.assert_allclose(fit.total_weight,8)
    assert np.array_equal(sample_indices(100,8,'a'),sample_indices(100,8,'a'))
    bags['a'][:]=-10000
    other,_,_=fit_dictionary(frame,0,load,n_clusters=2,cap=8)
    np.testing.assert_array_equal(model.cluster_centers_,other.cluster_centers_)
    assert set(prototypes.pa_id)<=set(fit.pa_id)


def test_class_aggregation_preserves_patch_attention_and_ig():
    labels=np.array([0,1,0,2]);a=np.array([.1,.2,.3,.4]);b=a[::-1];c=np.array([1.,-2.,.5,-.3])
    row=type('Context',(),dict(harm_event=2.,harm_other=3.,repair_event=1.,repair_other=4.))()
    out=aggregate_bag(labels,a,b,c,np.array([1.,2.,3.,4.]),np.array([0.,.1,.2,.3]),row,4)
    assert out.patch_count.sum()==4 and out.loc[3,'patch_count']==0
    np.testing.assert_allclose(out.best_mass.sum(),1);np.testing.assert_allclose(out.late_mass.sum(),1)
    np.testing.assert_allclose(out.ig_sum.sum(),c.sum())
    assert np.isnan(out.loc[3,'mean_uni_norm'])


def test_net_context_margin_equals_explicit_two_endpoint_difference():
    time=np.array([1.,2.,3.,4.]);event=np.array([1,1,0,1],bool)
    before=np.array([1.,3.,2.,0.]);components=np.array([[2.,-.5],[-2.,.1],[0.,.2],[.3,-.2]])
    after=before+components.sum(1);pairs,_=pair_changes(before,after,time,event)
    coefficients=context_coefficients(before,after,time,event)
    observed=changed_margin(components,coefficients,len(pairs),2.)
    expected=((components[pairs.event_index]-components[pairs.other_index])*np.abs(pairs.credit_change.to_numpy())[:,None]).sum(0)/len(pairs)/2
    np.testing.assert_allclose(observed,expected)
    assert np.isfinite(observed).all()


def test_no_credit_change_has_zero_context_coefficients():
    result=context_coefficients(np.array([3.,2.,1.]),np.array([4.,3.,2.]),[1.,2.,3.],[1,0,1])
    np.testing.assert_array_equal(result,np.zeros(3))


def test_fit_receives_raw_values_and_patient_balanced_weights(monkeypatch):
    import cluster_tools
    original=cluster_tools.MiniBatchKMeans;captured={}
    def factory(*args,**kwargs):
        model=original(*args,**kwargs);fit=model.fit
        def record(x,sample_weight):
            captured['x']=x.copy();captured['w']=sample_weight.copy();return fit(x,sample_weight=sample_weight)
        model.fit=record;return model
    monkeypatch.setattr(cluster_tools,'MiniBatchKMeans',factory)
    frame=pd.DataFrame(dict(pa_id=['val','small','big'],split=[0,1,2]))
    bags={'small':np.full((3,4),2.,np.float32),'big':np.full((20,4),10.,np.float32)}
    cluster_tools.fit_dictionary(frame,0,lambda row:bags[row.pa_id],n_clusters=2,cap=8)
    np.testing.assert_allclose(captured['w'][:3].sum(),8)
    np.testing.assert_allclose(captured['w'][3:].sum(),8)
    assert captured['x'].max()==10 and np.linalg.norm(captured['x'][0])==4


def test_subset_patient_contexts_are_recomputed():
    from cluster_tools import patient_contexts
    data=pd.DataFrame(dict(frozen_best_attention_risk=[1.,3.,2.],frozen_late_attention_risk=[4.,2.,1.],time=[1.,2.,3.],event=[1,1,0],harm_event=[999.]*3))
    out=patient_contexts(data.iloc[:2])
    assert out.harm_event.sum()==0 and out.repair_event.sum()==1
    assert out.repair_other.sum()==1
