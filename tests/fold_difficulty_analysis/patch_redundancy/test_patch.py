import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0,str(Path(__file__).parent))
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))


@pytest.mark.parametrize('device',['cpu','cuda:0'])
def test_cached_subset_matches_direct_forward_and_preserves_parameters(device):
    if device.startswith('cuda') and not torch.cuda.is_available(): pytest.skip('GPU unavailable')
    from diagnose import select_indices,subset_risk
    from model.abmil_cox import ABMIL
    torch.manual_seed(123)
    model=ABMIL(dropout=.25).to(device).eval()
    before={name:p.clone() for name,p in model.state_dict().items()}
    x=torch.randn(37,1024,device=device)*3
    with torch.inference_mode():
        h=model.compression(x.unsqueeze(0));logits=model.attention(h)
        for mode in ('top','bottom','random'):
            for k in (1,8,37,128):
                indices=select_indices(logits[0,:,0],mode,k,123,'patient-A',0)
                assert len(indices)==min(k,len(x)) and len(indices.unique())==len(indices)
                assert indices.tolist()==sorted(indices.tolist())
                expected=model(x[indices])[0].item()
                actual=subset_risk(model,h,logits,indices)
                assert actual==pytest.approx(expected,abs=2e-6,rel=2e-6)
    assert not model.training and all(p.grad is None for p in model.parameters())
    assert all(torch.equal(before[n],v) for n,v in model.state_dict().items())


def test_selection_ranks_and_sampling_are_patient_stable():
    from diagnose import select_indices
    logits=torch.arange(300,dtype=torch.float32)
    assert select_indices(logits,'top',3,123,'a',0).tolist()==[297,298,299]
    assert select_indices(logits,'bottom',3,123,'a',0).tolist()==[0,1,2]
    a=select_indices(logits,'random',128,123,'a',0)
    b=select_indices(logits,'random',256,123,'a',0)
    assert set(a.tolist())<set(b.tolist())
    assert torch.equal(a,select_indices(logits,'random',128,123,'a',0))
    assert not torch.equal(a,select_indices(logits,'random',128,123,'a',1))
    assert not torch.equal(a,select_indices(logits,'random',128,123,'b',0))


def test_attention_concentration_uniform_peaked_and_single_patch():
    from diagnose import attention_stats
    row=attention_stats(torch.zeros(10),[1,3,128])
    assert row['normalized_entropy']==pytest.approx(1)
    assert row['effective_patches']==pytest.approx(10)
    assert row['effective_ratio']==pytest.approx(1)
    assert row['top_mass_3']==pytest.approx(.3)
    assert row['top_mass_128']==pytest.approx(1)
    peaked=attention_stats(torch.tensor([1000.,-1000.,-1000.]),[1])
    assert peaked['normalized_entropy']==pytest.approx(0)
    assert peaked['effective_patches']==pytest.approx(1)
    single=attention_stats(torch.tensor([3.]),[128])
    assert single['normalized_entropy']==0 and single['effective_patches']==1


def test_refuses_stochastic_model_or_invalid_choices():
    from diagnose import subset_risk,select_indices
    from model.abmil_cox import ABMIL
    model=ABMIL(dropout=.25).train()
    with pytest.raises(ValueError,match='eval'):
        subset_risk(model,torch.ones(1,2,512),torch.ones(1,2,1),torch.tensor([0]))
    with pytest.raises(ValueError): select_indices(torch.ones(2),'wrong',1,123,'a',0)
    with pytest.raises(ValueError): select_indices(torch.ones(2),'random',0,123,'a',0)


def test_reporting_separates_fold_and_monte_carlo_variation(tmp_path):
    from report import summarize
    rows=[]
    for fold in range(5):
        for mode in ('full','top','bottom','random'):
            for repeat in (range(30) if mode=='random' else [-1]):
                rows.append(dict(fold=fold,strategy=mode,k=0 if mode=='full' else 128,repeat=repeat,
                                 cindex=.6+.01*fold+(.001*repeat if mode=='random' else 0),
                                 spearman=1.,spearman_status='ok',n_val=10,mean_kept=128,full_retained_fraction=0.))
    config={'ks':[128],'random_repeats':30,'seed':123}
    summary,contrasts=summarize(pd.DataFrame(rows),tmp_path,config)
    random=summary.loc[summary.strategy.eq('random')].iloc[0]
    assert random.cindex_mean==pytest.approx(.62+.0145)
    assert random.cindex_fold_sd==pytest.approx(np.std([.6,.61,.62,.63,.64],ddof=1))
    assert random.cindex_repeat_sd==pytest.approx(np.std(np.arange(30)*.001,ddof=1))
    assert contrasts.loc[contrasts.contrast.eq('top_minus_random'),'delta_cindex'].iloc[0]==pytest.approx(-.0145)
    assert (tmp_path/'summary.csv').is_file() and (tmp_path/'report.md').is_file()


def test_constant_risk_spearman_is_explicitly_undefined():
    from diagnose import fold_metrics
    records=[]
    for strategy in ('full','top'):
        for i in range(4):
            records.append(dict(case_id=str(i),strategy=strategy,k=0 if strategy=='full' else 2,
                                repeat=-1,time=i+1,event=1,risk=0.,n_kept=2,n_patches=4))
    result=fold_metrics(pd.DataFrame(records))
    assert result.cindex.eq(.5).all()
    assert result.spearman.isna().all() and result.spearman_status.eq('constant_risk').all()


def test_fold_end_to_end_replays_saved_predictions_and_only_reads_validation(tmp_path):
    from diagnose import run_fold,sha256
    from model.abmil_cox import ABMIL
    from sksurv.metrics import concordance_index_censored
    torch.manual_seed(123);model=ABMIL(dropout=.25).eval()
    source=tmp_path/'reference/fold_0/private/outer';source.mkdir(parents=True)
    torch.save(dict(model=model.state_dict(),step=54,seed=123),source/'model.pth')
    hashes={};rows=[];predictions=[]
    for i in range(8):
        path=tmp_path/f'{i}.pt';x=torch.randn(i*3+1,1024)
        torch.save(x,path);hashes[str(path)]=sha256(path)
        rows.append(dict(pa_id=str(i),split=0,time=float(i+1),event=int(i%2==0),pa_path=str(path)))
        with torch.inference_mode():risk=model(x)[0].item()
        predictions.append({'case_id':str(i),'dfs.month':i+1.,'dfs.status':int(i%2==0),'risk_score':risk})
    pd.DataFrame(predictions).to_csv(source/'val_predictions.csv',index=False)
    ci=concordance_index_censored(np.array([i%2==0 for i in range(8)]),np.arange(1,9),
                                  np.array([p['risk_score'] for p in predictions],dtype=np.float32))[0]
    pd.DataFrame([dict(fold=0,val_cindex=ci)]).to_csv(tmp_path/'reference/metrics.csv',index=False)
    # This other-fold case has no feature file: loading it would fail the test.
    rows.append(dict(pa_id='other',split=1,time=1.,event=1,pa_path='must_not_be_read.pt'))
    samples=pd.DataFrame(rows)
    table,attention,audit=run_fold(samples,0,tmp_path/'reference',tmp_path/'output','cpu',[2,4,128],3,123,hashes)
    assert len(table)==16 and len(attention)==8 and audit['model_frozen']
    assert audit['full_cindex']==ci and audit['checkpoint_step']==54
    assert audit['full_risk_max_abs_error']<2e-6
    saved=pd.read_csv(tmp_path/'output/private/fold_0_predictions.csv')
    assert len(saved)==8*16 and set(saved.case_id)==set(range(8))
    assert saved.loc[saved.k.eq(128),'n_kept'].equals(saved.loc[saved.k.eq(128),'n_patches'])
    broken=samples.copy();broken.loc[0,'event']=0
    with pytest.raises(ValueError,match='labels differ'):
        run_fold(broken,0,tmp_path/'reference',tmp_path/'bad','cpu',[2],1,123,hashes)
