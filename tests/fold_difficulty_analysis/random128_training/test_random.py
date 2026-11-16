import copy
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0,str(Path(__file__).parent))


@pytest.mark.parametrize('device',['cpu','cuda:0'])
def test_sampling_replay_and_joint_gradient_equal_direct_graph(device):
    if device.startswith('cuda') and not torch.cuda.is_available():pytest.skip('GPU unavailable')
    from train_random import fresh_model,core
    from loss.survival_loss import cox_loss
    model=fresh_model(123,device)
    full=core.fresh_model(1024,123,device)
    assert sum(p.numel() for p in model.parameters())==722178
    assert all(torch.equal(model.state_dict()[k],v) for k,v in full.state_dict().items())
    bags=[torch.randn(n,1024) for n in (180,70,240,160)]
    times=torch.tensor([1.,2.,3.,4.]);events=torch.tensor([1.,0.,1.,0.])
    seen=[]
    hook=model.compression[0].register_forward_pre_hook(lambda module,args:seen.append(args[0].detach().cpu().clone()))
    risks,_=core.joint_gradients(model,bags,times,events,'raw',123)
    hook.remove()
    assert len(seen)==8
    for i,n in enumerate((128,70,128,128)):
        assert seen[i].shape==(1,n,1024) and torch.equal(seen[i],seen[i+4])
        assert len(torch.unique(seen[i][0],dim=0))==n
    direct=copy.deepcopy(model);direct.zero_grad();direct.train()
    with torch.random.fork_rng(devices=[torch.device(device).index] if device.startswith('cuda') else []):
        torch.manual_seed(123)
        predicted=torch.cat([direct(x.to(device))[0] for x in bags])
        assert torch.equal(predicted.detach(),risks)
        cox_loss(predicted,times.to(device),events.to(device)).backward()
    for a,b in zip(model.parameters(),direct.parameters()):torch.testing.assert_close(a.grad,b.grad,atol=2e-6,rtol=2e-4)
    changed=[]
    hook=model.compression[0].register_forward_pre_hook(lambda module,args:changed.append(args[0].detach().cpu().clone()))
    core.joint_gradients(model,bags,times,events,'raw',124);hook.remove()
    assert not torch.equal(seen[0],changed[0])
    model.eval()
    with torch.no_grad():
        a,_,weights=model(bags[0].to(device));b=model(bags[0].to(device))[0]
    assert weights.shape[1]==180 and torch.equal(a,b)


def test_fixed_policy_evaluation_uses_eval_and_prespecified_subsets():
    from train_random import fresh_model,evaluate_policies,patch
    model=fresh_model(123,'cpu')
    bags=[torch.randn(n,1024) for n in (180,70,240,160)]
    samples=pd.DataFrame(dict(pa_id=['a','b','c','d'],time=[1.,2.,3.,4.],event=[1,0,1,0]))
    full,random=evaluate_policies(model,bags,samples,np.arange(4),123,repeats=3)
    again=evaluate_policies(model,bags,samples,np.arange(4),123,repeats=3)
    assert np.array_equal(full,again[0]) and np.array_equal(random,again[1])
    assert random.shape==(3,4) and np.all(random[:,1]==full[1])
    assert not model.training
    with torch.no_grad():
        for r in range(3):
            chosen=patch.select_indices(torch.zeros(180),'random',128,123,'a',r)
            expected=model(bags[0][chosen])[0].item()
            assert random[r,0]==pytest.approx(expected,abs=2e-6,rel=2e-6)


def test_outer_labels_do_not_change_step_or_outer_weights(tmp_path):
    from train_random import run_outer
    rng=np.random.default_rng(123)
    bags=[torch.tensor(rng.normal(size=(140+i%3,1024)),dtype=torch.float32) for i in range(18)]
    samples=pd.DataFrame(dict(pa_id=[f'p{i}' for i in range(18)],split=np.arange(18)%5,
                              time=np.arange(1,19,dtype=float),event=(np.arange(18)%3!=0).astype(int)))
    other=samples.copy();other.loc[other.split.eq(2),'time']=[40.,2.,20.,5.]
    other.loc[other.split.eq(2),'event']=[0,1,1,0]
    first=run_outer(bags,samples,2,tmp_path/'a',123,'cpu',candidates=(1,2),cox_batch_size=4,repeats=3)
    second=run_outer(bags,other,2,tmp_path/'b',123,'cpu',candidates=(1,2),cox_batch_size=4,repeats=3)
    assert first['selected_step']==second['selected_step']
    states=[torch.load(tmp_path/name/'private/outer/model.pth',weights_only=True,map_location='cpu')['model'] for name in ('a','b')]
    assert all(torch.equal(states[0][key],states[1][key]) for key in states[0])
    audit=pd.read_csv(tmp_path/'a/fit_audit.csv')
    assert len(audit)==4 and audit.initial_state_sha256.nunique()==1
    assert audit.loc[audit.role.eq('outer'),'evaluation_calls'].eq(0).all()
    assert len(pd.read_csv(tmp_path/'a/random_metrics.csv'))==3


def test_five_fold_report_pairing_and_separate_random_variability(tmp_path):
    import json
    from report_random import summarize
    from sksurv.metrics import concordance_index_censored
    root=tmp_path/'random';reference=tmp_path/'full';root.mkdir();reference.mkdir()
    (root/'run_config.json').write_text(json.dumps({'eval_random_repeats':3}))
    samples=pd.DataFrame(dict(pa_id=[f'p{i}' for i in range(30)],split=np.arange(30)%5,
                              time=np.arange(1,31,dtype=float),event=(np.arange(30)%3!=0).astype(int)))
    rng=np.random.default_rng(123);rows=[]
    for fold in range(5):
        group=samples.loc[samples.split.eq(fold)]
        risks=rng.normal(size=len(group)).astype(np.float32)
        ci=concordance_index_censored(group.event.astype(bool),group.time,risks)[0]
        row=dict(fold=fold,selected_step=2,train_cindex=.8,val_cindex=ci,gap=.8-ci,auc_36m=.75,auc_60m=.7)
        rows.append(row)
        prediction=pd.DataFrame({'case_id':group.pa_id,'dfs.month':group.time,'dfs.status':group.event,'risk_score':risks})
        audit=pd.DataFrame(dict(role=['inner']*3+['outer'],inner_fold=[0,1,2,-1],initial_state_sha256=['identical']*4,evaluation_calls=[2,2,2,0]))
        for base in (root,reference):
            folder=base/f'fold_{fold}';(folder/'private/outer').mkdir(parents=True)
            # Different row order must not change patient pairing.
            prediction.sample(frac=1,random_state=fold).to_csv(folder/'private/outer/val_predictions.csv',index=False)
            audit.to_csv(folder/'fit_audit.csv',index=False)
        pd.DataFrame([dict(**{key:value for key,value in row.items() if key not in ('val_cindex','gap')},
                           repeat=repeat,val_cindex=.6+.001*repeat,gap=.2-.001*repeat) for repeat in range(3)]).to_csv(root/f'fold_{fold}/random_metrics.csv',index=False)
        pd.DataFrame({'inner_fold':[0,1,2],'step':[2]*3,'cindex':[.6]*3}).to_csv(root/f'fold_{fold}/inner_curves.csv',index=False)
        pd.DataFrame({'step':[2],'selected':[True]}).to_csv(root/f'fold_{fold}/step_selection.csv',index=False)
    for base in (root,reference):pd.DataFrame(rows).to_csv(base/'metrics.csv',index=False)
    summary,interval=summarize(root,reference,samples,123,lambda path:path)
    assert interval['delta_cindex']==0 and interval['lower']==0 and interval['upper']==0
    assert len(summary)==4 and summary.iloc[-1].val_cindex_mean==pytest.approx(.601)
    assert summary.iloc[-1].val_cindex_sd==pytest.approx(0)
    variability=pd.read_csv(root/'random_repeat_variability.csv').set_index('metric')
    assert variability.loc['val_cindex','random_repeat_sd']==pytest.approx(.001)
    assert len(pd.read_csv(root/'fit_audit.csv'))==20
    assert (root/'report.md').is_file() and (root/'private/primary_oof_comparison.csv').is_file()
    missing=pd.read_csv(root/'fold_0/random_metrics.csv');missing.loc[0,'auc_60m']=np.nan
    missing.to_csv(root/'fold_0/random_metrics.csv',index=False)
    summary,_=summarize(root,reference,samples,123,lambda path:path)
    assert np.isnan(summary.iloc[-1].auc_60m_mean)
