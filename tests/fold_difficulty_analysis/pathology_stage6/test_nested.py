import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0,str(Path(__file__).parent))


def fixture():
    torch.manual_seed(123)
    bags=[torch.randn(3+i%3,5) for i in range(18)]
    samples=pd.DataFrame(dict(pa_id=[f'p{i}' for i in range(18)],
                              split=np.arange(18)%5,time=np.arange(1,19,dtype=float),
                              event=(np.arange(18)%3!=0).astype(int)))
    return bags,samples


def test_selection_compares_same_steps_and_ties_choose_earlier():
    from nested_train import select_step
    rows=[]
    for inner,scores in enumerate([[.8,.6,.4],[.5,.6,.7],[.5,.6,.7]]):
        for step,score in zip([9,18,36],scores):
            rows.append(dict(inner_fold=inner,step=step,cindex=score))
    chosen,table=select_step(pd.DataFrame(rows),(9,18,36))
    assert chosen==9 # all three shared-step means=.6, choose earliest
    assert len(table)==3
    with pytest.raises(ValueError): select_step(pd.DataFrame(rows[:-1]),(9,18,36))


def test_inner_roles_match_stage5_and_never_include_outer_val():
    from nested_train import inner_roles
    from sklearn.model_selection import StratifiedKFold
    bags,samples=fixture()
    train=np.flatnonzero(samples.split.ne(2))
    roles=inner_roles(samples,train,123)
    expected=np.full(len(train),-1)
    for inner,(_,held) in enumerate(StratifiedKFold(3,shuffle=True,random_state=123).split(np.zeros(len(train)),samples.iloc[train].event)):
        expected[held]=inner
    np.testing.assert_array_equal(roles,expected)
    for inner in range(3):
        fit=train[roles!=inner]; held=train[roles==inner]
        assert not set(fit)&set(held)
        assert not samples.iloc[np.r_[fit,held]].split.eq(2).any()


@pytest.mark.parametrize('device',['cpu','cuda:0'])
def test_fixed_training_without_held_has_exact_budget_and_no_evaluation(tmp_path,monkeypatch,device):
    if device.startswith('cuda') and not torch.cuda.is_available(): pytest.skip('GPU unavailable')
    import nested_train as n
    bags,samples=fixture()
    model=n.fresh_model(5,123,device)
    observed=[]
    def forbidden(*args,**kwargs): raise AssertionError('Outer refit evaluated before completion')
    monkeypatch.setattr(n,'evaluate',forbidden)
    meta,curves=n.train_steps(model,bags,samples,np.arange(12),None,(1,2),123,tmp_path,cox_batch_size=4)
    assert meta['optimizer_steps']==2 and meta['evaluation_calls']==0 and len(curves)==0
    assert meta['weight_decay']==5e-4 and meta['lr']==1e-4
    saved=torch.load(tmp_path/'model.pth',map_location='cpu',weights_only=True)
    assert saved['step']==2
    other=n.fresh_model(5,999,'cpu');other.load_state_dict(saved['model'],strict=True)
    for p,q in zip(model.parameters(),other.parameters()): torch.testing.assert_close(p.cpu(),q.cpu())


def test_outer_labels_do_not_change_inner_selection_or_refit(tmp_path,monkeypatch):
    import nested_train as module
    from nested_train import run_outer
    bags,samples=fixture()
    changed=samples.copy()
    changed.loc[changed.split.eq(2),'time']=[40.,2.,20.,5.]
    changed.loc[changed.split.eq(2),'event']=[0,1,1,0]
    original_evaluate=module.evaluate
    outer_calls=[]
    outer_val=set(np.flatnonzero(samples.split.eq(2)))
    def checked_evaluate(model,bags,samples,indices,condition,path):
        if outer_val&set(indices):
            assert set(indices)==outer_val
            assert (Path(path).parent/'model.pth').is_file()
            outer_calls.append(1)
        return original_evaluate(model,bags,samples,indices,condition,path)
    monkeypatch.setattr(module,'evaluate',checked_evaluate)
    a=run_outer(bags,samples,2,tmp_path/'a',123,'cpu',candidates=(1,2),cox_batch_size=4)
    b=run_outer(bags,changed,2,tmp_path/'b',123,'cpu',candidates=(1,2),cox_batch_size=4)
    assert a['selected_step']==b['selected_step']
    assert len(outer_calls)==2 # once per independent outer refit, never during inner selection
    assert np.isfinite(a['train_loss']) and np.isfinite(a['val_loss'])
    first=torch.load(tmp_path/'a/private/outer/model.pth',map_location='cpu',weights_only=True)
    second=torch.load(tmp_path/'b/private/outer/model.pth',map_location='cpu',weights_only=True)
    for key,value in first['model'].items(): assert torch.equal(value,second['model'][key])
    meta=pd.read_csv(tmp_path/'a/fit_audit.csv')
    assert meta.initial_state_sha256.nunique()==1
    assert len(meta)==4
    assert meta.loc[meta.role.eq('outer'),'evaluation_calls'].eq(0).all()
    with pytest.raises(FileExistsError): run_outer(bags,samples,2,tmp_path/'a',123,'cpu',candidates=(1,2),cox_batch_size=4)


def test_paired_summary_preserves_fold_scales_and_returns_zero_for_equal_predictions():
    from nested_report import paired_interval
    frames=[]
    for fold in range(5):
        r=np.array([4.,3.,2.,1.])*(fold+1)*100
        frames.append(pd.DataFrame({'dfs.month':[1.,2.,3.,4.],'dfs.status':[1,1,0,1],
                                     'risk_score':r,'ridge_risk':r}))
    row=paired_interval(frames,123,repeats=50)
    assert row['delta_cindex']==0 and row['lower']==0 and row['upper']==0


def test_complete_twenty_fit_comparison_report_on_small_data(tmp_path,monkeypatch):
    import nested_report
    from nested_train import run_outer
    from shutil import copyfile
    bags,samples=fixture()
    # Six validation patients per fold keep paired bootstrap well supported.
    torch.manual_seed(123)
    bags=[torch.randn(3+i%3,5) for i in range(30)]
    samples=pd.DataFrame(dict(pa_id=[f'p{i}' for i in range(30)],split=np.arange(30)%5,
                              time=np.arange(1,31,dtype=float),event=(np.arange(30)%3!=0).astype(int)))
    root=tmp_path/'nested';ridge=tmp_path/'ridge';root.mkdir();ridge.mkdir()
    rows=[]
    for fold in range(5):
        row=run_outer(bags,samples,fold,root/f'fold_{fold}',123,'cpu',candidates=(1,2),cox_batch_size=8)
        rows.append(row)
        location=ridge/f'private/fold_{fold}';location.mkdir(parents=True)
        copyfile(root/f'fold_{fold}/private/outer/val_predictions.csv',location/'val_predictions.csv')
    pd.DataFrame(rows).to_csv(root/'metrics.csv',index=False)
    pd.DataFrame(rows).assign(alpha=10000.).to_csv(ridge/'metrics.csv',index=False)
    monkeypatch.setattr(nested_report,'plot',lambda *args:None)
    nested_report.summarize(root,ridge,samples,123,lambda path:path)
    import json
    value=json.loads((root/'paired_comparison.json').read_text())
    assert value['delta_cindex']==0. and value['lower']==0. and value['upper']==0.
    assert len(pd.read_csv(root/'fit_audit.csv'))==20
    assert (root/'private/oof_comparison.csv').is_file()
    assert (root/'report.md').is_file()
