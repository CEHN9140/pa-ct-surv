import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0,str(Path(__file__).parent))


def test_requested_dimensions_and_parameter_reduction():
    from light_model import make_model,parameter_counts
    model=make_model(123,'cpu')
    assert model.compression[0].in_features==1024
    assert model.compression[0].out_features==128
    assert model.attention[0].out_features==32
    assert model.post_compression[0].in_features==128
    assert model.post_compression[0].out_features==64
    assert model.risk_head.in_features==64 and model.risk_head.out_features==1
    row=parameter_counts(model)
    assert row['full_parameters']==722178 and row['light_parameters']==143682
    assert row['reduction_fraction']==pytest.approx(1-143682/722178)


def test_optional_attention128_reports_true_reduction():
    from light_model import make_model,parameter_counts
    model=make_model(123,'cpu',attention_dim=128)
    assert model.attention[0].out_features==128
    row=parameter_counts(model)
    assert row['light_parameters']==156162
    assert row['reduction_fraction']==pytest.approx(1-156162/722178)


@pytest.mark.parametrize('device',['cpu','cuda:0'])
def test_full_patch_raw_input_dropout_eval_and_cox_gradients(tmp_path,device):
    if device.startswith('cuda') and not torch.cuda.is_available(): pytest.skip('GPU unavailable')
    from light_model import make_model
    from train_light import joint_gradients
    model=make_model(123,device)
    bags=[torch.randn(n,1024)*3 for n in (7,11,5,9)]
    original=[x.clone() for x in bags]
    inputs=[]
    hook=model.compression[0].register_forward_pre_hook(lambda module,args:inputs.append(args[0].detach().cpu().clone()))
    model.eval()
    with torch.no_grad():
        a,embedding,weights=model(bags[0].to(device))
        b,_,_=model(bags[0].to(device))
    hook.remove()
    assert torch.equal(inputs[0],bags[0].unsqueeze(0))
    assert torch.equal(a,b) and embedding.shape==(1,64) and weights.shape==(1,7,1)
    torch.testing.assert_close(weights.sum(dim=1),torch.ones((1,1),device=device))
    time=torch.tensor([1.,2.,3.,4.]);event=torch.tensor([1.,0.,1.,0.])
    joint_gradients(model,bags,time,event,'raw',123)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    for before,after in zip(original,bags): assert torch.equal(before,after)
    torch.save(model.state_dict(),tmp_path/'model.pth')
    restored=make_model(999,device);restored.load_state_dict(torch.load(tmp_path/'model.pth',weights_only=True,map_location=device),strict=True)
    model.eval();restored.eval()
    with torch.no_grad(): assert torch.equal(model(bags[0].to(device))[0],restored(bags[0].to(device))[0])


def test_nested_light_never_uses_outer_validation_for_selection(tmp_path):
    from train_light import run_outer
    rng=np.random.default_rng(123)
    bags=[torch.tensor(rng.normal(size=(3+i%3,1024)),dtype=torch.float32) for i in range(18)]
    samples=pd.DataFrame(dict(pa_id=[f'p{i}' for i in range(18)],split=np.arange(18)%5,
                              time=np.arange(1,19,dtype=float),event=(np.arange(18)%3!=0).astype(int)))
    other=samples.copy();other.loc[other.split.eq(2),'time']=[40.,2.,20.,5.]
    other.loc[other.split.eq(2),'event']=[0,1,1,0]
    a=run_outer(bags,samples,2,tmp_path/'a',123,'cpu',candidates=(1,2),cox_batch_size=4)
    b=run_outer(bags,other,2,tmp_path/'b',123,'cpu',candidates=(1,2),cox_batch_size=4)
    assert a['selected_step']==b['selected_step']
    states=[torch.load(tmp_path/name/'private/outer/model.pth',weights_only=True,map_location='cpu')['model'] for name in ('a','b')]
    for key in states[0]: assert torch.equal(states[0][key],states[1][key])
    audit=pd.read_csv(tmp_path/'a/fit_audit.csv')
    assert len(audit)==4 and audit.initial_state_sha256.nunique()==1
    assert audit.loc[audit.role.eq('outer'),'evaluation_calls'].eq(0).all()
    assert a['light_parameters']==143682


def test_complete_comparison_and_paired_signs_on_small_data(tmp_path,monkeypatch):
    import json
    from shutil import copyfile
    import compare_light
    from train_light import run_outer
    rng=np.random.default_rng(123)
    bags=[torch.tensor(rng.normal(size=(3+i%3,1024)),dtype=torch.float32) for i in range(30)]
    samples=pd.DataFrame(dict(pa_id=[f'p{i}' for i in range(30)],split=np.arange(30)%5,
                              time=np.arange(1,31,dtype=float),event=(np.arange(30)%3!=0).astype(int)))
    root=tmp_path/'light';reference=tmp_path/'reference';root.mkdir();reference.mkdir()
    rows=[]
    for fold in range(5):
        rows.append(run_outer(bags,samples,fold,root/f'fold_{fold}',123,'cpu',candidates=(1,2),cox_batch_size=8))
        destination=reference/f'fold_{fold}/private/outer';destination.mkdir(parents=True)
        copyfile(root/f'fold_{fold}/private/outer/val_predictions.csv',destination/'val_predictions.csv')
    pd.DataFrame(rows).to_csv(root/'metrics.csv',index=False)
    pd.DataFrame(rows).to_csv(reference/'metrics.csv',index=False)
    (root/'run_config.json').write_text(json.dumps(dict(attention_dim=32,full_parameters=722178,
                                                      light_parameters=143682,reduction_fraction=1-143682/722178)))
    monkeypatch.setattr(compare_light,'plot',lambda *args:None)
    compare_light.summarize(root,reference,samples,123,lambda path:path)
    value=json.loads((root/'paired_comparison.json').read_text())
    assert value['delta_cindex']==0 and value['lower']==0 and value['upper']==0
    assert len(pd.read_csv(root/'fit_audit.csv'))==20
    assert (root/'private/oof_comparison.csv').is_file()
    assert (root/'report.md').is_file()
