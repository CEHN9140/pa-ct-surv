from pathlib import Path
import sys
import torch
sys.path[:0]=[str(Path(__file__).parent),str(Path(__file__).parents[3])]
from model.build import Pa_Model
from freeze import configure,weight_hash


def test_freeze_c_keeps_gradient_path_and_train_dropout():
    torch.manual_seed(123);model=Pa_Model(abmil_dropout=.25).train();configure(model,'freeze_c');initial=weight_hash(model,'C')
    optimizer=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=1e-4,weight_decay=5e-4)
    risk=model(torch.randn(20,1024))[0];risk.square().mean().backward()
    assert model.mil.compression[0].weight.grad.abs().sum()>0
    assert model.mil.attention[0].weight.grad.abs().sum()>0
    assert all(p.grad is None for m in [model.mil.post_compression,model.mil.risk_head] for p in m.parameters())
    assert all(m.training for m in model.modules());optimizer.step();assert weight_hash(model,'C')==initial


def test_freeze_ab_updates_only_head_and_keeps_frozen_dropout():
    model=Pa_Model(abmil_dropout=.25).train();configure(model,'freeze_ab');initial=weight_hash(model,'AB')
    old=weight_hash(model,'C');optimizer=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=1e-4,weight_decay=5e-4)
    model(torch.randn(17,1024))[0].square().mean().backward();optimizer.step()
    assert weight_hash(model,'AB')==initial and weight_hash(model,'C')!=old
    assert all(m.training for m in model.modules())
    configure(model,'baseline');assert all(p.requires_grad for p in model.parameters())


def test_common_branch_rng_order_and_outer_guard(tmp_path):
    import pandas as pd
    import numpy as np
    from torch.utils.data import Dataset
    from train import common_start,branch,ARMS
    torch.set_num_threads(1)
    class Bags(list):
        def __getitem__(self,i):
            if i>=12:raise AssertionError('OuterVal accessed while fitting')
            return super().__getitem__(i)
    class Source(Dataset):
        samples=pd.DataFrame(dict(pa_id=[str(i) for i in range(16)],time=np.arange(1,17,dtype=float),event=[0,1]*8))
        bags=Bags([torch.randn(18+i,1024) for i in range(16)])
        def __len__(self):return 16
        def __getitem__(self,i):
            r=self.samples.iloc[i];return self.bags[i],torch.tensor(r.event),torch.tensor(r.time,dtype=torch.float32),r.pa_id
    source=Source();checkpoint,state=common_start(source,np.arange(8),np.arange(8,12),tmp_path/'warm',0,'cpu')
    audits=[]
    for arm in ARMS:
        _,curve,audit=branch(checkpoint,state,source,np.arange(8),np.arange(8,12),tmp_path/arm,0,arm,2,'cpu')
        assert len(curve)==4 and audit['frozen_unchanged'];audits.append(audit)
    assert len({a['initial_hash'] for a in audits})==1
    assert len({a['start_rng_hash'] for a in audits})==1
    assert len({tuple(a['order_hashes']) for a in audits})==1


def test_budget_zero_and_roundtrip_ties_choose_earlier():
    from freeze import choose_budget
    assert choose_budget([(.7,0),(.69,1)])==(.7,0)
    assert choose_budget([(.7,0),(.7+1e-16,2)])[1]==0
    assert choose_budget([(.7,0),(.8,3)])[1]==3


def test_rng_hash_covers_numpy_cursor_and_restore():
    import copy,numpy as np
    from freeze import capture_rng,rng_hash,restore_rng
    state=capture_rng();other=copy.deepcopy(state);v=list(other['numpy']);v[2]=(v[2]+1)%624;other['numpy']=tuple(v)
    assert rng_hash(state)!=rng_hash(other)
    restore_rng(state);expected=(np.random.rand(),torch.rand(1))
    restore_rng(state);actual=(np.random.rand(),torch.rand(1))
    assert actual[0]==expected[0] and torch.equal(actual[1],expected[1])


def test_report_reconstructs_fivefold_zero_budget_fixture(tmp_path,monkeypatch):
    import json,numpy as np,pandas as pd
    import report
    from train import partition,ARMS
    from freeze import weight_hash
    root=tmp_path/'root';base=root/'experiment';base.mkdir(parents=True)
    data=root/'data/trident/features_uni_v1';data.mkdir(parents=True)
    source=pd.DataFrame(dict(pa_id=[f'p{i}' for i in range(50)],time=np.arange(1,51,dtype=float),event=[1,0]*25,split=np.repeat(np.arange(5),10)))
    source.to_csv(data/'all_label_roi64.csv',index=False);model=Pa_Model(abmil_dropout=.25);state=model.state_dict();initial=weight_hash(model)
    monkeypatch.setattr(report,'ROOT',root);monkeypatch.setattr(report,'BASE',base)
    for fold in range(5):
        folder=base/f'fold_{fold}';(folder/'private').mkdir(parents=True);fit,stop,train,val=partition(source,fold)
        pd.DataFrame(dict(pa_id=source.iloc[np.r_[fit,stop]].pa_id.to_numpy(),role=['fit']*len(fit)+['stop']*len(stop))).to_csv(folder/'private/inner_roles.csv',index=False)
        (folder/'completed.json').write_text(json.dumps(dict(passed=True,outer_val_passes_per_arm=1)))
        choices=[];metrics=[]
        for scope in ['inner','outer']:
            warm=folder/f'private/{scope}/warmup';warm.mkdir(parents=True);torch.save(state,warm/'last_model.pth')
            (warm/'audit.json').write_text(json.dumps(dict(epochs=3,optimizer_reset=True)))
            for arm in ARMS:
                directory=folder/f'private/{scope}/{arm}';directory.mkdir(parents=True);torch.save(state,directory/'last_model.pth')
                audit=dict(arm=arm,epochs=20 if scope=='inner' else 0,initial_hash=initial,start_rng_hash='same',frozen_unchanged=True,optimizer_initial_state_entries=0,outer_val_used=False,
                    fit_patients=len(fit) if scope=='inner' else len(train),stop_patients=len(stop) if scope=='inner' else 0,order_hashes=['same']*20 if scope=='inner' else [])
                (directory/'audit.json').write_text(json.dumps(audit))
                if scope=='inner':pd.DataFrame([dict(epoch=i,split='stop',cindex=.8) for i in range(1,21)]).to_csv(directory/'eval_curves.csv',index=False)
                else:
                    (directory/'results').mkdir()
                    for split,indices in [('train',train),('val',val)]:
                        g=source.iloc[indices];pd.DataFrame({'case_id':g.pa_id,'dfs.month':g.time,'dfs.status':g.event,'risk_score':-g.time}).to_csv(directory/f'results/{split}_predictions.csv',index=False)
        for arm in ARMS:
            choices.append(dict(arm=arm,score_at_fork=.9,continuation_epochs=0,inner_stop_cindex=.9))
            metrics.append(dict(fold=fold,arm=arm,train_cindex=1.,val_cindex=1.,train_loss=1.,val_loss=1.,auc_36m=.9,auc_60m=.8))
        pd.DataFrame(choices).to_csv(folder/'selection.csv',index=False);pd.DataFrame(metrics).to_csv(folder/'metrics.csv',index=False)
    report.summarize();assert json.loads((base/'results/validation.json').read_text())['passed']
    assert len(pd.read_csv(base/'results/fold_metrics.csv'))==15
