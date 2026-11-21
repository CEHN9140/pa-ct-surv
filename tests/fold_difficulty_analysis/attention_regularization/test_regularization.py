import sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).parent))
from regularization import attention_penalty,AttentionObjective


def test_penalty_values_and_gradients():
    p=torch.tensor([.7,.2,.1],requires_grad=True)
    entropy=attention_penalty(p,'entropy')
    expected=1+(p*p.log()).sum()/np.log(3)
    torch.testing.assert_close(entropy,expected)
    q=1/(3*p.square().sum())
    torch.testing.assert_close(attention_penalty(p,'neff',.8),(.8-q).clamp_min(0).square())
    for n in [1,4,128]:
        uniform=torch.full((n,),1/n,requires_grad=True)
        for mode in ['entropy','neff']:
            loss=attention_penalty(uniform,mode,.1)
            assert abs(float(loss))<1e-6
            gradient=torch.autograd.grad(loss,uniform,retain_graph=True)[0]
            assert torch.isfinite(gradient).all()
    torch.testing.assert_close(attention_penalty(torch.tensor([1.,0.,0.]),'entropy'),torch.tensor(1.))
    logits=torch.tensor([2.,0.,-1.],requires_grad=True)
    loss=attention_penalty(logits.softmax(0),'entropy');loss.backward()
    assert logits.grad[0]>0 and torch.isfinite(logits.grad).all()


def test_cox_batch_patient_mean_and_eval_separation():
    import copy
    from model.build import Pa_Model
    from loss.survival_loss import cox_loss
    torch.set_num_threads(1);torch.manual_seed(123)
    first=Pa_Model(abmil_dropout=.25);second=copy.deepcopy(first)
    bags=[torch.randn(n,1024) for n in [17,23,1]]
    times=torch.tensor([1.,2.,3.]);events=torch.tensor([1.,0.,1.])
    objective=AttentionObjective(first,'entropy',.1)
    torch.manual_seed(321)
    risks=torch.cat([first(b)[0] for b in bags]);total=objective(risks,times,events);total.backward();objective.close()
    torch.manual_seed(321)
    outputs=[second(b) for b in bags];risk=torch.cat([v[0] for v in outputs])
    expected=cox_loss(risk,times,events)+.1*torch.stack([attention_penalty(v[2],'entropy') for v in outputs]).mean()
    expected.backward();torch.testing.assert_close(total,expected,rtol=0,atol=0)
    for a,b in zip(first.parameters(),second.parameters()):torch.testing.assert_close(a.grad,b.grad,rtol=1e-5,atol=1e-7)
    first.eval();objective=AttentionObjective(first,'entropy',.1)
    risk=torch.cat([first(b)[0] for b in bags])
    torch.testing.assert_close(objective(risk,times,events),cox_loss(risk,times,events),rtol=0,atol=0)
    assert not objective.pending;objective.close()


def test_inner_split_and_selection_ignore_outer_labels():
    import pandas as pd
    from protocol import partition,choose,CANDIDATES
    samples=pd.DataFrame({'split':np.repeat(np.arange(5),10),'event':[0,1]*25,'time':np.arange(1,51)})
    before=partition(samples,0);changed=samples.copy();changed.loc[changed.split.eq(0),'event']=1-changed.loc[changed.split.eq(0),'event']
    changed.loc[changed.split.eq(0),'time']+=10000
    after=partition(changed,0)
    for a,b in zip(before,after):np.testing.assert_array_equal(a,b)
    fit,stop,train,outer=before
    assert set(fit)|set(stop)==set(train) and not (set(train)&set(outer))
    records={c['key']:dict(selection_cindex=.7,selected_epoch=3) for c in CANDIDATES}
    assert choose(CANDIDATES,records,'entropy')[0]['mode']=='baseline'
    records['entropy_w0.1']['selection_cindex']=.75
    assert choose(CANDIDATES,records,'entropy')[0]['weight']==.1


def test_zero_weight_matches_original_and_outer_uses_fixed_last(tmp_path):
    import pandas as pd
    from torch.utils.data import Dataset
    from types import SimpleNamespace
    from dataset import build_dataloader
    from model.build import Pa_Model
    from utils import seed_everything
    import path_train
    from protocol import fit_phase
    torch.set_num_threads(1)
    class Guarded(list):
        def __getitem__(self,index):
            if index<4:raise AssertionError('Outer patient accessed during training')
            return super().__getitem__(index)
    class Source(Dataset):
        workers=0
        samples=pd.DataFrame({'pa_id':[str(i) for i in range(16)],'time':np.arange(1,17,dtype=float),'event':[1,0]*8,'split':[0]*4+[1]*12})
        bags=Guarded([torch.randn(20+i,1024) for i in range(16)])
        def __len__(self):return 16
        def __getitem__(self,index):
            row=self.samples.iloc[index]
            return self.bags[index],torch.tensor(row.event),torch.tensor(row.time,dtype=torch.float32),row.pa_id
    source=Source();fit=np.arange(4,12);stop=np.arange(12,16)
    args=SimpleNamespace(survival_loss='cox',rank_weight=1.,cox_batch_size=32,num_epochs=2,patience=10)
    reference=tmp_path/'reference';reference.mkdir()
    seed_everything(123)
    train=build_dataloader(source,fit,shuffle=True,pin_memory=True);val=build_dataloader(source,stop,pin_memory=True)
    model=Pa_Model(feature_dim=1024,abmil_dropout=.25)
    path_train.train_path(model,train,val,torch.optim.Adam(model.parameters(),lr=1e-4,weight_decay=5e-4),args,torch.device('cpu'),0,reference)
    expected=torch.load(reference/'last_model.pth',weights_only=True);expected_rng=torch.get_rng_state().clone()
    model,audit=fit_phase(source,fit,stop,dict(key='baseline',mode='baseline',weight=0.),tmp_path/'adapted',0,'cpu',epochs=2)
    actual=torch.load(tmp_path/'adapted/last_model.pth',weights_only=True)
    assert all(torch.equal(expected[k],actual[k]) for k in expected)
    assert torch.equal(expected_rng,torch.get_rng_state())
    model,audit=fit_phase(source,np.arange(4,16),None,dict(key='entropy_w0.1',mode='entropy',weight=.1),tmp_path/'outer',0,'cpu',epochs=2)
    last=torch.load(tmp_path/'outer/last_model.pth',weights_only=True);best=torch.load(tmp_path/'outer/best_model.pth',weights_only=True)
    assert audit['selected_epoch']==audit['last_epoch']==2 and audit['selection_cindex'] is None
    assert all(torch.equal(last[k],best[k]) and torch.equal(last[k],model.state_dict()[k]) for k in last)
    assert audit['regularizer_active_patient_exposures']>0


def test_complete_nested_report_fixture(tmp_path,monkeypatch):
    import json,pandas as pd,yaml
    import report as reporter
    from protocol import CANDIDATES,partition,choose
    source=pd.DataFrame({'pa_id':[str(i) for i in range(50)],'time':np.arange(1,51,dtype=float),'event':[1,0]*25,'split':np.repeat(np.arange(5),10)})
    data=tmp_path/'data/trident/features_uni_v1';data.mkdir(parents=True);source.to_csv(data/'all_label_roi64.csv',index=False)
    base=tmp_path/'experiment';base.mkdir();monkeypatch.setattr(reporter,'ROOT',tmp_path)
    for fold in range(5):
        folder=base/f'fold_{fold}';(folder/'private').mkdir(parents=True)
        (folder/'status.json').write_text(json.dumps(dict(status='completed',completed_inner=7,completed_outer=3)))
        fit,stop,train,val=partition(source,fold)
        pd.DataFrame(dict(pa_id=source.iloc[np.r_[fit,stop]].pa_id.to_numpy(),role=['fit']*len(fit)+['stop']*len(stop))).to_csv(folder/'private/inner_roles.csv',index=False)
        records={}
        for i,config in enumerate(CANDIDATES):
            directory=folder/'inner'/config['key'];directory.mkdir(parents=True)
            score=.65+i*.001
            audit=dict(config=config,phase='inner',fold=fold,fit_patients=len(fit),stop_patients=len(stop),selected_epoch=1,last_epoch=2,
                selection_cindex=score,patient_exposures=2*len(fit),regularizer_active_patient_exposures=0,initial_state_sha256='same_fixture')
            records[config['key']]=audit;(directory/'audit.json').write_text(json.dumps(audit))
            rows=[dict(epoch=e,split=split,cindex=score if e==1 else score-.01,mean_q=.5,mean_entropy=.9) for e in [1,2] for split in ['train','stop']]
            pd.DataFrame(rows).to_csv(directory/'epoch_metrics.csv',index=False)
        for family in ['baseline','entropy','neff']:
            config,inner=choose(CANDIDATES,records,family);directory=folder/'outer'/family;(directory/'results').mkdir(parents=True)
            audit=dict(config=config,phase='outer',fold=fold,selected_epoch=1,last_epoch=1,stop_patients=0,fit_patients=len(train),patient_exposures=len(train),
                regularizer_active_patient_exposures=0,initial_state_sha256='same_fixture')
            (directory/'audit.json').write_text(json.dumps(audit));(directory/'completed.json').write_text(json.dumps(dict(checkpoint_reload_exact=True,outer_val_passes=1)))
            (directory/'checkpoint_metadata.yaml').write_text(yaml.safe_dump(dict(best_epoch=1,last_epoch=1,outer_validation_used=False)))
            for name in ['best_model.pth','last_model.pth']:torch.save(dict(weight=torch.tensor([1.])),directory/name)
            for split,indices in [('train',train),('val',val)]:
                group=source.iloc[indices]
                pd.DataFrame({'case_id':group.pa_id,'dfs.month':group.time,'dfs.status':group.event,'risk_score':-group.time.astype(np.float32)}).to_csv(directory/f'results/{split}_predictions.csv',index=False)
            row=dict(fold=fold,train_cindex=1.,val_cindex=1.,auc_36m=.9,auc_60m=.8,train_loss=1.,val_loss=1.,gap=0.,
                val_mean_q=.5,val_mean_entropy=.9,val_mean_top128_mass=.2)
            pd.DataFrame([row]).to_csv(directory/'metrics.csv',index=False)
    reporter.summarize(base)
    validation=json.loads((base/'results/validation.json').read_text())
    assert validation['passed'] and validation['neural_fits']==50
    assert len(pd.read_csv(base/'results/summary.csv'))==3
    assert (base/'results/inner_stability.png').is_file()


def test_publisher_explicit_scope(tmp_path,monkeypatch):
    import json
    from types import SimpleNamespace
    import publish as publisher
    root=tmp_path/'repo';base=root/'tests/experiment';(base/'results').mkdir(parents=True)
    (base/'results/validation.json').write_text(json.dumps({'passed':True}))
    (base/'results/private_predictions.csv').write_text('sensitive')
    (base/'unrelated.py').write_text('do not publish')
    monkeypatch.setattr(publisher,'ROOT',root);monkeypatch.setattr(publisher,'BASE',base)
    monkeypatch.setattr(Path,'is_file',lambda self:True)
    monkeypatch.setattr(Path,'stat',lambda self:SimpleNamespace(st_size=100))
    staged=[];calls=[]
    def fake_git(*args):
        calls.append(args)
        if args==('diff','--cached','--name-only'):return '\n'.join(staged)
        if args==('branch','--show-current'):return 'main'
        if args==('config','--get','remote.origin.url'):return 'https://github.com/CEHN9140/pa-ct-surv.git'
        if args[0]=='rev-parse':return 'new' if args[1]=='HEAD' and any(c[0]=='commit' for c in calls) else 'old'
        if args[0]=='add':staged.extend(args[3:])
        if args[0]=='diff-tree':return '\n'.join(staged)
        return ''
    monkeypatch.setattr(publisher,'git',fake_git);monkeypatch.setattr(publisher,'network',fake_git)
    assert publisher.publish()=='new'
    assert staged and all(p.startswith('tests/experiment/') for p in staged)
    assert not any('predictions' in p or '/private/' in p or p.endswith('.pth') or p.endswith('unrelated.py') for p in staged)
    assert ('push','origin','new:refs/heads/main') in calls
