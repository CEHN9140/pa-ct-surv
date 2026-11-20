import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
sys.path.insert(0,str(Path(__file__).parent))
from diagnostics import risk_change,feature_change,evaluate_checkpoint


def test_risk_scaling_and_censor_pairs():
    risk=np.array([4.,3.,2.,1.]);time=np.arange(1,5,dtype=float);event=np.array([1,0,1,0])
    row=risk_change(risk,5*risk+7,time,event)
    assert row['spearman']==1 and row['risk_sd_ratio']==5
    assert abs(row['affine_r2']-1)<1e-12 and row['corrected_pairs']==0 and row['broken_pairs']==0
    assert row['comparable_pairs']==4
    swapped=risk.copy();swapped[0],swapped[3]=swapped[3],swapped[0]
    row=risk_change(risk,swapped,time,event)
    assert row['broken_pairs']>0 and row['delta_cindex']<0


def test_tied_event_censor_pair():
    row=risk_change(np.array([0.,0.,1.]),np.array([2.,0.,1.]),np.array([1.,1.,2.]),np.array([1,0,0]))
    from sksurv.metrics import concordance_index_censored
    old=concordance_index_censored(np.array([1,0,0],bool),np.array([1.,1.,2.]),np.array([0.,0.,1.]))[0]
    new=concordance_index_censored(np.array([1,0,0],bool),np.array([1.,1.,2.]),np.array([2.,0.,1.]))[0]
    assert abs(row['delta_cindex']-(new-old))<1e-12


def test_feature_cka_scale_invariant():
    x=np.random.default_rng(1).normal(size=(20,8))
    row=feature_change(x,3*x+2)
    assert abs(row['linear_cka']-1)<1e-12


def test_eval_does_not_change_training_rng(tmp_path):
    from model.build import Pa_Model
    torch.set_num_threads(1);torch.manual_seed(123)
    class Source:
        samples=pd.DataFrame({'pa_id':[str(i) for i in range(8)],'time':np.arange(1,9,dtype=float),'event':[1,0,1,0,1,0,1,0],'split':[0,0,0,0,1,1,1,1]})
        bags=[torch.randn(140+i,1024) for i in range(8)]
    model=Pa_Model(abmil_dropout=.25).train();state=torch.get_rng_state().clone()
    rows=evaluate_checkpoint(model,Source(),0,1,18,tmp_path,capture=True)
    assert model.training and torch.equal(state,torch.get_rng_state())
    assert len(rows)==2 and all(np.isfinite(r['cindex']) for r in rows)
    saved=np.load(tmp_path/'private/epoch_001.npz',allow_pickle=False)
    assert saved['risk'].shape==(8,) and saved['pool512'].shape==(8,512)
    assert saved['post256'].shape==(8,256) and len(saved['attention'])==sum(140+i for i in range(8))
    evaluate_checkpoint(model,Source(),0,2,36,tmp_path,capture=False)
    assert np.load(tmp_path/'private/epoch_002.npz')['risk'].shape==(8,)


def test_instrumentation_matches_original_training(tmp_path):
    from types import SimpleNamespace
    from torch.utils.data import Dataset
    import train as entry
    from dataset import build_dataloader
    from model.build import Pa_Model
    class Source(Dataset):
        samples=pd.DataFrame({'pa_id':[str(i) for i in range(8)],'time':np.arange(1,9,dtype=float),'event':[1,0,1,0,1,0,1,0],'split':[0,0,0,0,1,1,1,1]})
        bags=[torch.randn(20+i,1024) for i in range(8)]
        def __len__(self):return 8
        def __getitem__(self,i):
            row=self.samples.iloc[i]
            return self.bags[i],torch.tensor(row.event),torch.tensor(row.time,dtype=torch.float32),row.pa_id
    source=Source();states=[];rng_states=[]
    for diagnostic in [False,True]:
        torch.manual_seed(123)
        model=Pa_Model(abmil_dropout=.25)
        train=build_dataloader(source,[4,5,6,7],shuffle=True,pin_memory=False)
        val=build_dataloader(source,[0,1,2,3],pin_memory=False)
        directory=tmp_path/str(diagnostic);directory.mkdir()
        args=SimpleNamespace(survival_loss='cox',cox_batch_size=3,num_epochs=3,patience=10,rank_weight=1.,results_root=str(directory/'results'))
        function=entry.diagnostic_train if diagnostic else entry.ORIGINAL_TRAIN
        function(model,train,val,torch.optim.Adam(model.parameters(),lr=1e-4,weight_decay=5e-4),args,torch.device('cpu'),0,directory)
        states.append(torch.load(directory/'last_model.pth',weights_only=True));rng_states.append(torch.get_rng_state())
    assert all(torch.equal(states[0][k],states[1][k]) for k in states[0])
    assert torch.equal(*rng_states)


def test_complete_report_fixture(tmp_path):
    import json
    from report import summarize
    base=tmp_path/'diagnostic_fixture';base.mkdir()
    frame=pd.DataFrame({'pa_id':[str(i) for i in range(10)],'time':np.arange(1,11,dtype=float),'event':[1,0]*5,'split':np.repeat(np.arange(5),2)})
    baseline=[]
    for batch,per_epoch in [(32,18),(64,9),(128,4)]:
        condition=base/f'bs{batch}';(condition/'results').mkdir(parents=True)
        (condition/'status.json').write_text(json.dumps(dict(status='completed',completed_folds=5)))
        final=[]
        for fold in range(5):
            directory=condition/f'diagnostics/fold_{fold}';(directory/'private').mkdir(parents=True)
            frame.to_csv(directory/'private_cohort.csv',index=False)
            (directory/'audit.json').write_text(json.dumps(dict(last_epoch=11,best_epoch=1,optimizer_updates=11*per_epoch,
                train_patients=8,val_patients=2,initial_state_sha256='fixture_same_initialization')))
            rows=[]
            for epoch in range(12):
                risk=-frame.time.to_numpy(np.float32)+epoch*.01
                pool=np.repeat(risk[:,None],512,axis=1);post=pool[:,:256]
                np.savez_compressed(directory/f'private/epoch_{epoch:03d}.npz',risk=risk,pool512=pool,post256=post,
                    attention=np.ones(130,dtype=np.float32)/13,offsets=np.arange(11)*13)
                for split in ['train','val']:
                    rows.append(dict(fold=fold,epoch=epoch,optimizer_updates=epoch*per_epoch,split=split,cindex=1.,cox_loss=1.,risk_sd=1.,attention_entropy=1.,effective_patches=13.,max_attention=1/13))
            pd.DataFrame(rows).to_csv(directory/'epoch_metrics.csv',index=False)
            output=condition/f'results/fold_{fold}';output.mkdir()
            for split,mask in [('train',frame.split.ne(fold)),('val',frame.split.eq(fold))]:
                pd.DataFrame({'case_id':frame.loc[mask,'pa_id'],'risk_score':-frame.loc[mask,'time'].to_numpy(np.float32)+.01}).to_csv(output/f'{split}_predictions.csv',index=False)
            final.append(dict(fold=fold,train_cindex=1.,val_cindex=1.,gap=0.,auc_36m=.9,auc_60m=.8))
        pd.DataFrame(final).to_csv(condition/'results/metrics.csv',index=False)
        if batch==32:baseline=final
    reference=tmp_path/'full_patch_epoch_controls/abmil/results';reference.mkdir(parents=True)
    pd.DataFrame(baseline).to_csv(reference/'metrics.csv',index=False)
    summarize(base)
    validation=json.loads((base/'results/validation.json').read_text())
    assert validation['passed'] and validation['completed_folds']==15
    assert len(pd.read_csv(base/'results/batch_summary.csv'))==3
    assert (base/'results/diagnostic_curves.png').exists()


def test_publication_uses_exact_whitelist(tmp_path,monkeypatch):
    import json
    import publish as publisher
    base=tmp_path/'tests/diagnostic';base.mkdir(parents=True)
    names=['.gitignore','README.md','PLAN.md','diagnostics.py','train.py','report.py','publish.py','run_all.py','run.sh','test_diagnostics.py',
           'status.json','validation/verification.json','validation/source_sha256.json']
    names+=['results/'+n for n in ['learning_curves.csv','fold_metrics.csv','training_budgets.csv','risk_pair_changes.csv','risk_phase_changes.csv',
        'feature_attention_changes.csv','batch_summary.csv','matched_update_comparisons.csv','diagnostic_curves.png','fold_learning_curves.png','report.md','validation.json']]
    for batch in [32,64,128]:
        names += [f'bs{batch}/'+n for n in ['training.log','status.json','results/metrics.csv','results/run_config.yaml']]
        for fold in range(5):names += [f'bs{batch}/diagnostics/fold_{fold}/audit.json',f'bs{batch}/diagnostics/fold_{fold}/epoch_metrics.csv',
            f'bs{batch}/results/fold_{fold}/km_high_vs_low.png',f'bs{batch}/checkpoints/fold_{fold}/checkpoint_metadata.yaml']
    for name in names+['results/patient_risks.csv','results/cohort.csv','results/debug.json','bs32/checkpoints/secret.pth']:
        path=base/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('{}')
    (base/'results/validation.json').write_text(json.dumps({'passed':True}))
    state=dict(paths=[],committed=False)
    def git(*args):
        if args==('diff','--cached','--name-only'):return '' if not state['paths'] else '\n'.join(state['paths'])
        if args==('branch','--show-current'):return 'main'
        if args==('config','--get','remote.origin.url'):return 'https://github.com/CEHN9140/pa-ct-surv.git'
        if args==('rev-parse','HEAD'):return 'commit' if state['committed'] else 'before'
        if args in [('rev-parse','origin/main'),('rev-parse','commit^')]:return 'before'
        if args[0]=='add':state['paths']=list(args[args.index('--')+1:]);return ''
        if args[:3]==('diff','--cached','--check'):return ''
        if args[0]=='commit':
            assert '--only' in args;state['committed']=True;return ''
        if args[0]=='diff-tree':return '\n'.join(state['paths'])
        raise AssertionError(args)
    monkeypatch.setattr(publisher,'BASE',base);monkeypatch.setattr(publisher,'ROOT',tmp_path)
    monkeypatch.setattr(publisher,'git',git);monkeypatch.setattr(publisher,'network',lambda *args: '')
    assert publisher.publish()=='commit'
    assert set(state['paths'])=={str((base/name).relative_to(tmp_path)) for name in names}
