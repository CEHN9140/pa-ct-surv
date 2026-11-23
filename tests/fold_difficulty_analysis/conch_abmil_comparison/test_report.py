from pathlib import Path
import sys,json
import numpy as np,pandas as pd
from sksurv.metrics import concordance_index_censored
sys.path.insert(0,str(Path(__file__).parent))


def test_three_model_report_audits_budgets_replays_and_subgroup_intervals(tmp_path,monkeypatch):
    import report
    from compare import paired_group_interval
    here=tmp_path/'conch';here.mkdir();monkeypatch.setattr(report,'HERE',here)
    candidates=[9,18,36,54,72,90,117,144,180]
    cfg=dict(candidate_steps=candidates,groups=['all','EE','EC','EE_gap>24'],bootstrap_repeats=30,bootstrap_seed=123)
    (here/'config.json').write_text(json.dumps(cfg))
    reference=tmp_path/'titan_aggregation_comparison/results';reference.mkdir(parents=True)
    t=np.array([1,5,10,40,50,60,70,80,90,100,110,120],float);e=np.arange(12)<8
    predictions={};tables={}
    for name in ['abmil','conch_mean','titan']:
        rows=[];predictions[name]=[]
        for fold in range(5):
            base=here/f'results/private/fold_{fold}/private/outer' if name=='abmil' else reference/name/f'private/fold_{fold}'
            base.mkdir(parents=True)
            risk=np.arange(12,0,-1,dtype=float)
            if name=='conch_mean':risk[[1,2]]=risk[[2,1]]
            if name=='abmil':risk[[2,3]]=risk[[3,2]]
            score=concordance_index_censored(e,t,risk)[0]
            p=pd.DataFrame({'case_id':[f'{fold}_{i}' for i in range(12)],'dfs.month':t,'dfs.status':e.astype(int),'risk_score':risk})
            for role in ['train','val']:p.to_csv(base/f'{role}_predictions.csv',index=False)
            predictions[name].append(p)
            rows.append(dict(fold=fold,selected_step=9,train_cindex=score,val_cindex=score,gap=0,auc_36m=.7,auc_60m=.7))
        tables[name]=pd.DataFrame(rows)
        out=here/'results' if name=='abmil' else reference/name
        tables[name].to_csv(out/'metrics.csv',index=False)
    for fold in range(5):
        d=here/f'results/private/fold_{fold}'
        (d/'completed.json').write_text(json.dumps(dict(neural_fits=4,outer_val_evaluations=1,checkpoint_reload_exact=True)))
        rows=[dict(role='inner',initial_state_sha256='same',evaluation_calls=9,optimizer_steps=180) for _ in range(3)]+[dict(role='outer',initial_state_sha256='same',evaluation_calls=0,optimizer_steps=9)]
        pd.DataFrame(rows).to_csv(d/'fit_audit.csv',index=False)
        pd.DataFrame([dict(inner_fold=i,step=s,cindex=.6) for i in range(3) for s in candidates]).to_csv(d/'inner_curves.csv',index=False)
        pd.DataFrame(dict(step=candidates,selected=[True]+[False]*8)).to_csv(d/'step_selection.csv',index=False)
    frames=[]
    for p,q in zip(predictions['titan'],predictions['conch_mean']):
        x=p.copy();x['reference_risk']=q.risk_score;frames.append(x)
    old=paired_group_interval(frames,'all',repeats=30,seed=123);(reference/'paired_comparison.json').write_text(json.dumps(old))
    report.build_report()
    intervals=pd.read_csv(here/'results/paired_intervals.csv');assert len(intervals)==12
    assert json.loads((here/'results/validation.json').read_text())['neural_fits']==20
    for f in (here/'results').glob('*.csv'):assert 'case_id' not in pd.read_csv(f,nrows=0).columns
