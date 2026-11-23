"""Epoch-budget learning curve reusing the validated exact-Cox VJP trainer."""
import json,sys,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from curve_data import HERE,ROOT,REFERENCE,prepare,dump,fingerprint,subgroup,sha
sys.path.insert(0,str(HERE.parent/'pathology_stage6'))
from nested_train import fresh_model,inner_roles,select_step,train_steps,evaluate,evaluate_predictions
from dataset import cox_batch_sizes
import importlib.util
spec=importlib.util.spec_from_file_location('conch_loader',HERE.parent/'conch_abmil_comparison/run.py')
loader=importlib.util.module_from_spec(spec);spec.loader.exec_module(loader)

def epoch_steps(n,epochs,batch):return [e*len(cox_batch_sizes(n,batch)) for e in epochs]

def choose_epoch(rows,epochs):
    return select_step(rows[['inner_fold','epoch','cindex']].rename(columns={'epoch':'step'}),epochs)

def fit_epochs(model,bags,c,fit,held,epochs,seed,directory,batch):
    steps=epoch_steps(len(fit),epochs,batch)
    meta,rows=train_steps(model,bags,c,fit,held,steps,seed,directory,batch)
    assert meta['patient_exposures']==max(epochs)*len(fit)
    assert meta['event_exposures']==max(epochs)*int(c.iloc[fit].event.sum())
    for row in rows:row['epoch']=row['step']//len(cox_batch_sizes(len(fit),batch))
    return meta,rows

def run_job(bags,c,j,cfg,out):
    out.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    train=np.array(j['indices']);val=np.flatnonzero(c.split.eq(j['fold']))
    assert not set(train)&set(val)
    roles=inner_roles(c,train,cfg['inner_seed']);rows=[];audit=[]
    pd.DataFrame(dict(case_id=c.iloc[train].pa_id,inner_fold=roles)).to_csv(out/'inner_roles.csv',index=False)
    for inner in range(3):
        model=fresh_model(768,cfg['training_seed'],'cuda:0')
        fit,held=train[roles!=inner],train[roles==inner]
        meta,part=fit_epochs(model,bags,c,fit,held,cfg['candidate_epochs'],cfg['training_seed'],out/f'inner{inner}',cfg['cox_batch_size'])
        audit.append(dict(role=f'inner{inner}',**meta));rows.extend(dict(inner_fold=inner,**r) for r in part)
        del model
    chosen,selection=choose_epoch(pd.DataFrame(rows),cfg['candidate_epochs'])
    pd.DataFrame(rows).to_csv(out/'inner_curves.csv',index=False);selection.to_csv(out/'selection.csv',index=False)
    model=fresh_model(768,cfg['training_seed'],'cuda:0')
    meta,_=fit_epochs(model,bags,c,train,None,[chosen],cfg['training_seed'],out/'outer',cfg['cox_batch_size'])
    audit.append(dict(role='outer',**meta));assert len({r['initial_state_sha256'] for r in audit})==1
    risks={};diagnostics={}
    for name,ix in [('train',train),('val',val)]:
        metrics,values=evaluate(model,bags,c,ix,'raw',out/f'{name}_evaluation.csv')
        risks[name]=values['risk'].numpy()
        pd.DataFrame(dict(case_id=c.iloc[ix].pa_id,time=c.iloc[ix].time,event=c.iloc[ix].event,risk=risks[name])).to_csv(out/f'{name}_predictions.csv',index=False)
        diagnostics.update({f'{name}_{k}':v for k,v in subgroup(c.iloc[ix],risks[name]).items()})
        diagnostics[f'{name}_cox_loss']=metrics['cox_loss']
    restored=fresh_model(768,cfg['training_seed'],'cuda:0');restored.load_state_dict(torch.load(out/'outer/model.pth',weights_only=True)['model']);restored.eval();model.eval()
    with torch.no_grad():assert torch.equal(model(bags[train[0]].cuda())[0],restored(bags[train[0]].cuda())[0])
    pd.DataFrame(audit).to_csv(out/'fit_audit.csv',index=False)
    row={k:v for k,v in j.items() if k!='indices'}
    row.update(model='abmil',n_train=len(train),n_events=int(c.iloc[train].event.sum()),selected_epoch=chosen,optimizer_steps=meta['optimizer_steps'],patient_exposures=meta['patient_exposures'],elapsed_seconds=time.monotonic()-start,**evaluate_predictions(c.iloc[train],c.iloc[val],risks['train'],risks['val']),**diagnostics)
    dump(out/'completed.json',row);return row

def main():
    cfg,c,jobs=prepare();assert torch.cuda.is_available();torch.set_num_threads(4)
    out=HERE/'results';private=out/'private/abmil';private.mkdir(parents=True,exist_ok=True)
    hashes=fingerprint();dump(out/'private/plan.json',jobs);dump(out/'private/cohort_hash.json',dict(sha256=sha(ROOT/'data/trident/features_uni_v1/all_label_roi64.csv')))
    records=json.loads((REFERENCE/'private/input_provenance.json').read_text());bags=[];stats={};cohort_hash=sha(ROOT/'data/trident/features_uni_v1/all_label_roi64.csv')
    for i,r in enumerate(c.itertuples()):
        bag,stat=loader.load_aligned(r.pa_path,records[r.pa_id]);bags.append(bag);stats[r.pa_path]=stat
        if (i+1)%100==0:print('Cached CONCH',i+1,flush=True)
    rows=[]
    for j in jobs:
        directory=private/j['key'];print('Starting',j['key'],flush=True)
        if directory.exists():raise FileExistsError('Refusing partial overwrite: '+str(directory))
        dump(out/'abmil_status.json',dict(status='running',completed=len(rows),total=len(jobs),active=j['key']))
        rows.append(run_job(bags,c,j,cfg,directory));pd.DataFrame(rows).to_csv(out/'abmil_metrics.csv',index=False)
    assert fingerprint()==hashes and sha(ROOT/'data/trident/features_uni_v1/all_label_roi64.csv')==cohort_hash
    for path,stat in stats.items():
        now=Path(path).stat();assert (now.st_size,now.st_mtime_ns)==(stat['size'],stat['mtime_ns'])
    dump(out/'abmil_validation.json',dict(status='completed',jobs=50,neural_fits=200,code_sha256=hashes,outer_val_evaluations_per_job=1))
    dump(out/'abmil_status.json',dict(status='completed',completed=50,total=50))

if __name__=='__main__':main()
