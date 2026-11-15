"""Raw UNI ABMIL: fixed candidate-step nested CV; Outer Val evaluated once."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
for stage in ('pathology_stage2','pathology_stage3','pathology_stage5'):
    sys.path.insert(0,str(Path(__file__).parent.parent/stage))

from dataset import load_path_feature
from diagnose import joint_gradients,sha256
from train_scale import batch_schedule
from scale_eval import evaluate
from run_stage5 import evaluate_predictions
from model.abmil_cox import ABMIL
from utils import seed_everything


def fresh_model(feature_dim,seed,device):
    seed_everything(seed)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    return ABMIL(in_dim=feature_dim,dropout=.25).to(device)


def inner_roles(samples,train_indices,seed):
    roles=np.full(len(train_indices),-1,dtype=int)
    splitter=StratifiedKFold(3,shuffle=True,random_state=seed)
    for inner,(_,held) in enumerate(splitter.split(np.zeros(len(train_indices)),samples.iloc[train_indices].event)):
        roles[held]=inner
    return roles


def select_step(curves,candidates):
    expected={(inner,step) for inner in range(3) for step in candidates}
    if len(curves)!=len(expected) or set(zip(curves.inner_fold,curves.step))!=expected:
        raise ValueError('Incomplete or duplicate shared-step inner validation curves')
    if not np.isfinite(curves.cindex).all(): raise ValueError('Invalid inner C-index')
    table=curves.groupby('step').cindex.agg(['mean','std']).reset_index()
    best=table['mean'].max()
    step=int(table.loc[np.isclose(table['mean'],best,atol=1e-12,rtol=0),'step'].min())
    return step,table.rename(columns={'mean':'mean_inner_cindex','std':'sd_inner_cindex'})


def train_steps(model,bags,samples,fit_indices,held_indices,candidates,seed,directory,cox_batch_size=64):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    fit_indices=np.asarray(fit_indices,dtype=int)
    if held_indices is not None and set(fit_indices)&set(held_indices): raise ValueError('FIT/HELD overlap')
    if not candidates or sorted(set(candidates))!=list(candidates) or candidates[0]<1:
        raise ValueError('Use fixed positive increasing candidate steps')
    fingerprint=hashlib.sha256(b''.join(p.detach().cpu().numpy().tobytes() for p in model.state_dict().values())).hexdigest()
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-4,weight_decay=5e-4,betas=(.9,.999),eps=1e-8,foreach=False)
    schedule=batch_schedule(fit_indices,cox_batch_size,max(candidates),seed)
    curves=[];exposures=events_seen=0
    for entry in schedule:
        batch=samples.iloc[entry['indices']]
        t=torch.tensor(batch.time.to_numpy(),dtype=torch.float32)
        e=torch.tensor(batch.event.to_numpy(),dtype=torch.float32)
        _,loss=joint_gradients(model,[bags[i] for i in entry['indices']],t,e,'raw',entry['dropout_seed'])
        optimizer.step()
        if not all(torch.isfinite(p).all() for p in model.parameters()): raise RuntimeError('Nonfinite parameters')
        exposures+=len(batch);events_seen+=int(e.sum())
        if held_indices is not None and entry['step'] in candidates:
            metrics,_=evaluate(model,bags,samples,held_indices,'raw',directory/f'step_{entry["step"]:03d}_held_predictions.csv')
            curves.append(dict(step=entry['step'],batch_loss=float(loss),**metrics))
            print(f"  Step {entry['step']}/{max(candidates)} | Inner HELD Loss: {metrics['cox_loss']:.6f} | "
                  f"Inner HELD C-index: {metrics['cindex']:.6f}",flush=True)
        elif held_indices is None and (entry['step']%18==0 or entry['step']==max(candidates)):
            print(f"  Outer refit step {entry['step']}/{max(candidates)} | Cox batch loss: {float(loss):.6f}",flush=True)
    torch.save(dict(model=model.state_dict(),step=max(candidates),seed=seed),directory/'model.pth')
    metadata=dict(optimizer_steps=max(candidates),evaluation_calls=len(curves),initial_state_sha256=fingerprint,
                  patient_exposures=exposures,event_exposures=events_seen,n_fit=len(fit_indices),
                  n_held=len(held_indices) if held_indices is not None else 0,lr=1e-4,weight_decay=5e-4)
    (directory/'training_metadata.json').write_text(json.dumps(metadata,indent=2))
    return metadata,curves


def run_outer(bags,samples,fold,directory,seed,device,
              candidates=(9,18,36,54,72,90,117,144,180),cox_batch_size=64):
    directory=Path(directory)
    if directory.exists() and any(directory.iterdir()): raise FileExistsError('Refusing to overwrite fold results')
    directory.mkdir(parents=True,exist_ok=True)
    (directory/'private').mkdir()
    train=np.flatnonzero(samples.split.ne(fold));val=np.flatnonzero(samples.split.eq(fold))
    roles=inner_roles(samples,train,seed)
    pd.DataFrame(dict(case_id=samples.iloc[train].pa_id.to_numpy(),inner_fold=roles)).to_csv(
        directory/'private/inner_roles.csv',index=False)
    curves,audit=[],[]
    start=time.monotonic()
    for inner in range(3):
        fit,held=train[roles!=inner],train[roles==inner]
        if set(held)&set(val) or set(fit)&set(val): raise RuntimeError('Outer Val entered inner fit/selection')
        model=fresh_model(bags[0].shape[1],seed,device)
        print(f'Fold {fold} Inner {inner}: FIT={len(fit)}, HELD={len(held)}, budget={max(candidates)} steps',flush=True)
        metadata,part=train_steps(model,bags,samples,fit,held,candidates,seed,
                                  directory/'private'/f'inner_{inner}',cox_batch_size)
        audit.append(dict(role='inner',inner_fold=inner,**metadata))
        curves.extend(dict(inner_fold=inner,**row) for row in part)
        pd.DataFrame(curves).to_csv(directory/'inner_curves.csv',index=False)
        del model
    chosen,selection=select_step(pd.DataFrame(curves),candidates)
    selection['selected']=selection.step.eq(chosen)
    selection.to_csv(directory/'step_selection.csv',index=False)
    print(f'Fold {fold}: selected Step={chosen} by three-fold mean; fresh Outer Train refit; Outer Val unused',flush=True)
    model=fresh_model(bags[0].shape[1],seed,device)
    metadata,_=train_steps(model,bags,samples,train,None,(chosen,),seed,directory/'private/outer',cox_batch_size)
    audit.append(dict(role='outer',inner_fold=-1,**metadata))
    # The first and only evaluation of Outer Val occurs after the selected fixed budget.
    predictions={};final_diagnostics={}
    for split,indices in [('train',train),('val',val)]:
        final,values=evaluate(model,bags,samples,indices,'raw',directory/'private/outer'/f'{split}_predictions.csv')
        final_diagnostics[split+'_loss']=final['cox_loss']
        final_diagnostics[split+'_risk_sd']=final['risk_sd']
        part=samples.iloc[indices]
        pred=pd.DataFrame({'case_id':part.pa_id.to_numpy(),'dfs.month':part.time.to_numpy(),
                           'dfs.status':part.event.to_numpy(),'risk_score':values['risk'].numpy()})
        pred.to_csv(directory/'private/outer'/f'{split}_predictions.csv',index=False)
        predictions[split]=pred
    metrics=evaluate_predictions(samples.iloc[train],samples.iloc[val],
                                  predictions['train'].risk_score.to_numpy(),predictions['val'].risk_score.to_numpy())
    # Final checkpoint reconstructs the sole outer predictions exactly on the same device.
    state=torch.load(directory/'private/outer/model.pth',map_location=device,weights_only=True)
    restored=fresh_model(bags[0].shape[1],seed,device);restored.load_state_dict(state['model'],strict=True)
    restored.eval()
    with torch.no_grad():
        # Check a training patient; do not run a second Outer Val evaluation.
        original=model(bags[train[0]].to(device))[0]
        reloaded=restored(bags[train[0]].to(device))[0]
        if not torch.equal(original,reloaded): raise RuntimeError('Outer checkpoint reload changed risk')
    pd.DataFrame(audit).to_csv(directory/'fit_audit.csv',index=False)
    result=dict(fold=fold,selected_step=chosen,n_train=len(train),n_val=len(val),
                n_event_train=int(samples.iloc[train].event.sum()),**metrics,**final_diagnostics)
    pd.DataFrame([result]).to_csv(directory/'metrics.csv',index=False)
    completed=dict(**result,neural_fits=4,outer_val_evaluations=1,
                    checkpoint_reload_exact=True,elapsed_seconds=time.monotonic()-start)
    (directory/'completed.json').write_text(json.dumps(completed,indent=2))
    print(f"Fold {fold} completed | Step={chosen} | Train C-index={metrics['train_cindex']:.6f} | "
          f"Val C-index={metrics['val_cindex']:.6f} | Gap={metrics['gap']:.6f}",flush=True)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv',type=Path,default=ROOT/'data/trident/features_uni_v1/all_label_roi64.csv')
    parser.add_argument('--ridge_root',type=Path,default=Path(__file__).parent.parent/'pathology_stage5/results')
    parser.add_argument('--output',type=Path,default=Path(__file__).parent/'results')
    parser.add_argument('--seed',type=int,default=123)
    parser.add_argument('--device',default='cuda:0')
    args=parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()): raise FileExistsError('Use a new --output; do not overwrite')
    args.output.mkdir(parents=True,exist_ok=True)
    status=dict(status='running',pid=os.getpid(),completed_outer_folds=0,seed=args.seed)
    (args.output/'status.json').write_text(json.dumps(status,indent=2))
    sources={}
    def track(path):
        path=Path(path).resolve();value=sha256(path)
        if str(path) in sources and sources[str(path)]!=value: raise RuntimeError('Source changed')
        sources[str(path)]=value
        return path
    code_paths=[Path(__file__),Path(__file__).with_name('nested_report.py'),
                Path(__file__).parent.parent/'pathology_stage2/diagnose.py',
                Path(__file__).parent.parent/'pathology_stage3/train_scale.py',
                Path(__file__).parent.parent/'pathology_stage3/scale_eval.py',
                Path(__file__).parent.parent/'pathology_stage5/run_stage5.py',
                ROOT/'model/abmil_cox.py',ROOT/'loss/survival_loss.py',ROOT/'dataset.py',ROOT/'utils.py',
                ROOT/'tests/15_pathology_query_ct_tokens/risk_calibration.py']
    code_hashes={str(p.relative_to(ROOT)):sha256(p) for p in code_paths}
    try:
        torch.set_num_threads(4)
        samples=pd.read_csv(track(args.csv),dtype={'pa_id':str,'ct_id':str})
        ridge_validation=json.loads(track(args.ridge_root/'validation.json').read_text())
        if ridge_validation['status']!='completed' or ridge_validation['seed']!=args.seed \
                or ridge_validation['source_sha256'][str(args.csv.resolve())]!=sha256(args.csv):
            raise ValueError('Ridge protocol/cohort/seed does not match')
        if len(samples)!=721 or samples.event.sum()!=251 or samples.pa_id.duplicated().any() or set(samples.split)!=set(range(5)):
            raise ValueError('Expected unchanged 721 patients/251 events/five folds')
        if not samples.event.isin([0,1]).all() or not np.isfinite(samples.time).all() or (samples.time<0).any():
            raise ValueError('Invalid survival labels')
        samples['time']=samples.time.to_numpy(np.float32).astype(float)
        expected_features=json.loads(track(args.ridge_root/'private/feature_provenance.json').read_text())
        for fold in range(5):
            train=np.flatnonzero(samples.split.ne(fold))
            roles=inner_roles(samples,train,args.seed)
            saved=pd.read_csv(track(args.ridge_root/f'private/fold_{fold}/inner_roles.csv'),dtype={'case_id':str})
            if saved.case_id.duplicated().any() or set(saved.case_id)!=set(samples.iloc[train].pa_id):
                raise ValueError('Ridge inner training cohort differs')
            if not np.array_equal(saved.set_index('case_id').loc[samples.iloc[train].pa_id,'inner_fold'].to_numpy(),roles):
                raise ValueError('Inner roles differ from Ridge')
        print('Stage6 | Raw UNI1024 | ABMIL | fixed3-inner/5-outer CV | 15 inner + 5 fresh outer fits',flush=True)
        bags=[]
        for i,path in enumerate(samples.pa_path):
            if expected_features.get(path)!=sha256(path): raise ValueError('UNI features differ from Stage5')
            bags.append(load_path_feature(path,1024))
            if (i+1)%100==0: print(f'Read-only full-WSI CPU cache: {i+1}/{len(samples)}',flush=True)
        rows=[]
        for fold in range(5):
            status.update(active_outer_fold=fold)
            (args.output/'status.json').write_text(json.dumps(status,indent=2))
            rows.append(run_outer(bags,samples,fold,args.output/f'fold_{fold}',args.seed,args.device))
            pd.DataFrame(rows).to_csv(args.output/'metrics.csv',index=False)
            status['completed_outer_folds']=fold+1
            (args.output/'status.json').write_text(json.dumps(status,indent=2))
        from nested_report import summarize
        summarize(args.output,args.ridge_root,samples,args.seed,track)
        for path,value in sources.items():
            if sha256(path)!=value: raise RuntimeError('A source changed during training')
        for path in samples.pa_path:
            if sha256(path)!=expected_features[path]: raise RuntimeError('UNI source changed during training')
        if any(sha256(ROOT/p)!=value for p,value in code_hashes.items()): raise RuntimeError('Code changed during training')
        result=dict(status='completed',neural_fits=20,inner_fits=15,outer_fits=5,
                    n_patients=721,n_events=251,feature_dim=1024,n_patches=sum(len(x) for x in bags),
                    candidate_steps=[9,18,36,54,72,90,117,144,180],seed=args.seed,
                    dropout=.25,cox_batch_size=64,lr=1e-4,weight_decay=5e-4,
                    outer_val_evaluations_per_fold=1,stage5_inner_roles_exact=True,
                    source_sha256=sources,source_hashes_unchanged=True,code_sha256=code_hashes,
                    torch_version=str(torch.__version__),device=args.device)
        (args.output/'validation.json').write_text(json.dumps(result,indent=2))
        status.update(status='completed')
        print('Stage6 completed all20 fits. Matched nested comparison and paired interval saved. No additional experiments started.',flush=True)
    except Exception as exc:
        status.update(status='failed',error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        (args.output/'status.json').write_text(json.dumps(status,indent=2))


if __name__=='__main__': main()
