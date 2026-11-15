"""Experiment16: isolated raw UNI lightweight ABMIL, matched to Stage6 nested CV."""

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'tests/fold_difficulty_analysis/pathology_stage6'))

from light_model import make_model,parameter_counts
import nested_train as core
from nested_train import joint_gradients


def run_outer(bags,samples,fold,directory,seed=123,device='cuda:0',attention_dim=32,
              candidates=(9,18,36,54,72,90,117,144,180),cox_batch_size=64):
    directory=Path(directory)
    if directory.exists() and any(directory.iterdir()): raise FileExistsError('Refusing to overwrite fold results')
    (directory/'private').mkdir(parents=True)
    train=np.flatnonzero(samples.split.ne(fold));val=np.flatnonzero(samples.split.eq(fold))
    roles=core.inner_roles(samples,train,seed)
    pd.DataFrame(dict(case_id=samples.iloc[train].pa_id.to_numpy(),inner_fold=roles)).to_csv(directory/'private/inner_roles.csv',index=False)
    curves,audit=[],[]
    for inner in range(3):
        fit,held=train[roles!=inner],train[roles==inner]
        if set(fit)&set(val) or set(held)&set(val): raise RuntimeError('Outer Val leaked into inner fit')
        model=make_model(seed,device,attention_dim)
        print(f'Fold {fold} Inner {inner}: FIT={len(fit)}, HELD={len(held)}, light ABMIL, {max(candidates)}-step fixed grid',flush=True)
        meta,part=core.train_steps(model,bags,samples,fit,held,candidates,seed,directory/'private'/f'inner_{inner}',cox_batch_size)
        audit.append(dict(role='inner',inner_fold=inner,**meta))
        for row in part:
            # The shared evaluator uses baseline-dimension names; save the true light dimensions.
            row['pool128_patient_variance_mean']=row.pop('pool512_patient_variance_mean')
            row['post64_patient_variance_mean']=row.pop('post256_patient_variance_mean')
            curves.append(dict(inner_fold=inner,**row))
        pd.DataFrame(curves).to_csv(directory/'inner_curves.csv',index=False)
        del model
    chosen,selection=core.select_step(pd.DataFrame(curves),candidates)
    selection['selected']=selection.step.eq(chosen)
    selection.to_csv(directory/'step_selection.csv',index=False)
    print(f'Fold {fold}: selected Step={chosen}; fresh light Outer refit; Outer Val unused',flush=True)
    model=make_model(seed,device,attention_dim)
    counts=parameter_counts(model)
    meta,_=core.train_steps(model,bags,samples,train,None,(chosen,),seed,directory/'private/outer',cox_batch_size)
    audit.append(dict(role='outer',inner_fold=-1,**meta))
    predictions={};diagnostics={}
    for split,indices in [('train',train),('val',val)]:
        result,values=core.evaluate(model,bags,samples,indices,'raw',directory/'private/outer'/f'{split}_predictions.csv')
        diagnostics[split+'_loss']=result['cox_loss']
        diagnostics[split+'_risk_sd']=result['risk_sd']
        part=samples.iloc[indices]
        predictions[split]=values['risk'].numpy()
        pd.DataFrame({'case_id':part.pa_id.to_numpy(),'dfs.month':part.time.to_numpy(),
                      'dfs.status':part.event.to_numpy(),'risk_score':predictions[split]}).to_csv(
                          directory/'private/outer'/f'{split}_predictions.csv',index=False)
    metrics=core.evaluate_predictions(samples.iloc[train],samples.iloc[val],predictions['train'],predictions['val'])
    state=torch.load(directory/'private/outer/model.pth',map_location=device,weights_only=True)
    restored=make_model(seed,device,attention_dim);restored.load_state_dict(state['model'],strict=True)
    model.eval();restored.eval()
    with torch.no_grad():
        if not torch.equal(model(bags[train[0]].to(device))[0],restored(bags[train[0]].to(device))[0]):
            raise RuntimeError('Reloaded light checkpoint differs')
    pd.DataFrame(audit).to_csv(directory/'fit_audit.csv',index=False)
    row=dict(fold=fold,selected_step=chosen,n_train=len(train),n_val=len(val),attention_dim=attention_dim,
             **counts,**metrics,**diagnostics)
    pd.DataFrame([row]).to_csv(directory/'metrics.csv',index=False)
    (directory/'completed.json').write_text(json.dumps(dict(**row,neural_fits=4,outer_val_evaluations=1,checkpoint_reload_exact=True),indent=2))
    print(f"Fold {fold} completed | Train C-index={metrics['train_cindex']:.6f} | Val C-index={metrics['val_cindex']:.6f} | Gap={metrics['gap']:.6f}",flush=True)
    return row


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv',type=Path,default=ROOT/'data/trident/features_uni_v1/all_label_roi64.csv')
    parser.add_argument('--reference',type=Path,default=ROOT/'tests/fold_difficulty_analysis/pathology_stage6/results')
    parser.add_argument('--output',type=Path,default=Path(__file__).parent/'results')
    parser.add_argument('--attention_dim',type=int,choices=[32,128],default=32)
    parser.add_argument('--seed',type=int,default=123)
    parser.add_argument('--device',default='cuda:0')
    args=parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()): raise FileExistsError('Use a new --output; old results are never overwritten')
    args.output.mkdir(parents=True,exist_ok=True)
    status=dict(status='running',pid=os.getpid(),completed_outer_folds=0,seed=args.seed)
    (args.output/'status.json').write_text(json.dumps(status,indent=2))
    sources={}
    def track(path):
        path=Path(path).resolve();value=core.sha256(path)
        if str(path) in sources and sources[str(path)]!=value: raise RuntimeError('Source changed during reading')
        sources[str(path)]=value
        return path
    code_paths=[Path(__file__),Path(__file__).with_name('light_model.py'),Path(__file__).with_name('compare_light.py'),
                ROOT/'model/abmil_cox.py',ROOT/'utils.py',ROOT/'dataset.py',ROOT/'loss/survival_loss.py',
                ROOT/'tests/fold_difficulty_analysis/pathology_stage6/nested_train.py',
                ROOT/'tests/fold_difficulty_analysis/pathology_stage2/diagnose.py',
                ROOT/'tests/fold_difficulty_analysis/pathology_stage3/train_scale.py',
                ROOT/'tests/fold_difficulty_analysis/pathology_stage3/scale_eval.py',
                ROOT/'tests/fold_difficulty_analysis/pathology_stage5/run_stage5.py',
                ROOT/'tests/15_pathology_query_ct_tokens/risk_calibration.py']
    hashes={str(p.relative_to(ROOT)):core.sha256(p) for p in code_paths}
    try:
        torch.set_num_threads(4)
        reference=json.loads(track(args.reference/'validation.json').read_text())
        if reference['status']!='completed' or reference['seed']!=args.seed:
            raise ValueError('Need a completed Stage6 raw UNI reference with the same seed')
        for name,value in hashes.items():
            if name in reference['code_sha256'] and reference['code_sha256'][name]!=value:
                raise ValueError(f'Shared training/evaluation source differs from Stage6: {name}')
        if any(reference[k]!=v for k,v in dict(dropout=.25,cox_batch_size=64,lr=1e-4,weight_decay=5e-4,feature_dim=1024).items()):
            raise ValueError('Stage6 reference configuration differs')
        if reference['candidate_steps']!=[9,18,36,54,72,90,117,144,180] \
                or reference['code_sha256']['model/abmil_cox.py']!=core.sha256(ROOT/'model/abmil_cox.py'):
            raise ValueError('Candidate grid or original ABMIL source differs from Stage6')
        if reference['source_sha256'][str(args.csv.resolve())]!=core.sha256(args.csv): raise ValueError('Raw UNI cohort differs from reference')
        samples=pd.read_csv(track(args.csv),dtype={'pa_id':str,'ct_id':str})
        if len(samples)!=721 or samples.event.sum()!=251 or samples.pa_id.duplicated().any() or set(samples.split)!=set(range(5)):
            raise ValueError('Expected original 721 patients, 251 events and five saved folds')
        if not samples.event.isin([0,1]).all() or not np.isfinite(samples.time).all() or (samples.time<0).any(): raise ValueError('Invalid labels')
        samples['time']=samples.time.to_numpy(np.float32).astype(float)
        for fold in range(5):
            train=np.flatnonzero(samples.split.ne(fold));roles=core.inner_roles(samples,train,args.seed)
            saved=pd.read_csv(track(args.reference/f'fold_{fold}/private/inner_roles.csv'),dtype={'case_id':str})
            if saved.case_id.duplicated().any() or set(saved.case_id)!=set(samples.iloc[train].pa_id): raise ValueError('Inner cohort differs')
            if not np.array_equal(saved.set_index('case_id').loc[samples.iloc[train].pa_id,'inner_fold'].to_numpy(),roles): raise ValueError('Inner roles differ')
        feature_map=ROOT/'tests/fold_difficulty_analysis/pathology_stage5/results/private/feature_provenance.json'
        if reference['source_sha256'][str(feature_map.resolve())]!=core.sha256(feature_map): raise ValueError('Feature provenance differs from reference')
        expected=json.loads(track(feature_map).read_text())
        probe=make_model(args.seed,args.device,args.attention_dim);counts=parameter_counts(probe);del probe
        config=dict(seed=args.seed,feature_dim=1024,hidden_dim=128,attention_dim=args.attention_dim,post_hidden_dim=64,
                     feature_source='raw_all_patch_UNI',l2_normalize=False,survival_loss='cox',dropout=.25,
                     cox_batch_size=64,lr=1e-4,weight_decay=5e-4,protocol='Stage6 matched nested CV',
                     candidate_steps=[9,18,36,54,72,90,117,144,180],device=args.device,
                     torch_version=str(torch.__version__),tf32=False,**counts)
        (args.output/'run_config.json').write_text(json.dumps(config,indent=2))
        print(f"Raw UNI light ABMIL | parameters {counts['full_parameters']} -> {counts['light_parameters']} ({100*counts['reduction_fraction']:.3f}% fewer)",flush=True)
        bags=[]
        for i,path in enumerate(samples.pa_path):
            if expected.get(path)!=core.sha256(path): raise ValueError('Raw full-WSI features differ from Stage6')
            bags.append(core.load_path_feature(path,1024))
            if (i+1)%100==0: print(f'Full UNI WSI cache: {i+1}/721',flush=True)
        rows=[]
        for fold in range(5):
            status['active_outer_fold']=fold
            (args.output/'status.json').write_text(json.dumps(status,indent=2))
            rows.append(run_outer(bags,samples,fold,args.output/f'fold_{fold}',args.seed,args.device,args.attention_dim))
            pd.DataFrame(rows).to_csv(args.output/'metrics.csv',index=False)
            status['completed_outer_folds']=fold+1
            (args.output/'status.json').write_text(json.dumps(status,indent=2))
        from compare_light import summarize
        summarize(args.output,args.reference,samples,args.seed,track)
        if any(core.sha256(path)!=value for path,value in sources.items()): raise RuntimeError('Source file changed during training')
        if any(core.sha256(path)!=expected[path] for path in samples.pa_path): raise RuntimeError('Raw UNI files changed during training')
        if any(core.sha256(ROOT/p)!=value for p,value in hashes.items()): raise RuntimeError('Experiment code changed during training')
        validation=dict(status='completed',neural_fits=20,inner_fits=15,outer_fits=5,n_patients=721,n_events=251,
                        n_patches=sum(len(x) for x in bags),outer_val_evaluations_per_fold=1,
                        source_hashes_unchanged=True,source_sha256=sources,code_sha256=hashes,**config)
        (args.output/'validation.json').write_text(json.dumps(validation,indent=2))
        status['status']='completed'
        print('Light ABMIL completed all five nested folds; matched full ABMIL comparison saved.',flush=True)
    except Exception as exc:
        status.update(status='failed',error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        (args.output/'status.json').write_text(json.dumps(status,indent=2))


if __name__=='__main__': main()
