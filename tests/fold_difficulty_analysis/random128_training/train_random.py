"""Train-only Random128 raw UNI ABMIL, strictly matched to Stage6 Nested CV."""

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(Path(__file__).parent.parent/'pathology_stage6'))
import nested_train as core
from model.abmil_cox import ABMIL,ABMIL_RandomSample
from utils import seed_everything
from loss.survival_loss import cox_loss

# Stage2 also has a module named diagnose; load the frozen subset helper by its unique name.
spec=importlib.util.spec_from_file_location('frozen_patch_diagnostics',Path(__file__).parent.parent/'patch_redundancy/diagnose.py')
patch=importlib.util.module_from_spec(spec);spec.loader.exec_module(patch)


def fresh_model(seed,device):
    seed_everything(seed)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    return ABMIL_RandomSample(in_dim=1024,hidden_dim=512,attention_dim=128,k=128,dropout=.25).to(device)


@torch.inference_mode()
def evaluate_policies(model,bags,samples,indices,seed,repeats=30):
    model.eval();device=next(model.parameters()).device
    full=np.empty(len(indices),dtype=np.float32)
    random=np.empty((repeats,len(indices)),dtype=np.float32)
    for position,index in enumerate(indices):
        x=bags[index].to(device)
        # The inherited evaluator disables sampling; base forward only exposes cached per-patch H.
        output,_,_,h=ABMIL.forward(model,x,return_patch_features=True)
        full[position]=output.item();logits=model.attention(h)
        for repeat in range(repeats):
            if len(x)<=128:random[repeat,position]=full[position]
            else:
                selected=patch.select_indices(logits[0,:,0],'random',128,seed,samples.iloc[index].pa_id,repeat)
                random[repeat,position]=patch.subset_risk(model,h,logits,selected)
    if not np.isfinite(full).all() or not np.isfinite(random).all():raise RuntimeError('Nonfinite policy predictions')
    return full,random


def run_outer(bags,samples,fold,directory,seed=123,device='cuda:0',
              candidates=(9,18,36,54,72,90,117,144,180),cox_batch_size=64,repeats=30):
    directory=Path(directory)
    if directory.exists() and any(directory.iterdir()):raise FileExistsError('Refusing to overwrite fold results')
    (directory/'private').mkdir(parents=True)
    train=np.flatnonzero(samples.split.ne(fold));val=np.flatnonzero(samples.split.eq(fold))
    roles=core.inner_roles(samples,train,seed)
    pd.DataFrame(dict(case_id=samples.iloc[train].pa_id.to_numpy(),inner_fold=roles)).to_csv(directory/'private/inner_roles.csv',index=False)
    curves,audit=[],[]
    for inner in range(3):
        fit,held=train[roles!=inner],train[roles==inner]
        if set(fit)&set(val) or set(held)&set(val):raise RuntimeError('Outer Val entered inner fit')
        model=fresh_model(seed,device)
        print(f'Fold {fold} Inner {inner}: FIT={len(fit)}, HELD={len(held)} | train Random128 | select on Full WSI',flush=True)
        meta,part=core.train_steps(model,bags,samples,fit,held,candidates,seed,directory/'private'/f'inner_{inner}',cox_batch_size)
        audit.append(dict(role='inner',inner_fold=inner,**meta));curves.extend(dict(inner_fold=inner,**row) for row in part)
        pd.DataFrame(curves).to_csv(directory/'inner_curves.csv',index=False)
        del model
    chosen,selection=core.select_step(pd.DataFrame(curves),candidates)
    selection['selected']=selection.step.eq(chosen);selection.to_csv(directory/'step_selection.csv',index=False)
    print(f'Fold {fold}: selected Step={chosen}; fresh Random128 outer refit; Outer Val unused',flush=True)
    model=fresh_model(seed,device)
    meta,_=core.train_steps(model,bags,samples,train,None,(chosen,),seed,directory/'private/outer',cox_batch_size)
    audit.append(dict(role='outer',inner_fold=-1,**meta))
    predictions={};diagnostics={}
    for split,indices in [('train',train),('val',val)]:
        full,random=evaluate_policies(model,bags,samples,indices,seed,repeats)
        predictions[split]=(full,random);part=samples.iloc[indices]
        pd.DataFrame({'case_id':part.pa_id.to_numpy(),'dfs.month':part.time.to_numpy(),
                      'dfs.status':part.event.to_numpy(),'risk_score':full}).to_csv(directory/'private/outer'/f'{split}_predictions.csv',index=False)
        records=[pd.DataFrame(dict(case_id=part.pa_id.to_numpy(),time=part.time.to_numpy(),event=part.event.to_numpy(),
                                  repeat=repeat,risk_score=random[repeat])) for repeat in range(repeats)]
        pd.concat(records,ignore_index=True).to_csv(directory/'private/outer'/f'{split}_random_predictions.csv',index=False)
        diagnostics[split+'_loss']=float(cox_loss(torch.from_numpy(full),torch.tensor(part.time.to_numpy(),dtype=torch.float32),
                                                 torch.tensor(part.event.to_numpy(),dtype=torch.float32)))
        diagnostics[split+'_risk_sd']=float(full.std())
    metrics=core.evaluate_predictions(samples.iloc[train],samples.iloc[val],predictions['train'][0],predictions['val'][0])
    random_metrics=[]
    for repeat in range(repeats):
        values=core.evaluate_predictions(samples.iloc[train],samples.iloc[val],predictions['train'][1][repeat],predictions['val'][1][repeat])
        random_metrics.append(dict(fold=fold,repeat=repeat,**values))
    pd.DataFrame(random_metrics).to_csv(directory/'random_metrics.csv',index=False)
    restored=fresh_model(seed,device)
    restored.load_state_dict(torch.load(directory/'private/outer/model.pth',weights_only=True,map_location=device)['model'],strict=True)
    model.eval();restored.eval()
    with torch.no_grad():
        if not torch.equal(model(bags[train[0]].to(device))[0],restored(bags[train[0]].to(device))[0]):
            raise RuntimeError('Own checkpoint reload differs')
    pd.DataFrame(audit).to_csv(directory/'fit_audit.csv',index=False)
    row=dict(fold=fold,selected_step=chosen,n_train=len(train),n_val=len(val),**metrics,**diagnostics)
    pd.DataFrame([row]).to_csv(directory/'metrics.csv',index=False)
    (directory/'completed.json').write_text(json.dumps(dict(**row,neural_fits=4,
        outer_checkpoint_selections=0,final_full_wsi_evaluations=1,final_prespecified_random_repeats=repeats,
        checkpoint_reload_exact=True),indent=2))
    print(f"Fold {fold} completed | Full WSI Train={metrics['train_cindex']:.6f} | Val={metrics['val_cindex']:.6f} | Gap={metrics['gap']:.6f}",flush=True)
    return row


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv',type=Path,default=ROOT/'data/trident/features_uni_v1/all_label_roi64.csv')
    parser.add_argument('--reference',type=Path,default=Path(__file__).parent.parent/'pathology_stage6/results')
    parser.add_argument('--output',type=Path,default=Path(__file__).parent/'results')
    parser.add_argument('--seed',type=int,default=123)
    parser.add_argument('--device',default='cuda:0')
    args=parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):raise FileExistsError('Use a new output; results are never overwritten')
    args.output.mkdir(parents=True)
    status=dict(status='running',pid=os.getpid(),completed_outer_folds=0,seed=args.seed)
    sources={}
    def track(path):
        path=Path(path).resolve();value=core.sha256(path)
        if str(path) in sources and sources[str(path)]!=value:raise RuntimeError('Source changed during reading')
        sources[str(path)]=value;return path
    code_paths=[Path(__file__),Path(__file__).with_name('report_random.py'),ROOT/'model/abmil_cox.py',ROOT/'utils.py',
                ROOT/'dataset.py',ROOT/'loss/survival_loss.py',Path(core.__file__),
                Path(core.joint_gradients.__code__.co_filename),Path(core.evaluate.__code__.co_filename),
                Path(core.evaluate_predictions.__code__.co_filename),Path(core.batch_schedule.__code__.co_filename),
                Path(patch.__file__),ROOT/'tests/15_pathology_query_ct_tokens/risk_calibration.py']
    hashes={str(path.relative_to(ROOT)):core.sha256(path) for path in code_paths}
    try:
        (args.output/'status.json').write_text(json.dumps(status,indent=2));torch.set_num_threads(4)
        reference=json.loads(track(args.reference/'validation.json').read_text())
        expected_config=dict(seed=args.seed,feature_dim=1024,dropout=.25,cox_batch_size=64,lr=1e-4,weight_decay=5e-4)
        if reference['status']!='completed' or any(reference[key]!=value for key,value in expected_config.items()):
            raise ValueError('Stage6 reference configuration differs')
        for name,value in hashes.items():
            if name in reference['code_sha256'] and reference['code_sha256'][name]!=value:raise ValueError(f'Shared source differs from Stage6: {name}')
        if reference['candidate_steps']!=[9,18,36,54,72,90,117,144,180]:raise ValueError('Stage6 candidate grid differs')
        if reference['source_sha256'][str(args.csv.resolve())]!=core.sha256(args.csv):raise ValueError('Cohort differs from Stage6')
        samples=pd.read_csv(track(args.csv),dtype={'pa_id':str,'ct_id':str})
        if len(samples)!=721 or samples.event.sum()!=251 or samples.pa_id.duplicated().any() or set(samples.split)!=set(range(5)):
            raise ValueError('Expected unchanged 721 patients / 251 events / saved five folds')
        if not samples.event.isin([0,1]).all() or not np.isfinite(samples.time).all() or (samples.time<0).any():raise ValueError('Invalid labels')
        samples['time']=samples.time.to_numpy(np.float32).astype(float)
        for fold in range(5):
            train=np.flatnonzero(samples.split.ne(fold));roles=core.inner_roles(samples,train,args.seed)
            saved=pd.read_csv(track(args.reference/f'fold_{fold}/private/inner_roles.csv'),dtype={'case_id':str})
            if saved.case_id.duplicated().any() or set(saved.case_id)!=set(samples.iloc[train].pa_id):raise ValueError('Inner cohort differs')
            if not np.array_equal(saved.set_index('case_id').loc[samples.iloc[train].pa_id,'inner_fold'].to_numpy(),roles):raise ValueError('Inner roles differ')
        provenance=Path(__file__).parent.parent/'pathology_stage5/results/private/feature_provenance.json'
        if reference['source_sha256'][str(provenance.resolve())]!=core.sha256(provenance):raise ValueError('Feature provenance differs')
        features=json.loads(track(provenance).read_text())
        probe=fresh_model(args.seed,args.device);original=core.fresh_model(1024,args.seed,args.device)
        if not all(torch.equal(probe.state_dict()[key],value) for key,value in original.state_dict().items()):raise RuntimeError('Initialization differs')
        count=sum(p.numel() for p in probe.parameters());del probe,original
        config=dict(**expected_config,train_k=128,eval_random_k=128,eval_random_repeats=30,
                    eval_random_seed=args.seed,inner_selection_policy='Full WSI',primary_final_policy='Full WSI',
                    l2_normalize=False,feature_source='raw_all_patch_UNI',parameters=count,
                    candidate_steps=reference['candidate_steps'],device=args.device,torch_version=str(torch.__version__),tf32=False)
        (args.output/'run_config.json').write_text(json.dumps(config,indent=2))
        print('Raw UNI Random128 training | ABMIL 722178 parameters | same Stage6 nested roles and optimization | 20 fits',flush=True)
        bags=[]
        for i,path in enumerate(samples.pa_path):
            if features.get(path)!=core.sha256(path):raise ValueError('Raw UNI feature differs from Stage6')
            bags.append(core.load_path_feature(path,1024))
            if (i+1)%100==0:print(f'Complete UNI CPU cache: {i+1}/721',flush=True)
        rows=[]
        for fold in range(5):
            status['active_outer_fold']=fold;(args.output/'status.json').write_text(json.dumps(status,indent=2))
            rows.append(run_outer(bags,samples,fold,args.output/f'fold_{fold}',args.seed,args.device))
            pd.DataFrame(rows).to_csv(args.output/'metrics.csv',index=False)
            status['completed_outer_folds']=fold+1;(args.output/'status.json').write_text(json.dumps(status,indent=2))
        from report_random import summarize
        summarize(args.output,args.reference,samples,args.seed,track)
        if any(core.sha256(path)!=value for path,value in sources.items()) or any(core.sha256(path)!=features[path] for path in samples.pa_path):
            raise RuntimeError('Input source changed during experiment')
        if any(core.sha256(ROOT/path)!=value for path,value in hashes.items()):raise RuntimeError('Experiment code changed during training')
        validation=dict(status='completed',neural_fits=20,inner_fits=15,outer_fits=5,n_patients=721,n_events=251,
                        n_patches=sum(len(bag) for bag in bags),same_initialization_verified=True,outer_checkpoint_selections=0,
                        source_sha256=sources,code_sha256=hashes,source_hashes_unchanged=True,**config)
        (args.output/'validation.json').write_text(json.dumps(validation,indent=2));status['status']='completed'
        print('Random128 nested training completed; Full WSI and fixed random evaluations reported separately.',flush=True)
    except Exception as exc:
        status.update(status='failed',error=f'{type(exc).__name__}: {exc}');raise
    finally:
        (args.output/'status.json').write_text(json.dumps(status,indent=2))


if __name__=='__main__':main()
