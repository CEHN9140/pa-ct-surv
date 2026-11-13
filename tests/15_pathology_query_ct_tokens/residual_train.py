"""Nested five-fold residual teachers: joint PA/CT training, no unimodal warm-start."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import torch
import yaml
from sksurv.metrics import concordance_index_censored

from dataset import Pa_CT_Dataset, build_dataloader, cox_batch_sizes, cv_splits
from pact_train import build_pact_optimizer
from residual_teacher import ResidualTeacher, residual_objective, pair_changes
from risk_calibration import paired_comparison
from utils import save_cv_results, save_run_config, seed_everything
from visualization.survival_plots import analyze_fold


def select_residual_config(records):
    choices=[]
    frame=pd.DataFrame(records)
    expected=set(frame.inner_fold)
    for name,rows in frame.groupby('name'):
        if set(rows.inner_fold)!=expected or rows.inner_fold.duplicated().any():
            raise ValueError('All residual candidates require the same complete inner folds')
        if not np.isfinite(rows[['held_cindex','selected_epoch','lambda_r']]).all().all():
            raise ValueError('Nonfinite inner model selection data')
        choices.append(dict(name=name,lambda_r=float(rows.lambda_r.iloc[0]),
            mean_inner_cindex=float(rows.held_cindex.mean()),epochs=int(np.median(rows.selected_epoch))))
    return max(choices,key=lambda row:(row['mean_inner_cindex'],row['lambda_r']))


def source_partitions(samples,outer_train,outer_val,roles_path):
    ids=(samples.pa_id+'|'+samples.ct_id).to_numpy()
    lookup={identity:i for i,identity in enumerate(ids)}
    allowed=set(ids[outer_train]);excluded=set(ids[outer_val])
    table=pd.read_csv(roles_path,dtype={'case_id':str})
    result=[];all_held=[]
    for inner,rows in table.groupby('inner_fold',sort=True):
        groups={role:group.case_id.tolist() for role,group in rows.groupby('role')}
        if (set(groups)!={'fit','stop','oof'} or rows.case_id.duplicated().any()
                or set(rows.case_id)!=allowed or set(rows.case_id)&excluded):
            raise ValueError('Source inner fit/stop/OOF roles are not a disjoint partition of outer training')
        all_held.extend(groups['oof'])
        result.append((int(inner),*[np.array([lookup[x] for x in groups[role]]) for role in ('fit','stop','oof')]))
    if len(result)!=3 or len(all_held)!=len(allowed) or set(all_held)!=allowed:
        raise ValueError('Require exactly three source inner folds and honest OOF coverage')
    return result


@torch.inference_mode()
def predict_teacher(model,loader,device):
    model.eval()
    records=[]
    for ct,pa,event,time,identities in loader:
        total,delta,path=model(ct.to(device),pa.to(device))
        for case,t,e,r,d,p in zip(identities,time,event,total.cpu(),delta.cpu(),path.cpu()):
            records.append(dict(case_id=str(case),**{'dfs.month':float(t),'dfs.status':int(e)},
                                risk_score=float(r),ct_correction=float(d),pathology_risk=float(p)))
    return pd.DataFrame(records)


def frame_objective(frame,args,penalty):
    r,d,p,t,e=[torch.tensor(frame[key].to_numpy(),dtype=torch.float64) for key in
        ('risk_score','ct_correction','pathology_risk','dfs.month','dfs.status')]
    return {key:float(value) for key,value in residual_objective(r,d,p,t,e,args.lambda_pa,penalty).items()}


def fit_teacher(dataset,fit,stop,args,device,checkpoint,penalty,fixed_epochs=None,
                model_factory=ResidualTeacher):
    if (stop is None)==(fixed_epochs is None):
        raise ValueError('Use inner stopping OR fixed-epoch outer refitting')
    seed_everything(args.seed)
    model=model_factory(ct_pretrained_path=args.ct_pretrained_path,dropout=args.dropout).to(device)
    optimizer=build_pact_optimizer(model,args.lr,args.weight_decay,args.ct_backbone_lr)
    if not all(p.requires_grad for p in model.parameters()):raise ValueError('Both branches must remain trainable')
    checkpoint=Path(checkpoint);checkpoint.mkdir(parents=True,exist_ok=True)
    train=build_dataloader(dataset,indices=fit,batch_size=1,shuffle=True,num_workers=args.num_workers)
    stopping=None if stop is None else build_dataloader(dataset,indices=stop,batch_size=1,num_workers=args.num_workers)
    epochs=args.num_epochs if fixed_epochs is None else int(fixed_epochs)
    ends=set(np.cumsum(cox_batch_sizes(len(train),args.cox_batch_size)))
    best,selected,wait,history=-np.inf,epochs,0,[]
    for epoch in range(1,epochs+1):
        model.train();optimizer.zero_grad(set_to_none=True)
        buffers=[[],[],[],[],[]];online=[[],[],[]];losses=[]
        for step,(ct,pa,event,time,_) in enumerate(train,1):
            total,delta,path=model(ct.to(device),pa.to(device))
            for target,value in zip(buffers,(total,delta,path,time.to(device),event.to(device))):target.append(value.reshape(-1))
            for target,value in zip(online,(total.detach().cpu(),time,event)):target.append(value.reshape(-1))
            if step in ends:
                values=residual_objective(*[torch.cat(x) for x in buffers],args.lambda_pa,penalty)
                values['total'].backward();optimizer.step();optimizer.zero_grad(set_to_none=True)
                losses.append({key:float(value.detach()) for key,value in values.items()})
                buffers=[[],[],[],[],[]]
        if any(buffers):raise RuntimeError('Unconsumed Cox tail')
        row=dict(epoch=epoch,optimizer_updates=len(losses),
                 **{'train_'+key:float(np.mean([x[key] for x in losses])) for key in losses[0]})
        row['train_loss']=row['train_total']
        risks,times,events=[torch.cat(x).numpy() for x in online]
        row['train_cindex']=float(concordance_index_censored(events.astype(bool),times,risks)[0])
        line=(f'Epoch {epoch}/{epochs} | Train Loss: {row["train_total"]:.4f} | '
              f'Train C-index: {row["train_cindex"]:.4f} | Fused Cox: {row["train_fused_cox"]:.4f} | '
              f'PA Cox: {row["train_pathology_cox"]:.4f} | CT correction L2: {row["train_correction_l2"]:.6f}')
        if stopping is not None:
            frame=predict_teacher(model,stopping,device)
            row.update({'val_'+key:value for key,value in frame_objective(frame,args,penalty).items()})
            row['val_loss']=row['val_total']
            row['val_cindex']=float(concordance_index_censored(frame['dfs.status'].astype(bool),frame['dfs.month'],frame.risk_score)[0])
            line+=(f' | Val Loss: {row["val_total"]:.4f} | Val C-index: {row["val_cindex"]:.4f}'
                   f' | Val Fused Cox: {row["val_fused_cox"]:.4f} | Val PA Cox: {row["val_pathology_cox"]:.4f}'
                   f' | Val CT correction L2: {row["val_correction_l2"]:.6f}')
            if row['val_cindex']>best:
                best,selected,wait=row['val_cindex'],epoch,0
                torch.save({k:v.detach().cpu() for k,v in model.state_dict().items()},checkpoint/'best_model.pth')
            else:wait+=1
        history.append(row);pd.DataFrame(history).to_csv(checkpoint/'training_history.csv',index=False)
        print(line,flush=True)
        if stopping is not None and args.patience>0 and wait>=args.patience:break
    if stopping is None:
        torch.save({k:v.detach().cpu() for k,v in model.state_dict().items()},checkpoint/'best_model.pth')
    else:model.load_state_dict(torch.load(checkpoint/'best_model.pth',map_location=device,weights_only=True),strict=True)
    metadata=dict(selected_epoch=int(selected),epochs_completed=epoch,lambda_r=float(penalty),
        selection='fixed_from_inner_median' if stop is None else 'inner_stop_cindex',
        n_fit=len(fit),n_stop=0 if stop is None else len(stop),
        cox_group_sizes=cox_batch_sizes(len(train),args.cox_batch_size),
        initialization=getattr(model,'initialization_description',
            'new ABMIL + public MedicalNet; zero final CT correction layer; no downstream checkpoint'))
    (checkpoint/'checkpoint_metadata.yaml').write_text(yaml.safe_dump(metadata))
    return model,metadata


def ordered_teacher(model,dataset,indices,args,device):
    loader=build_dataloader(dataset,indices=indices,batch_size=1,num_workers=args.num_workers)
    frame=predict_teacher(model,loader,device)
    rows=dataset.samples.iloc[indices]
    np.testing.assert_array_equal(frame.case_id,(rows.pa_id+'|'+rows.ct_id).to_numpy())
    np.testing.assert_allclose(frame['dfs.month'],rows.time,rtol=1e-7,atol=1e-5)
    np.testing.assert_array_equal(frame['dfs.status'],rows.event)
    np.testing.assert_allclose(frame.risk_score,frame.pathology_risk+frame.ct_correction,rtol=1e-5,atol=1e-7)
    return frame


def export_teacher(name,fold,frames,penalty,args):
    destination=Path(args.results_root)/name/f'fold_{fold}';destination.mkdir(parents=True,exist_ok=True)
    frames['train'].to_csv(destination/'train_predictions.csv',index=False)
    metrics=analyze_fold(frames['train'],frames['val'],destination)
    for split,frame in frames.items():
        loss=frame_objective(frame,args,penalty)
        metrics.update({split+'_'+key:value for key,value in loss.items()})
        metrics[split+'_loss']=loss['total']
    val=frames['val'];time=val['dfs.month'].to_numpy();event=val['dfs.status'].to_numpy()
    within=pair_changes(time,event,val.pathology_risk,val.risk_score)
    baseline=pd.read_csv(Path(args.source_run)/f'pathology/fold_{fold}/val_predictions.csv',dtype={'case_id':str}).set_index('case_id').loc[val.case_id]
    np.testing.assert_allclose(baseline['dfs.month'],val['dfs.month'])
    np.testing.assert_array_equal(baseline['dfs.status'],event)
    external=pair_changes(time,event,baseline.risk_score,val.risk_score)
    details=dict(within_teacher=within,versus_independent_pathology_baseline=external,
        ct_correction_mean=float(val.ct_correction.mean()),ct_correction_std=float(val.ct_correction.std(ddof=0)),
        ct_correction_rms=float(np.sqrt(np.mean(val.ct_correction**2))),
        caveat='Within-teacher risk subtraction is an inference ablation; independent model differences also include PA training changes.')
    (destination/'pair_changes.json').write_text(json.dumps(details,indent=2))
    comparison=val.copy();comparison['baseline_pathology']=baseline.risk_score.to_numpy()
    print(f'{name} fold{fold} | Train C-index: {metrics["train_cindex"]:.4f} | Val C-index: {metrics["val_cindex"]:.4f} | '
          f'Corrected/broken pairs versus PA baseline: {external["corrected_pairs"]}/{external["broken_pairs"]}',flush=True)
    return dict(fold=fold,**metrics),comparison


def run(args):
    source=Path(args.source_run);source_config=yaml.safe_load((source/'run_config.yaml').read_text())
    if not (source/'completed.json').is_file() or source_config['fusion_type']!='risk_increment_nested_cv':
        raise ValueError('Require the completed nested-risk source for fixed cohort/roles/reference predictions')
    args.data_dir=source_config['data_dir'];args.ct_pretrained_path=source_config['ct_pretrained_path']
    args.cohort_sha256=hashlib.sha256((Path(args.data_dir)/'all_label_roi64.csv').read_bytes()).hexdigest()
    args.public_pretrained_sha256=hashlib.sha256(Path(args.ct_pretrained_path).read_bytes()).hexdigest()
    if args.cohort_sha256!=source_config['cohort_sha256'] or args.public_pretrained_sha256!=source_config['public_pretrained_sha256']:
        raise ValueError('Source cohort/public pretrained weights changed')
    args.source_config_sha256=hashlib.sha256((source/'run_config.yaml').read_bytes()).hexdigest()
    for path in (Path(args.results_root),Path(args.checkpoint_root)):
        if path.exists():raise FileExistsError(f'Refusing to overwrite {path}')
    dataset=Pa_CT_Dataset(args.data_dir,roi_size=64,feature_dim=1280)
    if len(dataset)!=721 or not dataset.samples.pa_id.is_unique or not dataset.samples.ct_id.is_unique:
        raise ValueError('Require original721 unique PA/CT pairs')
    device=torch.device(args.device)
    if device.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA requested but unavailable')
    torch.set_num_threads(4)
    args.fusion_type='residual_teacher_nested_cv';args.survival_loss='cox'
    args.pathology_patch_l2=True;args.ct_pooled_l2=False;args.ct_bn_stats='fixed_public_statistics'
    args.initialization='new ABMIL + public MedicalNet; no project unimodal checkpoints'
    args.correction_head=[512,128,1];args.correction_penalty='mean squared scalar CT correction'
    save_run_config(args,'pact','residual_teacher')
    configurations={f'residual_r{x:g}'.replace('.','p'):x for x in args.correction_weights}
    metrics={name:[] for name in [*configurations,'inner_selected']}
    comparisons={name:[] for name in metrics}
    print(f'Using Device: {device} | Loaded721 paired patients | Penalties: {args.correction_weights}',flush=True)
    print(f'Joint PA/CT; public pretrained CT only; PA auxiliary={args.lambda_pa:g}; '
          f'Cox accumulation={args.cox_batch_size}; fixed BN; no CT-only Cox.',flush=True)
    for fold,(training,validation) in enumerate(cv_splits(dataset.samples)):
        if fold not in args.outer_folds:continue
        partitions=source_partitions(dataset.samples,training,validation,source/f'fold_{fold}/inner_split_roles.csv')
        rows=[]
        for name,penalty in configurations.items():
            for inner,fit,stop,held in partitions:
                print(f'\nOuter{fold} | {name} | Inner{inner} | fit={len(fit)}, stop={len(stop)}, held={len(held)}',flush=True)
                checkpoint=Path(args.checkpoint_root)/name/f'fold_{fold}/inner_{inner}'
                model,metadata=fit_teacher(dataset,fit,stop,args,device,checkpoint,penalty)
                prediction=ordered_teacher(model,dataset,held,args,device)
                score=concordance_index_censored(prediction['dfs.status'].astype(bool),prediction['dfs.month'],prediction.risk_score)[0]
                rows.append(dict(name=name,lambda_r=float(penalty),inner_fold=inner,
                    held_cindex=float(score),selected_epoch=metadata['selected_epoch']))
                local=Path(args.results_root)/name/f'fold_{fold}/inner_{inner}';local.mkdir(parents=True)
                prediction.to_csv(local/'val_predictions.csv',index=False)
                del model
                if device.type=='cuda':torch.cuda.empty_cache()
        selected=select_residual_config(rows)
        epochs={name:int(np.median([x['selected_epoch'] for x in rows if x['name']==name])) for name in configurations}
        selection=Path(args.results_root)/f'fold_{fold}';selection.mkdir()
        (selection/'selection.json').write_text(json.dumps(dict(primary=selected,per_candidate_epochs=epochs,inner_models=rows,
            policy='Select using inner held C-index; tie favors stronger penalty. Epoch=floor median inner-stop best. '
                   'Saved before all outer refits/predictions; outer metrics never choose lambda/epoch.'),indent=2))
        print(f'Outer{fold}: inner-selected primary={selected}; refit epochs={epochs}',flush=True)
        for name,penalty in configurations.items():
            model,_=fit_teacher(dataset,training,None,args,device,
                Path(args.checkpoint_root)/name/f'fold_{fold}/outer',penalty,fixed_epochs=epochs[name])
            frames={split:ordered_teacher(model,dataset,indices,args,device)
                    for split,indices in [('train',training),('val',validation)]}
            row,prediction=export_teacher(name,fold,frames,penalty,args)
            metrics[name].append(row);comparisons[name].append(prediction)
            if name==selected['name']:
                # Re-export the already selected model; never select using outer scores.
                primary_row,primary_prediction=export_teacher('inner_selected',fold,frames,penalty,args)
                metrics['inner_selected'].append(primary_row);comparisons['inner_selected'].append(primary_prediction)
            del model
            if device.type=='cuda':torch.cuda.empty_cache()
        (selection/'completed.json').write_text(json.dumps(dict(fold=fold,selected=selected),indent=2))
        for name,values in metrics.items():
            pd.DataFrame(values).to_csv(Path(args.results_root)/name/'partial_metrics.csv',index=False)
    if sorted(args.outer_folds)==list(range(5)):
        for name,values in metrics.items():
            save_cv_results(values,Path(args.results_root)/name)
            comparison=paired_comparison(comparisons[name],seed=args.seed,
                                        reference_columns={'pathology_baseline':'baseline_pathology'})
            (Path(args.results_root)/name/'paired_comparison.json').write_text(json.dumps(comparison,indent=2))
    (Path(args.results_root)/'completed.json').write_text(json.dumps(dict(models=list(metrics),folds=args.outer_folds,
        primary='inner_selected',secondary=list(configurations),
        caveat='Historical outer-stopped Concat/Cross-Attention are descriptive references only; not matched controls.'),indent=2))
    print('Residual teacher experiment complete.',flush=True)


def main():
    experiment=Path(__file__).resolve().parent
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source_run',default=str(experiment/'results/nested_risk_increment'))
    parser.add_argument('--results_root',default=str(experiment/'results/residual_teacher'))
    parser.add_argument('--checkpoint_root',default=str(experiment/'checkpoints/residual_teacher'))
    parser.add_argument('--correction_weights',nargs='+',type=float,default=[0.,.01])
    parser.add_argument('--lambda_pa',type=float,default=.3)
    parser.add_argument('--outer_folds',nargs='+',type=int,default=list(range(5)))
    parser.add_argument('--num_epochs',type=int,default=30)
    parser.add_argument('--patience',type=int,default=10)
    parser.add_argument('--cox_batch_size',type=int,default=32)
    parser.add_argument('--dropout',type=float,default=.25)
    parser.add_argument('--lr',type=float,default=1e-4)
    parser.add_argument('--ct_backbone_lr',type=float,default=1e-5)
    parser.add_argument('--weight_decay',type=float,default=5e-4)
    parser.add_argument('--seed',type=int,default=123)
    parser.add_argument('--num_workers',type=int,default=8)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    args=parser.parse_args()
    if (args.num_epochs<1 or args.cox_batch_size<1 or args.patience<0 or args.num_workers<0 or not 0<=args.dropout<1
        or not args.outer_folds or len(set(args.outer_folds))!=len(args.outer_folds) or not set(args.outer_folds).issubset(range(5))
        or not args.correction_weights or len(set(args.correction_weights))!=len(args.correction_weights)
        or not np.isfinite([*args.correction_weights,args.lambda_pa,args.lr,args.ct_backbone_lr,args.weight_decay]).all()
        or min(args.correction_weights)<0 or args.lambda_pa<=0 or min(args.lr,args.ct_backbone_lr)<=0 or args.weight_decay<0):
        parser.error('Invalid fold, objective weight, epoch, batch, learning rate, dropout or worker configuration')
    run(args)


if __name__=='__main__':main()
