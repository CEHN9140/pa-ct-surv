"""Nested CT-only auxiliary sphere/allShape supervision; not a teacher."""

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
from sklearn.metrics import r2_score,mean_absolute_error
from sksurv.metrics import concordance_index_censored
from threadpoolctl import threadpool_limits

from ct_shape_aux_model import CTWithShape,auxiliary_objective,shape_optimizer,select_auxiliary
from ct_shape_aux_probe import evaluate_selected_probes,write_aux_report
from ct_layer_train import reference_fold
from dataset import CT_Dataset,build_dataloader,cox_batch_sizes,cv_splits
from feature_increment import file_hash,extract_ct_features
from loss import cox_loss
from phenotype_followup import read_feature_cache
from residual_train import source_partitions
from risk_increment import ordered_predictions
from risk_calibration import paired_comparison
from utils import seed_everything,save_run_config,save_cv_results
from visualization.survival_plots import analyze_fold


@torch.inference_mode()
def predict_auxiliary(model,loader,device):
    model.eval();rows=[];decoded=[]
    for ct,event,time,ids in loader:
        risk,shape=model(ct.to(device,non_blocking=True))
        decoded.append((shape*model.shape_std+model.shape_mean).cpu().numpy())
        rows.extend(dict(case_id=str(case),**{'dfs.month':float(t),'dfs.status':int(e)},risk_score=float(r))
                    for case,t,e,r in zip(ids,time,event,risk.cpu()))
    return pd.DataFrame(rows),np.concatenate(decoded)


def fit_auxiliary(dataset,shape_by_ct,fit,stop,args,device,checkpoint,weight,fixed_epochs=None):
    if (fixed_epochs is None)==(stop is None):raise ValueError('Use inner STOP or fixed outer epochs')
    seed_everything(args.seed)
    model=CTWithShape(len(args.shape_columns),args.ct_pretrained_path,args.dropout).to(device)
    fit_ids=dataset.samples.iloc[fit].ct_id.tolist()
    model.set_shape_scaler(shape_by_ct.loc[fit_ids,args.shape_columns].to_numpy())
    if not all(p.requires_grad for p in model.parameters()):raise ValueError('All CT/Shape parameters must train')
    optimizer=shape_optimizer(model,args.lr,args.weight_decay,args.ct_backbone_lr)
    train=build_dataloader(dataset,indices=fit,batch_size=args.batch_size,shuffle=True,merge_tail=True,num_workers=args.num_workers)
    stopping=None if stop is None else build_dataloader(dataset,indices=stop,batch_size=args.batch_size,num_workers=args.num_workers)
    checkpoint=Path(checkpoint);checkpoint.mkdir(parents=True)
    epochs=args.num_epochs if fixed_epochs is None else fixed_epochs
    best=-np.inf;wait=0;selected=None;history=[]
    scaler=dict(columns=args.shape_columns,mean=model.shape_mean.cpu().tolist(),std=model.shape_std.cpu().tolist(),
                n_fit=len(fit),ddof=0,fit_ct_ids_sha256=hashlib.sha256('\n'.join(fit_ids).encode()).hexdigest())
    for epoch in range(1,epochs+1):
        model.train();optimizer.zero_grad();risks=[];times=[];events=[];totals=[];coxes=[];shapes=[]
        for ct,event,time,ids in train:
            risk,pred=model(ct.to(device,non_blocking=True))
            # Match the source loop's diagnostics/single-batch concatenation
            # order too; this controls float32 gradient accumulation on CUDA.
            risks.append(risk.detach().cpu());times.append(time.reshape(-1));events.append(event.reshape(-1))
            truth=torch.as_tensor(shape_by_ct.loc[list(ids),args.shape_columns].to_numpy(),dtype=pred.dtype,device=device)
            values=auxiliary_objective(model,torch.cat([risk.reshape(-1)]),pred,truth,
                torch.cat([time.to(device).reshape(-1)]),torch.cat([event.to(device).reshape(-1)]),weight)
            values['total'].backward();optimizer.step();optimizer.zero_grad()
            totals.append(float(values['total'].detach()));coxes.append(float(values['cox'].detach()));shapes.append(float(values['shape_mse'].detach()))
        ci=concordance_index_censored(torch.cat(events).numpy().astype(bool),torch.cat(times).numpy(),torch.cat(risks).numpy())[0]
        row=dict(epoch=epoch,train_loss=float(np.mean(totals)),train_cox=float(np.mean(coxes)),
                 train_shape_mse=float(np.mean(shapes)) if weight>0 else np.nan,train_cindex=float(ci))
        line=f'Epoch {epoch}/{epochs} | Train Loss: {row["train_loss"]:.4f} | Train C-index: {ci:.4f} | Train Cox: {row["train_cox"]:.4f}'
        if weight>0:line+=f' | Train Shape MSE: {row["train_shape_mse"]:.4f}'
        if stopping is not None:
            frame,predicted=predict_auxiliary(model,stopping,device)
            risk,time,event=[torch.tensor(frame[k].to_numpy(),device=device) for k in ['risk_score','dfs.month','dfs.status']]
            truth=torch.as_tensor(shape_by_ct.loc[frame.case_id,args.shape_columns].to_numpy(),dtype=model.shape_mean.dtype,device=device)
            pred=torch.as_tensor(predicted,dtype=truth.dtype,device=device)
            mse=float((((pred-truth)/model.shape_std)**2).mean())
            val_cox=float(cox_loss(risk,time,event));val_ci=float(concordance_index_censored(event.cpu().numpy().astype(bool),time.cpu().numpy(),risk.cpu().numpy())[0])
            row.update(val_loss=val_cox+weight*mse,val_cox=val_cox,val_shape_mse=mse if weight>0 else np.nan,val_cindex=val_ci)
            line+=f' | Val Loss: {row["val_loss"]:.4f} | Val C-index: {val_ci:.4f} | Val Cox: {val_cox:.4f}'
            if weight>0:line+=f' | Val Shape MSE: {mse:.4f}'
            if val_ci>best:
                best=val_ci;selected=epoch;wait=0
                torch.save({k:v.detach().cpu() for k,v in model.state_dict().items()},checkpoint/'best_model.pth')
            else:wait+=1
        history.append(row);print(line,flush=True)
        pd.DataFrame(history).to_csv(checkpoint/'training_history.csv',index=False)
        if stopping is not None and args.patience>0 and wait>=args.patience:
            print(f'Early stopping at epoch{epoch}',flush=True);break
    if stopping is None:
        selected=epochs
        torch.save({k:v.detach().cpu() for k,v in model.state_dict().items()},checkpoint/'best_model.pth')
    else:model.load_state_dict(torch.load(checkpoint/'best_model.pth',map_location=device,weights_only=True),strict=True)
    metadata=dict(weight=weight,selected_epoch=int(selected),epochs_completed=epoch,scaler=scaler,
        training_risk_set_sizes=cox_batch_sizes(len(fit),args.batch_size),best_STOP_cindex=None if stopping is None else best,
        initialization='PUBLIC MedicalNet + fresh original risk head; no downstream warm-start',
        selection='fixed epochs from inner selection' if stopping is None else 'STOP C-index only')
    (checkpoint/'checkpoint_metadata.json').write_text(json.dumps(metadata,indent=2))
    return model,metadata


def load_sources(args):
    source=Path(args.source_run);config=yaml.safe_load((source/'run_config.yaml').read_text())
    if not(source/'completed.json').is_file() or config['fusion_type']!='risk_increment_nested_cv':raise ValueError('Require completed strict CT baseline')
    for key in ['data_dir','ct_pretrained_path','batch_size','cox_batch_size','dropout','lr','ct_backbone_lr','weight_decay','seed','num_epochs','patience']:
        setattr(args,key,config[key])
    dataset=CT_Dataset(args.data_dir,roi_size=64);samples=dataset.samples
    if len(samples)!=721 or int(samples.event.sum())!=251 or not samples.ct_id.is_unique:raise ValueError('Require721 patients/251 events')
    cohort=Path(args.data_dir)/'all_label_roi64.csv'
    if file_hash(cohort)!=config['cohort_sha256'] or file_hash(args.ct_pretrained_path)!=config['public_pretrained_sha256']:
        raise ValueError('Source cohort/public initialization changed')
    feature=Path(args.feature_run);feature_config=yaml.safe_load((feature/'run_config.yaml').read_text())
    if Path(feature_config['source_run']).resolve()!=source.resolve():raise ValueError('Feature baseline source mismatch')
    if feature_config['source_config_sha256']!=file_hash(source/'run_config.yaml'):
        raise ValueError('A training configuration changed since validated feature extraction')
    shape_path=feature/'cache/shape_features.csv';native=pd.read_csv(shape_path,dtype={'case_id':str}).set_index('case_id')
    ids=(samples.pa_id+'|'+samples.ct_id).tolist()
    if set(native.index)!=set(ids) or not native.index.is_unique:raise ValueError('Shape patient matching differs')
    shapes=native[['diameter_mm','sphericity']].copy();shapes.insert(0,'log_volume_mm3',np.log(native.volume_mm3))
    if not np.isfinite(shapes).all().all():raise ValueError('Nonfinite Shape target')
    by_ct=pd.DataFrame(shapes.loc[ids].to_numpy(),index=samples.ct_id,columns=shapes.columns)
    provenance={str(p):file_hash(p) for p in [source/'run_config.yaml',source/'completed.json',source/'ct/metrics.csv',
        feature/'run_config.yaml',shape_path,cohort,Path(args.ct_pretrained_path)]}
    partitions={};baseline_features={};zero_records={};references={}
    for fold,(training,validation) in enumerate(cv_splits(samples)):
        role=source/f'fold_{fold}/inner_split_roles.csv';partitions[fold]=source_partitions(samples,training,validation,role)
        calibration=source/f'fold_{fold}/calibration.yaml';cal=yaml.safe_load(calibration.read_text())
        oof_path=source/f'fold_{fold}/inner_oof_predictions.csv';oof=pd.read_csv(oof_path,dtype={'case_id':str}).set_index('case_id')
        references[fold]=reference_fold(source,fold,samples,training,validation);zero_records[fold]=[]
        for inner,fit,stop,held in partitions[fold]:
            expected=(samples.iloc[held].pa_id+'|'+samples.iloc[held].ct_id).tolist()
            frame=oof[oof.inner_fold.eq(inner)]
            if set(frame.index)!=set(expected):raise ValueError('A held prediction membership mismatch')
            frame=frame.loc[expected]
            np.testing.assert_allclose(frame['dfs.month'],samples.iloc[held].time,rtol=1e-7,atol=1e-5)
            np.testing.assert_array_equal(frame['dfs.status'],samples.iloc[held].event)
            meta=next(x for x in cal['inner_models'] if x['modality']=='ct' and x['inner_fold']==inner)
            ci=concordance_index_censored(frame['dfs.status'].astype(bool),frame['dfs.month'],frame.raw_ct_risk)[0]
            zero_records[fold].append(dict(weight=0.,inner_fold=inner,held_cindex=float(ci),selected_epoch=meta['selected_epoch'],reuse='matched A control'))
        if int(np.median([r['selected_epoch'] for r in zero_records[fold]]))!=cal['selected_epochs']['ct']:
            raise ValueError('A epoch rule mismatch')
        baseline_features[fold]={}
        for stage in ['inner_0','inner_1','inner_2','outer']:
            expected=ids if stage=='outer' else [(samples.iloc[i].pa_id+'|'+samples.iloc[i].ct_id) for i in training]
            path=feature/f'cache/finetuned/fold_{fold}/{stage}.npz'
            metadata=json.loads(path.with_suffix('.json').read_text())
            if Path(metadata['checkpoint']).resolve()!=(Path(config['checkpoint_root'])/f'fold_{fold}'/stage/'ct/best_model.pth').resolve():
                raise ValueError('A feature checkpoint provenance mismatch')
            baseline_features[fold][stage]=read_feature_cache(path,expected,config['cohort_sha256'],False,provenance)
        for p in [role,calibration,oof_path,*[source/f'ct/fold_{fold}/{s}_predictions.csv' for s in ['train','val']]]:
            provenance[str(p)]=file_hash(p)
    args.cohort_sha256=config['cohort_sha256'];args.public_pretrained_sha256=config['public_pretrained_sha256']
    print('Audited721 cases,Shape labels,A risks/epochs and20 baseline feature caches; only PUBLIC initialization for new fits.',flush=True)
    return dataset,shapes,by_ct,partitions,zero_records,references,baseline_features,provenance


def model_features(model,dataset,indices,args,device,folder,checkpoint):
    loader=build_dataloader(dataset,indices=indices,batch_size=args.batch_size,num_workers=args.num_workers)
    ct_ids,features=extract_ct_features(model,loader,device)
    samples=dataset.samples.iloc[indices];np.testing.assert_array_equal(ct_ids,samples.ct_id.to_numpy())
    ids=(samples.pa_id+'|'+samples.ct_id).to_numpy()
    folder=Path(folder);folder.mkdir(parents=True)
    np.savez_compressed(folder/'features.npz',case_id=ids.astype(str),features=features)
    (folder/'features.json').write_text(json.dumps(dict(checkpoint=str(checkpoint),checkpoint_sha256=file_hash(checkpoint),
        patients=len(ids),feature_dim=512,cohort_sha256=args.cohort_sha256,computation='eval layer4 GAP before risk/aux heads; no L2; BN unchanged'),indent=2))
    return pd.DataFrame(features,index=ids)


def run(args):
    print(f'CT Shape auxiliary diagnostic: {args.target}; auditing sources...',flush=True)
    if np.__version__!='1.26.4':raise RuntimeError('Use isolated phenotype runtime')
    args.shape_columns=['sphericity'] if args.target=='sphericity' else ['log_volume_mm3','diameter_mm','sphericity']
    args.results_root=args.results_root or str(Path(__file__).parent/f'results/ct_shape_aux/{args.target}')
    args.checkpoint_root=args.checkpoint_root or str(Path(__file__).parent/f'checkpoints/ct_shape_aux/{args.target}')
    for p in [args.results_root,args.checkpoint_root]:
        if Path(p).exists():raise FileExistsError(f'Refusing to overwrite{p}')
    dataset,shapes,by_ct,partitions,zero,references,baseline_features,provenance=load_sources(args)
    device=torch.device(args.device)
    if device.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    torch.set_num_threads(4);root=Path(args.results_root);ckpt=Path(args.checkpoint_root)
    args.fusion_type='ct_shape_auxiliary_diagnostic';args.survival_loss='cox';args.roi_size=64
    save_run_config(args,'ct',args.target,extra=dict(source_hashes=provenance,teacher_student_unchanged=True,
        mse_reduction='mean patients and targets; C per-target coefficient=lambda/3',bn='fixed running stats; affine trainable'))
    print(f'Device={device} | batch={args.batch_size}+tail | headLR={args.lr}/backboneLR={args.ct_backbone_lr} | lambdas={args.lambda_grid}',flush=True)
    results=[];compared=[];selected_features={};selections=[];head_metrics=[]
    for fold,(training,validation) in enumerate(cv_splits(dataset.samples)):
        folder=root/f'fold_{fold}';folder.mkdir();records=[dict(r) for r in zero[fold]]
        for weight in [v for v in args.lambda_grid if v>0]:
            for inner,fit,stop,held in partitions[fold]:
                checkpoint=ckpt/f'fold_{fold}/lambda_{weight:g}/inner_{inner}'
                print(f'\n{args.target} | Outer{fold} | lambda{weight:g} | Inner{inner} | fit={len(fit)},STOP={len(stop)},HELD={len(held)}',flush=True)
                model,metadata=fit_auxiliary(dataset,by_ct,fit,stop,args,device,checkpoint,weight)
                held_frame=ordered_predictions(model,dataset,held,'ct',args,device)
                held_frame.to_csv(checkpoint/'held_predictions.csv',index=False)
                ci=concordance_index_censored(held_frame['dfs.status'].astype(bool),held_frame['dfs.month'],held_frame.risk_score)[0]
                records.append(dict(metadata,inner_fold=inner,held_cindex=float(ci)))
                del model
                if device.type=='cuda':torch.cuda.empty_cache()
        selected=select_auxiliary(records);selected['inner_models']=records
        (folder/'selection.json').write_text(json.dumps(selected,indent=2));selections.append(dict(fold=fold,**selected))
        print(f'{args.target} Outer{fold}: lambda={selected["weight"]:g},fixed outer epochs={selected["epochs"]} (outer labels unused)',flush=True)
        weight=selected['weight'];selected_features[fold]={}
        # Extract only selected INNER checkpoints, then freshly initialize the
        # outer network. No checkpoint from a trained model enters fit_auxiliary.
        for inner,fit,stop,held in partitions[fold]:
            stage=f'inner_{inner}'
            if weight==0:selected_features[fold][stage]=baseline_features[fold][stage]
            else:
                path=ckpt/f'fold_{fold}/lambda_{weight:g}/{stage}/best_model.pth'
                model=CTWithShape(len(args.shape_columns),dropout=args.dropout).to(device)
                model.load_state_dict(torch.load(path,map_location=device,weights_only=True),strict=True)
                selected_features[fold][stage]=model_features(model,dataset,training,args,device,root/f'cache/fold_{fold}/{stage}',path)
                del model
                if device.type=='cuda':torch.cuda.empty_cache()
        if weight==0:
            frames={k:v.copy() for k,v in references[fold].items()};selected_features[fold]['outer']=baseline_features[fold]['outer']
            print('Selected lambda0: reusing A comparison outputs only; auxiliary head metrics=N/A.',flush=True)
        else:
            path=ckpt/f'fold_{fold}/outer'
            model,_=fit_auxiliary(dataset,by_ct,training,None,args,device,path,weight,fixed_epochs=selected['epochs'])
            frames={split:ordered_predictions(model,dataset,indices,'ct',args,device) for split,indices in [('train',training),('val',validation)]}
            selected_features[fold]['outer']=model_features(model,dataset,np.arange(len(dataset)),args,device,root/f'cache/fold_{fold}/outer',path/'best_model.pth')
            loader=build_dataloader(dataset,indices=validation,batch_size=args.batch_size,num_workers=args.num_workers)
            decoded_frame,decoded=predict_auxiliary(model,loader,device)
            np.testing.assert_array_equal(decoded_frame.case_id,dataset.samples.iloc[validation].ct_id)
            truth=by_ct.loc[decoded_frame.case_id,args.shape_columns].to_numpy()
            for i,column in enumerate(args.shape_columns):
                head_metrics.append(dict(fold=fold,target=column,r2=float(r2_score(truth[:,i],decoded[:,i])),mae=float(mean_absolute_error(truth[:,i],decoded[:,i]))))
            pd.DataFrame(dict(case_id=decoded_frame.case_id,**{'true_'+c:truth[:,i] for i,c in enumerate(args.shape_columns)},
                             **{'pred_'+c:decoded[:,i] for i,c in enumerate(args.shape_columns)})).to_csv(folder/'shape_head_predictions.csv',index=False)
            del model
            if device.type=='cuda':torch.cuda.empty_cache()
        frames['train'].to_csv(folder/'train_predictions.csv',index=False)
        metrics=analyze_fold(frames['train'],frames['val'],folder)
        for split,frame in frames.items():metrics[split+'_loss']=float(cox_loss(*[torch.tensor(frame[c].to_numpy()) for c in ['risk_score','dfs.month','dfs.status']]))
        row=dict(fold=fold,selected_lambda=weight,selected_epoch=selected['epochs'],**metrics);results.append(row)
        base=references[fold]['val'].set_index('case_id').loc[frames['val'].case_id];pair=frames['val'].copy();pair['baseline_A']=base.risk_score.to_numpy();compared.append(pair)
        (folder/'completed.json').write_text(json.dumps(row,indent=2));pd.DataFrame(results).to_csv(root/'partial_metrics.csv',index=False)
        print(f'{args.target} final Outer{fold} | Train={row["train_cindex"]:.4f} | Val={row["val_cindex"]:.4f}',flush=True)
    save_cv_results(results,root)
    comparison=paired_comparison(compared,seed=args.seed,reference_columns={'A':'baseline_A'})
    comparison['positive_folds']=int(sum(r['val_cindex']>concordance_index_censored(references[i]['val']['dfs.status'].astype(bool),references[i]['val']['dfs.month'],references[i]['val'].risk_score)[0] for i,r in enumerate(results)))
    (root/'paired_comparison_A.json').write_text(json.dumps(comparison,indent=2))
    pd.DataFrame(head_metrics,columns=['fold','target','r2','mae']).to_csv(root/'shape_head_metrics.csv',index=False)
    probe_rows=evaluate_selected_probes(dataset,shapes,partitions,selected_features,baseline_features,args,root)
    write_aux_report(root,results,comparison,probe_rows,selections,args)
    for path,sha in provenance.items():
        if file_hash(path)!=sha:raise ValueError(f'Source changed during auxiliary run:{path}')
    (root/'completed.json').write_text(json.dumps(dict(target=args.target,patients=721,events=251,outer_folds=list(range(5)),
        selected_lambdas=[r['weight'] for r in selections],source_hashes_unchanged=True,
        teacher_student_training_started=False,public_only_new_initialization=True),indent=2))
    print(f'CT Shape auxiliary {args.target} complete; no teacher/student training started.',flush=True)


if __name__=='__main__':
    experiment=Path(__file__).resolve().parent
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--target',choices=['sphericity','allshape'],default='sphericity')
    parser.add_argument('--source_run',default=str(experiment/'results/nested_risk_increment'))
    parser.add_argument('--feature_run',default=str(experiment/'results/feature_increment'))
    parser.add_argument('--results_root',default=None)
    parser.add_argument('--checkpoint_root',default=None)
    parser.add_argument('--lambda_grid',nargs='+',type=float,default=[0.,.05,.2])
    parser.add_argument('--num_workers',type=int,default=8)
    parser.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    args=parser.parse_args()
    if sorted(args.lambda_grid)!=[0.,.05,.2] or args.num_workers<0:parser.error('Use predeclared lambdas0/.05/.2 and nonnegative workers')
    with threadpool_limits(limits=2):run(args)
