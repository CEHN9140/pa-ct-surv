"""Nested CT-only layer3/layer34 experiments against the completed layer4 baseline."""

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

from ct_layer_model import CTLayerModel
from dataset import CT_Dataset,build_dataloader,cv_splits,cox_batch_sizes
from loss import cox_loss
from residual_train import source_partitions
from risk_increment import train_risk_model,ordered_predictions
from risk_calibration import risk_scaler,standardize_risk,paired_comparison
from utils import build_ct_optimizer,seed_everything,save_run_config,save_cv_results
from visualization.survival_plots import analyze_fold


def fit_layer(dataset,fit,stop,args,device,checkpoint,fixed_epochs=None):
    seed_everything(args.seed)
    model=CTLayerModel(args.feature_source,pretrained_path=args.ct_pretrained_path,dropout=args.dropout).to(device)
    optimizer=build_ct_optimizer(model,args.lr,args.weight_decay,args.ct_backbone_lr)
    train=build_dataloader(dataset,indices=fit,batch_size=args.batch_size,shuffle=True,
                           merge_tail=True,num_workers=args.num_workers)
    stopping=None if stop is None else build_dataloader(dataset,indices=stop,batch_size=args.batch_size,
                                                         num_workers=args.num_workers)
    model,metadata=train_risk_model(model,optimizer,train,stopping,args,device,checkpoint,
                                    forwards_per_update=1,fixed_epochs=fixed_epochs)
    metadata.update(feature_source=args.feature_source,feature_dim=model.ct.feature_dim,
        backbone_initialization='public MedicalNet only; no project downstream checkpoint',
        training_risk_set_sizes=cox_batch_sizes(len(fit),args.batch_size),
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad))
    (Path(checkpoint)/'checkpoint_metadata.yaml').write_text(yaml.safe_dump(metadata))
    return model,metadata


def reference_fold(source,fold,samples,training,validation):
    frames={split:pd.read_csv(source/f'ct/fold_{fold}/{split}_predictions.csv',dtype={'case_id':str})
            for split in ['train','val']}
    for split,indices in [('train',training),('val',validation)]:
        truth=samples.iloc[indices].copy();truth.index=truth.pa_id+'|'+truth.ct_id
        frame=frames[split]
        if not frame.case_id.is_unique or set(frame.case_id)!=set(truth.index):
            raise ValueError('Layer4 reference patients differ from current saved fold')
        truth=truth.loc[frame.case_id]
        np.testing.assert_allclose(frame['dfs.month'],truth.time,rtol=1e-7,atol=1e-5)
        np.testing.assert_array_equal(frame['dfs.status'],truth.event)
    return frames


def run(args):
    source=Path(args.source_run)
    config=yaml.safe_load((source/'run_config.yaml').read_text())
    if not(source/'completed.json').is_file() or config['fusion_type']!='risk_increment_nested_cv':
        raise ValueError('Require completed nested CT baseline and role assignments')
    audit=Path(args.input_audit)
    completed=json.loads((audit/'completed.json').read_text())
    if not completed['complete'] or not completed['passed'] or completed['patients']!=721:
        raise ValueError('Require completed721-patient CT geometry/reconstruction audit')
    args.data_dir=config['data_dir'];args.ct_pretrained_path=config['ct_pretrained_path']
    for key in ['batch_size','dropout','lr','ct_backbone_lr','weight_decay','seed','num_epochs','patience']:
        setattr(args,key,config[key])
    args.cohort_sha256=hashlib.sha256((Path(args.data_dir)/'all_label_roi64.csv').read_bytes()).hexdigest()
    args.public_pretrained_sha256=hashlib.sha256(Path(args.ct_pretrained_path).read_bytes()).hexdigest()
    if args.cohort_sha256!=config['cohort_sha256'] or args.public_pretrained_sha256!=config['public_pretrained_sha256']:
        raise ValueError('Baseline cohort/public encoder changed')
    audit_config=json.loads((audit/'run_config.json').read_text())
    if audit_config['sha256']['cohort']!=args.cohort_sha256:raise ValueError('Audit cohort differs from CT experiment')
    args.source_config_sha256=hashlib.sha256((source/'run_config.yaml').read_bytes()).hexdigest()
    args.reference_layer4_results=str(source/'ct')
    args.reference_layer4_metrics_sha256=hashlib.sha256((source/'ct/metrics.csv').read_bytes()).hexdigest()
    args.input_provenance='Physician-provided NIfTI; unknown upstream transformations/acquisition retained as fixed input per user.'
    args.fusion_type='ct_layer_representation_diagnostic';args.survival_loss='cox'
    args.feature_dim=256 if args.feature_source=='layer3' else 768
    args.head_hidden_dim=256;args.ct_pooled_l2=False;args.ct_bn_stats='fixed_public_statistics';args.ct_augmentation=False
    for path in [Path(args.results_root),Path(args.checkpoint_root)]:
        if path.exists():raise FileExistsError(f'Refusing to overwrite {path}')
    dataset=CT_Dataset(args.data_dir,roi_size=64)
    if len(dataset)!=721 or int(dataset.samples.event.sum())!=251:raise ValueError('Require721 patients/251 events')
    device=torch.device(args.device)
    if device.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA requested but unavailable')
    torch.set_num_threads(4)
    save_run_config(args,'ct',args.feature_source)
    print(f'CT {args.feature_source} GAP | dimension={args.feature_dim} |721 patients/251 events | Device={device}',flush=True)
    print(f'Public-only initialization; fixed BN; no feature L2; batch={args.batch_size}; '
          f'headLR={args.lr:g}/backboneLR={args.ct_backbone_lr:g}; dropout={args.dropout}; '
          f'max epochs={args.num_epochs}/inner patience={args.patience}.',flush=True)
    rows=[];comparisons=[]
    for fold,(training,validation) in enumerate(cv_splits(dataset.samples)):
        if fold not in args.outer_folds:continue
        reference=reference_fold(source,fold,dataset.samples,training,validation)
        partitions=source_partitions(dataset.samples,training,validation,source/f'fold_{fold}/inner_split_roles.csv')
        destination=Path(args.results_root)/f'fold_{fold}';destination.mkdir()
        truth=dataset.samples.iloc[training]
        oof=pd.DataFrame(dict(case_id=(truth.pa_id+'|'+truth.ct_id).to_numpy(),
                              **{'dfs.month':truth.time.to_numpy(),'dfs.status':truth.event.to_numpy()})).set_index('case_id')
        oof['ct_risk']=np.nan;oof['raw_ct_risk']=np.nan;oof['inner_fold']=np.nan
        inner=[]
        for number,fit,stop,held in partitions:
            print(f'\n{args.feature_source} | Outer{fold} | Inner{number} | fit={len(fit)}, stop={len(stop)}, held={len(held)}',flush=True)
            checkpoint=Path(args.checkpoint_root)/f'fold_{fold}/inner_{number}'
            model,metadata=fit_layer(dataset,fit,stop,args,device,checkpoint)
            fitting=ordered_predictions(model,dataset,fit,'ct',args,device)
            held_predictions=ordered_predictions(model,dataset,held,'ct',args,device)
            scaler=risk_scaler(fitting.risk_score)
            identities=held_predictions.case_id
            oof.loc[identities,'ct_risk']=standardize_risk(held_predictions.risk_score,scaler)
            oof.loc[identities,'raw_ct_risk']=held_predictions.risk_score.to_numpy()
            oof.loc[identities,'inner_fold']=number
            score=concordance_index_censored(held_predictions['dfs.status'].astype(bool),held_predictions['dfs.month'],held_predictions.risk_score)[0]
            inner.append(dict(inner_fold=number,held_cindex=float(score),scaler=scaler,**metadata))
            del model
            if device.type=='cuda':torch.cuda.empty_cache()
        if oof.isna().any().any():raise ValueError('Incomplete honest CT OOF coverage')
        epochs=int(np.median([x['selected_epoch'] for x in inner]))
        selection=dict(feature_source=args.feature_source,selected_epoch=epochs,inner_models=inner,
            policy='floor median three inner-stop best epochs; fresh public initialization on all outer training; '
                   'outer validation is never used for checkpoint, layer or hyperparameter selection')
        (destination/'selection.json').write_text(json.dumps(selection,indent=2))
        oof.reset_index().to_csv(destination/'inner_oof_predictions.csv',index=False)
        print(f'{args.feature_source} Outer{fold}: fixed outer refit epochs={epochs}',flush=True)
        model,_=fit_layer(dataset,training,None,args,device,Path(args.checkpoint_root)/f'fold_{fold}/outer',fixed_epochs=epochs)
        frames={split:ordered_predictions(model,dataset,indices,'ct',args,device)
                for split,indices in [('train',training),('val',validation)]}
        frames['train'].to_csv(destination/'train_predictions.csv',index=False)
        metrics=analyze_fold(frames['train'],frames['val'],destination)
        for split,frame in frames.items():
            metrics[split+'_loss']=float(cox_loss(*[torch.tensor(frame[key].to_numpy()) for key in ['risk_score','dfs.month','dfs.status']]))
        baseline=reference['val'].set_index('case_id').loc[frames['val'].case_id]
        combined=frames['val'].copy();combined['baseline_layer4']=baseline.risk_score.to_numpy()
        comparisons.append(combined);rows.append(dict(fold=fold,**metrics))
        (destination/'completed.json').write_text(json.dumps(dict(fold=fold,selected_epoch=epochs,metrics=metrics),indent=2))
        pd.DataFrame(rows).to_csv(Path(args.results_root)/'partial_metrics.csv',index=False)
        print(f'{args.feature_source} final fold{fold} | Train={metrics["train_cindex"]:.4f} | '
              f'Val={metrics["val_cindex"]:.4f}',flush=True)
        del model
        if device.type=='cuda':torch.cuda.empty_cache()
    full=sorted(args.outer_folds)==list(range(5))
    if full:
        save_cv_results(rows,Path(args.results_root))
        comparison=paired_comparison(comparisons,seed=args.seed,reference_columns={'layer4':'baseline_layer4'})
        (Path(args.results_root)/'paired_comparison_layer4.json').write_text(json.dumps(comparison,indent=2))
    if hashlib.sha256((source/'ct/metrics.csv').read_bytes()).hexdigest()!=args.reference_layer4_metrics_sha256:
        raise ValueError('Layer4 reference changed during experiment')
    (Path(args.results_root)/'completed.json').write_text(json.dumps(dict(feature_source=args.feature_source,
        folds=args.outer_folds,complete_fivefold=full,independent_fits=4*len(args.outer_folds),
        reference_layer4=str(source/'ct'),public_initialization_only=True,source_results_unchanged=True),indent=2))
    print(f'CT {args.feature_source} comparison complete.',flush=True)


if __name__=='__main__':
    experiment=Path(__file__).resolve().parent
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--feature_source',choices=['layer3','layer34'],default='layer3')
    parser.add_argument('--source_run',default=str(experiment/'results/nested_risk_increment'))
    parser.add_argument('--input_audit',default=str(experiment/'results/ct_input_audit'))
    parser.add_argument('--results_root',default=None)
    parser.add_argument('--checkpoint_root',default=None)
    parser.add_argument('--outer_folds',type=int,nargs='+',default=list(range(5)))
    parser.add_argument('--num_workers',type=int,default=8)
    parser.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    args=parser.parse_args()
    if args.num_workers<0 or not args.outer_folds or len(set(args.outer_folds))!=len(args.outer_folds) or not set(args.outer_folds).issubset(range(5)):
        parser.error('Invalid worker count or outer folds')
    args.results_root=args.results_root or str(experiment/'results/ct_layer_comparison'/args.feature_source)
    args.checkpoint_root=args.checkpoint_root or str(experiment/'checkpoints/ct_layer_comparison'/args.feature_source)
    run(args)
