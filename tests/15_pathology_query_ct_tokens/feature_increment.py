"""Fold-restricted CT feature increment; conditional Shape follow-up, not a teacher."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import joblib
import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
import yaml
from threadpoolctl import threadpool_limits

from dataset import CT_Dataset, Path_Dataset, build_dataloader
from increment_cox import IncrementCox, select_increment, stable_increment
from model.build import CT_Model
from risk_calibration import inner_splits, paired_comparison, risk_scaler, standardize_risk
from risk_increment import fit_modality, ordered_predictions
from shape_increment import mask_shape, volume_probe
from utils import save_cv_results, save_run_config, seed_everything
from visualization.survival_plots import analyze_fold


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8*1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


@torch.inference_mode()
def extract_ct_features(model, loader, device):
    model.eval()
    before = {k: v.clone() for k, v in model.named_buffers()}
    ids, features = [], []
    for batch in loader:
        value = model.ct.extract_features(batch[0].to(device))
        if value.ndim != 2 or value.size(1) != 512 or not torch.isfinite(value).all():
            raise ValueError('Expected finite pre-risk-head CT features [B,512]')
        features.append(value.cpu().numpy())
        ids.extend(str(x.item()) if torch.is_tensor(x) else str(x) for x in batch[3])
    for k, v in model.named_buffers():
        torch.testing.assert_close(v, before[k], rtol=0, atol=0)
    return ids, np.concatenate(features)


def feature_table(dataset, indices, checkpoint, public, cache, args, device):
    cache = Path(cache)
    checksum = file_hash(checkpoint)
    expected = dataset.samples.iloc[indices]
    case_ids = (expected.pa_id+'|'+expected.ct_id).to_numpy()
    if cache.exists():
        metadata = json.loads(cache.with_suffix('.json').read_text())
        if metadata['checkpoint_sha256'] != checksum or metadata['cohort_sha256'] != args.cohort_sha256:
            raise ValueError('Feature-cache provenance mismatch')
        with np.load(cache, allow_pickle=False) as stored:
            np.testing.assert_array_equal(stored['case_id'], case_ids)
            return pd.DataFrame(stored['features'], index=case_ids)
    seed_everything(args.seed)
    model = CT_Model(pretrained_path=str(checkpoint) if public else None, dropout=args.dropout).to(device)
    if not public:
        model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True), strict=True)
    loader = build_dataloader(dataset, indices=indices, batch_size=args.batch_size, num_workers=args.num_workers)
    ids, features = extract_ct_features(model, loader, device)
    np.testing.assert_array_equal(ids, expected.ct_id.to_numpy())
    assert file_hash(checkpoint) == checksum
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, case_id=case_ids.astype(str), features=features)
    cache.with_suffix('.json').write_text(json.dumps(dict(checkpoint=str(checkpoint), checkpoint_sha256=checksum,
        cohort_sha256=args.cohort_sha256, patients=len(case_ids), feature_dim=512,
        source='public only' if public else 'fold-restricted DFS encoder',
        computation='layer4 global average; before risk head; eval; no L2; BN unchanged'), indent=2))
    del model
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    print(f'Cached {len(case_ids)} CT vectors: {cache}', flush=True)
    return pd.DataFrame(features, index=case_ids)


def load_bundle(fold, source, samples):
    oof = pd.read_csv(source/f'fold_{fold}/inner_oof_predictions.csv', dtype={'case_id':str}).set_index('case_id')
    frames = {split: pd.read_csv(source/f'late_fusion/fold_{fold}/{split}_predictions.csv',
              dtype={'case_id':str}).set_index('case_id') for split in ('train','val')}
    ids = samples.pa_id+'|'+samples.ct_id
    expected_train, expected_val = set(ids[samples.split.ne(fold)]), set(ids[samples.split.eq(fold)])
    assert set(oof.index) == set(frames['train'].index) == expected_train
    assert set(frames['val'].index) == expected_val and not expected_train & expected_val
    truth = samples.assign(case_id=ids).set_index('case_id')
    for frame in [oof, *frames.values()]:
        np.testing.assert_allclose(frame['dfs.month'], truth.loc[frame.index,'time'], rtol=1e-7, atol=1e-5)
        np.testing.assert_array_equal(frame['dfs.status'], truth.loc[frame.index,'event'])
    frames['train'] = frames['train'].loc[oof.index]
    return dict(oof=oof, **frames)


def calibration_contexts(fold, bundle, dataset, args, device):
    samples = dataset.samples
    index = {f'{row.pa_id}|{row.ct_id}':i for i,row in samples.iterrows()}
    contexts = []
    for inner in sorted(bundle['oof'].inner_fold.astype(int).unique()):
        held = bundle['oof'].index[bundle['oof'].inner_fold.eq(inner)].tolist()
        training = bundle['oof'].index[~bundle['oof'].inner_fold.eq(inner)].tolist()
        cache = Path(args.results_root)/f'cache/fold_{fold}/inner_{inner}'
        cache.mkdir(parents=True, exist_ok=True)
        prediction_file = cache/'pathology_train_oof.csv'
        if prediction_file.exists():
            predictions = pd.read_csv(prediction_file, dtype={'case_id':str}).set_index('case_id').loc[training]
        else:
            predictions = pd.DataFrame(index=training, columns=['pathology_risk'], dtype=float)
            roles, details = [], []
            for sub, (fit, stop, oof) in enumerate(inner_splits(samples, [index[x] for x in training],
                                                             2, args.stop_fraction, args.seed)):
                print(f'Outer {fold} | calibration fold {inner} | PA sub-OOF {sub+1}/2 | '
                      f'fit={len(fit)}, stop={len(stop)}, OOF={len(oof)}', flush=True)
                checkpoint = Path(args.checkpoint_root)/f'fold_{fold}/inner_{inner}/sub_{sub}/pathology'
                model, metadata = fit_modality('pathology', dataset, fit, stop, args, device, checkpoint)
                fitting = ordered_predictions(model, dataset, fit, 'pathology', args, device)
                scored = ordered_predictions(model, dataset, oof, 'pathology', args, device)
                scaler = risk_scaler(fitting.risk_score)
                predictions.loc[scored.case_id,'pathology_risk'] = standardize_risk(scored.risk_score, scaler)
                details.append(dict(subfold=sub, scaler=scaler, **metadata))
                for role, indices in [('fit',fit),('stop',stop),('oof',oof)]:
                    for i in indices:
                        identity=f'{samples.iloc[i].pa_id}|{samples.iloc[i].ct_id}'
                        assert identity not in held
                        roles.append(dict(subfold=sub, role=role, case_id=identity))
                del model
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
            assert not predictions.isna().any().any()
            predictions.rename_axis('case_id').to_csv(prediction_file)
            pd.DataFrame(roles).to_csv(cache/'nested_pa_roles.csv', index=False)
            (cache/'pa_provenance.json').write_text(json.dumps(details, indent=2))
        assert not set(training)&set(held)
        contexts.append(dict(inner_fold=int(inner), train_ids=training, val_ids=held,
            train_pa=predictions.pathology_risk.to_numpy(), val_pa=bundle['oof'].loc[held,'pathology_risk'].to_numpy(),
            train_time=bundle['oof'].loc[training,'dfs.month'].to_numpy(), train_event=bundle['oof'].loc[training,'dfs.status'].to_numpy(),
            val_time=bundle['oof'].loc[held,'dfs.month'].to_numpy(), val_event=bundle['oof'].loc[held,'dfs.status'].to_numpy()))
    return contexts


def source_contexts(source_name, fold, contexts, public_features, dataset, source_checkpoints, args, device):
    result = []
    for context in contexts:
        if source_name == 'public':
            features = public_features
        else:
            all_ids = set(context['train_ids']) | set(context['val_ids'])
            indices = [i for i,row in dataset.samples.iterrows() if f'{row.pa_id}|{row.ct_id}' in all_ids]
            features = feature_table(dataset, indices, source_checkpoints/f'fold_{fold}/inner_{context["inner_fold"]}/ct/best_model.pth',
                False, Path(args.results_root)/f'cache/finetuned/fold_{fold}/inner_{context["inner_fold"]}.npz', args, device)
        result.append(dict(context, train_ct=features.loc[context['train_ids']].to_numpy(),
                           val_ct=features.loc[context['val_ids']].to_numpy()))
    return result


def final_features(source_name, fold, public_features, dataset, source_checkpoints, args, device):
    if source_name == 'public':
        return public_features
    return feature_table(dataset, np.arange(len(dataset)), source_checkpoints/f'fold_{fold}/outer/ct/best_model.pth',
        False, Path(args.results_root)/f'cache/finetuned/fold_{fold}/outer.npz', args, device)


def export_head(name, fold, head, bundle, train_ct, val_ct, selection, args):
    destination = Path(args.results_root)/name/f'fold_{fold}'
    destination.mkdir(parents=True, exist_ok=True)
    joblib.dump(head, destination/'head.joblib')
    (destination/'head_parameters.json').write_text(json.dumps(dict(selection=selection,
        coefficients=head.model.coef_.tolist(), pathology_scaler_fit_n=int(head.pa_scaler.n_samples_seen_),
        ct_scaler_fit_n=None if head.ct_pipeline is None else int(head.ct_pipeline.named_steps['scale'].n_samples_seen_),
        final_training_source='outer-training OOF pathology risk; outer-training CT representation only'), indent=2))
    frames = {}
    for split, ct in [('train',train_ct),('val',val_ct)]:
        frame = bundle[split][['dfs.month','dfs.status']].copy()
        frame['risk_score'] = head.predict(bundle[split].pathology_risk.to_numpy(), ct)
        frames[split] = frame.reset_index()
    frames['train'].to_csv(destination/'train_predictions.csv', index=False)
    print(f'Outer {fold} | final {name}', flush=True)
    metrics = analyze_fold(frames['train'], frames['val'], destination)
    print(f'{name} | Train C-index: {metrics["train_cindex"]:.4f} | Val C-index: {metrics["val_cindex"]:.4f}', flush=True)
    return dict(fold=fold, **metrics), frames['val']


def comparison_for(name, predictions, metrics, args):
    combined=[]
    for model, m0, m1 in zip(predictions[name], predictions['M0'], predictions['M1']):
        frame=model.copy()
        for key,other in [('M0',m0),('M1',m1)]:
            other=other.set_index('case_id').loc[frame.case_id]
            frame[key]=other.risk_score.to_numpy()
        combined.append(frame)
    result=paired_comparison(combined, seed=args.seed, reference_columns={'M0':'M0','M1':'M1'})
    positive={key:sum(a['val_cindex']>b['val_cindex'] for a,b in zip(metrics[name],metrics[key])) for key in ('M0','M1')}
    result.update(positive_folds=positive, stable_increment=stable_increment(result,positive,len(combined)))
    (Path(args.results_root)/name/'paired_comparison.json').write_text(json.dumps(result,indent=2))
    return result


def shape_stage(bundles, contexts, public_features, dataset, source_checkpoints, predictions, metrics, args, device):
    records=[]
    for i,row in dataset.samples.iterrows():
        path=Path(args.mask_dir)/(row.ct_id+'.nii.gz')
        shape=mask_shape(sitk.ReadImage(str(path)))
        records.append(dict(case_id=f'{row.pa_id}|{row.ct_id}', **shape))
        if (i+1)%50==0:
            print(f'Shape extraction: {i+1}/{len(dataset)}',flush=True)
    shapes=pd.DataFrame(records).set_index('case_id')
    shapes.to_csv(Path(args.results_root)/'cache/shape_features.csv')
    shape_x=np.column_stack([np.log(shapes.volume_mm3),shapes.diameter_mm,shapes.sphericity])
    shapes_design=pd.DataFrame(shape_x,index=shapes.index)
    probes=[]
    for fold in args.outer_folds:
        bundle=bundles[fold];oof=bundle['oof'];ct_context=[]
        for context in contexts[fold]:
            ct_context.append(dict(context,train_ct=shapes_design.loc[context['train_ids']].to_numpy(),
                                   val_ct=shapes_design.loc[context['val_ids']].to_numpy()))
        for name,columns in [('M3_volume',[0]),('M4_shape',[0,1,2])]:
            if name=='M3_volume':
                choice=dict(dim=None,ridge=0.,selection='unpenalized predeclared log-volume')
            else:
                choice=select_increment(ct_context,dimensions=[None],penalties=args.ridge_grid)
            train_x=shapes_design.loc[oof.index,columns].to_numpy()
            head=IncrementCox(choice['dim'],choice['ridge']).fit(oof.pathology_risk,train_x,oof['dfs.month'],oof['dfs.status'])
            row,pred=export_head(name,fold,head,bundle,shapes_design.loc[bundle['train'].index,columns].to_numpy(),
                shapes_design.loc[bundle['val'].index,columns].to_numpy(),choice,args)
            metrics.setdefault(name,[]).append(row);predictions.setdefault(name,[]).append(pred)
        for source_name in args.ct_sources:
            context=source_contexts(source_name,fold,contexts[fold],public_features,dataset,source_checkpoints,args,device)
            for item in context:
                item['train_volume']=np.log(shapes.loc[item['train_ids'],'volume_mm3'].to_numpy())
                item['val_volume']=np.log(shapes.loc[item['val_ids'],'volume_mm3'].to_numpy())
            features=final_features(source_name,fold,public_features,dataset,source_checkpoints,args,device)
            probe=volume_probe(context,features.loc[oof.index].to_numpy(),np.log(shapes.loc[oof.index,'volume_mm3']),
                features.loc[bundle['val'].index].to_numpy(),np.log(shapes.loc[bundle['val'].index,'volume_mm3']),
                dimensions=[None],penalties=args.ridge_grid)
            target=Path(args.results_root)/f'volume_probe_{source_name}/fold_{fold}'
            target.mkdir(parents=True)
            joblib.dump(dict(transform=probe.pop('transform'),model=probe.pop('model')),target/'probe.joblib')
            pd.DataFrame(dict(case_id=bundle['val'].index,predicted_log_volume=probe.pop('prediction'),
                             volume_mm3=shapes.loc[bundle['val'].index,'volume_mm3'])).to_csv(target/'volume_predictions.csv',index=False)
            (target/'metrics.json').write_text(json.dumps(probe,indent=2))
            probes.append(dict(source=source_name,fold=fold,**{k:v for k,v in probe.items() if k not in ('selected','candidates')}))
            print(f'Volume probe {source_name} fold{fold}: R2(log volume)={probe["r2_log_volume"]:.4f}',flush=True)
    pd.DataFrame(probes).to_csv(Path(args.results_root)/'volume_probe_metrics.csv',index=False)


def run(args):
    source=Path(args.source_run);config=yaml.safe_load((source/'run_config.yaml').read_text())
    if not (source/'completed.json').is_file() or config['fusion_type']!='risk_increment_nested_cv':
        raise ValueError('Require the completed nested-risk source, not historical outer-selected baselines')
    args.data_dir=config['data_dir'];args.cohort_sha256=file_hash(Path(args.data_dir)/'all_label_roi64.csv')
    if args.cohort_sha256!=config['cohort_sha256']:
        raise ValueError('Source cohort has changed')
    args.ct_pretrained_path=config['ct_pretrained_path']
    if file_hash(args.ct_pretrained_path)!=config['public_pretrained_sha256']:
        raise ValueError('Public CT checkpoint changed')
    args.fusion_type='ct_feature_increment_diagnostic'
    for key in ('lr','ct_backbone_lr','weight_decay','dropout','cox_batch_size','batch_size','stop_fraction','seed'):
        setattr(args,key,config[key])
    args.source_config_sha256=file_hash(source/'run_config.yaml')
    for path in (Path(args.results_root),Path(args.checkpoint_root)):
        if path.exists():raise FileExistsError(f'Refusing to overwrite existing diagnostic: {path}')
    dataset=CT_Dataset(args.data_dir,roi_size=64);path_dataset=Path_Dataset(args.data_dir,roi_size=64,feature_dim=1280)
    if len(dataset)!=721:raise ValueError('Expected the original721-patient cohort')
    pd.testing.assert_frame_equal(dataset.samples[['pa_id','ct_id','event','time','split']],
                                  path_dataset.samples[['pa_id','ct_id','event','time','split']])
    for ct_id in dataset.samples.ct_id:
        if not(Path(args.mask_dir)/(ct_id+'.nii.gz')).is_file():raise FileNotFoundError(f'Missing original mask for {ct_id}')
    device=torch.device(args.device)
    if device.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA requested but unavailable')
    torch.set_num_threads(4);sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(4);seed_everything(args.seed)
    save_run_config(args,'diagnostic','feature_increment')
    stability=[]
    for fold in args.outer_folds:
        calibration=yaml.safe_load((source/f'fold_{fold}/calibration.yaml').read_text())
        stability.append(dict(fold=fold,outer_pathology_epochs=calibration['selected_epochs']['pathology'],
            inner_pathology_best_epochs=[x['selected_epoch'] for x in calibration['inner_models'] if x['modality']=='pathology']))
    (Path(args.results_root)/'source_pathology_stability.json').write_text(json.dumps(dict(folds=stability,
        policy='Reuse the same source pathology OOF/final predictions for all candidates; never reselect outer pathology epochs.'),indent=2))
    print(f'Using Device: {device} | Loaded {len(dataset)} paired patients | Outer folds: {args.outer_folds}',flush=True)
    print('M0/M1 freely calibrate pathology slope; M2 PCA16/32 + ridge; no outer tuning. '
          'Up to30 new PA sub-OOF fits; public/fold-restricted CT extraction only.',flush=True)
    public_features=feature_table(dataset,np.arange(len(dataset)),Path(args.ct_pretrained_path),True,
        Path(args.results_root)/'cache/public_ct_features.npz',args,device)
    source_checkpoints=Path(config['checkpoint_root']);bundles={};contexts={}
    metrics={'M0':[],'M1':[]};predictions={'M0':[],'M1':[]}
    for fold in args.outer_folds:
        bundle=load_bundle(fold,source,dataset.samples);bundles[fold]=bundle;oof=bundle['oof']
        for name in ('M0','M1'):
            ct=None if name=='M0' else oof.ct_risk.to_numpy()[:,None]
            head=IncrementCox().fit(oof.pathology_risk,ct,oof['dfs.month'],oof['dfs.status'])
            row,pred=export_head(name,fold,head,bundle,
                None if name=='M0' else bundle['train'].ct_risk.to_numpy()[:,None],
                None if name=='M0' else bundle['val'].ct_risk.to_numpy()[:,None],
                dict(dim=None,ridge=0.,selection='free OOF risk slopes; no hyperparameter tuning'),args)
            metrics[name].append(row);predictions[name].append(pred)
        contexts[fold]=calibration_contexts(fold,bundle,path_dataset,args,device)
    comparisons={}
    for source_name in args.ct_sources:
        name='M2_'+source_name;metrics[name]=[];predictions[name]=[]
        for fold in args.outer_folds:
            bundle=bundles[fold];oof=bundle['oof']
            inner=source_contexts(source_name,fold,contexts[fold],public_features,dataset,source_checkpoints,args,device)
            selected=select_increment(inner,dimensions=args.pca_dims,penalties=args.ridge_grid)
            print(f'{name} fold{fold}: selected PCA{selected["dim"]}, ridge{selected["ridge"]}',flush=True)
            features=final_features(source_name,fold,public_features,dataset,source_checkpoints,args,device)
            head=IncrementCox(selected['dim'],selected['ridge']).fit(oof.pathology_risk,
                features.loc[oof.index].to_numpy(),oof['dfs.month'],oof['dfs.status'])
            row,pred=export_head(name,fold,head,bundle,features.loc[bundle['train'].index].to_numpy(),
                                features.loc[bundle['val'].index].to_numpy(),selected,args)
            metrics[name].append(row);predictions[name].append(pred)
        comparisons[name]=comparison_for(name,predictions,metrics,args)
    full=sorted(args.outer_folds)==list(range(5))
    for name,rows in metrics.items():
        if full:save_cv_results(rows,Path(args.results_root)/name)
        else:pd.DataFrame(rows).to_csv(Path(args.results_root)/name/'partial_metrics.csv',index=False)
    shape_needed=full and not any(x['stable_increment'] for x in comparisons.values())
    decision=dict(shape_triggered=shape_needed,criterion='positive delta, paired lower>0, >=4/5 positive folds versus both M0/M1',
                  comparisons=comparisons,partial_run=not full)
    (Path(args.results_root)/'stage_decision.json').write_text(json.dumps(decision,indent=2))
    if shape_needed:
        print('Neither CT source meets predeclared stable-increment criterion; starting Shape and volume probes.',flush=True)
        shape_stage(bundles,contexts,public_features,dataset,source_checkpoints,predictions,metrics,args,device)
        for name in ('M3_volume','M4_shape'):
            comparison_for(name,predictions,metrics,args)
            save_cv_results(metrics[name],Path(args.results_root)/name)
    (Path(args.results_root)/'completed.json').write_text(json.dumps(dict(outer_folds=args.outer_folds,
        models=list(metrics),shape_triggered=shape_needed,diagnostic_only=True),indent=2))
    print('Feature-increment diagnostic complete.',flush=True)


def main():
    experiment=Path(__file__).resolve().parent
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source_run',default=str(experiment/'results/nested_risk_increment'))
    parser.add_argument('--results_root',default=str(experiment/'results/feature_increment'))
    parser.add_argument('--checkpoint_root',default=str(experiment/'checkpoints/feature_increment'))
    parser.add_argument('--mask_dir',default='/home/gly001/cqj/data/lung_cancer/ct/mask')
    parser.add_argument('--outer_folds',nargs='+',type=int,default=list(range(5)))
    parser.add_argument('--ct_sources',nargs='+',choices=['public','finetuned'],default=['public','finetuned'])
    parser.add_argument('--pca_dims',nargs='+',type=int,default=[16,32])
    parser.add_argument('--ridge_grid',nargs='+',type=float,default=[.01,.1,1.,10.,100.,1000.])
    parser.add_argument('--num_epochs',type=int,default=30)
    parser.add_argument('--patience',type=int,default=10)
    parser.add_argument('--num_workers',type=int,default=8)
    parser.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    args=parser.parse_args()
    if (not args.outer_folds or len(set(args.outer_folds))!=len(args.outer_folds)
        or not set(args.outer_folds).issubset(range(5)) or args.num_epochs<1 or args.num_workers<0
        or not args.pca_dims or min(args.pca_dims)<1 or not args.ridge_grid or not np.isfinite(args.ridge_grid).all()
        or min(args.ridge_grid)<=0 or len(set(args.ct_sources))!=len(args.ct_sources)):
        parser.error('Invalid fold, epoch, PCA, ridge or source configuration')
    with threadpool_limits(limits=4):run(args)


if __name__=='__main__':
    main()
