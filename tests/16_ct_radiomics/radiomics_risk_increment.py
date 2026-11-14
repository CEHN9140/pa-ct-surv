"""M3: strictly cross-fitted Radiomics risk added to existing nested PA risk."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'15_pathology_query_ct_tokens'))

import joblib
import numpy as np
import pandas as pd
import torch
import yaml
from threadpoolctl import threadpool_limits

from increment_cox import IncrementCox
from risk_calibration import risk_scaler, standardize_risk, paired_comparison
from radiomics_ct import load_cohort, load_features, extraction_signature, feature_names, select_penalty, fit_cox
from teacher_data import file_digest
from loss import cox_loss
from utils import save_cv_results
from visualization.survival_plots import analyze_fold


def checked_table(frame,samples,indices):
    truth=samples.iloc[indices].set_index('case_id')
    required=['case_id','dfs.month','dfs.status']
    if (not set(required).issubset(frame) or frame.case_id.duplicated().any()
            or set(frame.case_id)!=set(truth.index)):
        raise ValueError('Prediction identity does not match the intended patients')
    result=frame.set_index('case_id').loc[truth.index].copy()
    if (not np.isfinite(result.to_numpy(dtype=float)).all()
            or not np.array_equal(result['dfs.status'],truth.event)
            or not np.array_equal(result['dfs.month'].to_numpy(dtype=np.float32),truth.time.to_numpy(dtype=np.float32))):
        raise ValueError('Nonfinite predictions or labels differ from source cohort')
    result['dfs.month'],result['dfs.status']=truth.time.to_numpy(),truth.event.to_numpy()
    return result


def checked_roles(samples,table,fold):
    allowed=set(samples.loc[samples.split.ne(fold),'case_id'])
    lookup={identity:i for i,identity in enumerate(samples.case_id)}
    if set(table.inner_fold)!={0,1,2}:
        raise ValueError('Require three complete inner roles')
    partitions=[];coverage=[]
    for inner,rows in table.groupby('inner_fold',sort=True):
        if (rows.case_id.duplicated().any() or set(rows.case_id)!=allowed
                or set(rows.role)!={'fit','stop','oof'}):
            raise ValueError('Inner roles must partition outer training, excluding outer validation')
        parts=[np.array([lookup[i] for i in rows.loc[rows.role.eq(role),'case_id']],dtype=int)
               for role in ['fit','stop','oof']]
        partitions.append((int(inner),*parts));coverage.extend(parts[-1])
    if len(coverage)!=len(allowed) or len(set(coverage))!=len(allowed):
        raise ValueError('Require exactly one HELD prediction for every outer-training patient')
    return partitions


def radiomics_risks(x,samples,fold,partitions,directory):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=False)
    train=np.flatnonzero(samples.split.to_numpy()!=fold)
    val=np.flatnonzero(samples.split.to_numpy()==fold)
    if x.index.tolist()!=samples.case_id.tolist():
        raise ValueError('Radiomics row order differs from paired patients')
    all_held=[];oof=[];outer={}
    for inner,fit,stop,held in partitions:
        training=np.sort(np.r_[fit,stop])
        if (len(set(training))!=len(training) or set(training)&set(held)
                or set(training)|set(held)!=set(train)):
            raise ValueError('OOF fit/selection patients overlap HELD or outer validation')
        all_held.extend(held)
    if len(all_held)!=len(train) or set(all_held)!=set(train):
        raise ValueError('Incomplete/repeated Radiomics OOF coverage')
    jobs=[(f'inner_{inner}',np.sort(np.r_[fit,stop]),held,inner)
          for inner,fit,stop,held in partitions]+[('outer',train,val,None)]
    for name,training,prediction,inner in jobs:
        folder=directory/name;folder.mkdir()
        print(f'Radiomics outer{fold}/{name}: tuning only on {len(training)} patients; predicting {len(prediction)}',flush=True)
        penalty,tuning=select_penalty(x.iloc[training],samples.iloc[training].time.to_numpy(),
                                      samples.iloc[training].event.to_numpy())
        model=fit_cox(x.iloc[training],samples.iloc[training].time,samples.iloc[training].event,penalty)
        fit_risk=model.predict(x.iloc[training]);scaler=risk_scaler(fit_risk)
        joblib.dump(model,folder/'ct_cox.joblib')
        tuning.to_csv(folder/'inner_cv.csv',index=False)
        model.named_steps['filter'].audit_.to_csv(folder/'feature_selection.csv',index=False)
        metadata=dict(penalty=float(penalty),fit_n=len(training),
            fit_case_ids=samples.iloc[training].case_id.tolist(),
            prediction_case_ids=samples.iloc[prediction].case_id.tolist(),risk_scaler=scaler,
            selection='3-fold CV inside base fitting patients; held and outer labels excluded',
            n_selected=len(model.named_steps['filter'].selected_))
        (folder/'metadata.json').write_text(json.dumps(metadata,indent=2))
        for split,indices in ([('held',prediction)] if inner is not None else [('train',train),('val',val)]):
            rows=samples.iloc[indices].set_index('case_id')[['time','event']].rename(
                columns={'time':'dfs.month','event':'dfs.status'})
            rows['raw_radiomics_risk']=model.predict(x.iloc[indices])
            rows['radiomics_risk']=standardize_risk(rows.raw_radiomics_risk,scaler)
            if inner is not None:rows['inner_fold']=inner;oof.append(rows)
            else:outer[split]=rows
            rows.to_csv(folder/f'{split}_predictions.csv')
    outer['oof']=pd.concat(oof).loc[samples.iloc[train].case_id]
    outer['oof'].to_csv(directory/'radiomics_oof_predictions.csv')
    return outer


def fit_heads(oof,ridge):
    columns=['pathology_risk','ct_A','ct_C','radiomics_risk','dfs.month','dfs.status']
    if not np.isfinite(ridge) or ridge<0 or not oof.index.is_unique or not np.isfinite(oof[columns]).all().all():
        raise ValueError('Require finite unique OOF covariates and nonnegative ridge')
    return {name:IncrementCox(ridge=ridge).fit(oof.pathology_risk,
        None if column is None else oof[[column]].to_numpy(),oof['dfs.month'],oof['dfs.status'])
        for name,column in [('M0',None),('M1','ct_A'),('M2','ct_C'),('M3','radiomics_risk')]}


def load_sources(args):
    source,comparison=Path(args.source_run),Path(args.comparison_run)
    config=yaml.safe_load((source/'run_config.yaml').read_text())
    previous=yaml.safe_load((comparison/'run_config.yaml').read_text())
    if (Path(previous['source_run']).resolve()!=source.resolve() or previous['cox_ridge']!=args.cox_ridge
            or previous['shape_weight']!=.2 or previous['seed']!=123 or config['seed']!=123
            or config['inner_folds']!=3):
        raise ValueError('Require matching fixed-ridge1/Shape-lambda0.2/seed123 comparison')
    samples=pd.read_csv(args.cohort_csv,dtype={'pa_id':str,'ct_id':str})
    samples['case_id']=samples.pa_id+'|'+samples.ct_id
    checksum=file_digest(args.cohort_csv)
    if (len(samples)!=721 or samples.event.sum()!=251 or checksum!=config['cohort_sha256']
            or checksum!=previous['cohort_sha256']):
        raise ValueError('Require unchanged721/251 source cohort')
    feature_path=Path(args.features_csv)
    extraction=yaml.safe_load((feature_path.parent/'run_config.yaml').read_text())
    cohort=load_cohort(args.cohort_csv,extraction['ct_root'])
    if json.loads((feature_path.parent/'extraction_manifest.json').read_text())!=extraction_signature(cohort,args.cohort_csv):
        raise ValueError('Radiomics extraction source/settings changed')
    features=load_features(feature_path,cohort)[feature_names()]
    features.index=samples.case_id
    provenance={}
    def read(path):
        path=Path(path);provenance[str(path.resolve())]=file_digest(path)
        return pd.read_csv(path,dtype={'case_id':str},float_precision='round_trip')
    bundles=[]
    for fold in range(5):
        train=np.flatnonzero(samples.split.to_numpy()!=fold);val=np.flatnonzero(samples.split.to_numpy()==fold)
        roles=read(source/f'fold_{fold}/inner_split_roles.csv')
        partitions=checked_roles(samples,roles,fold)
        original=checked_table(read(source/f'fold_{fold}/inner_oof_predictions.csv'),samples,train)
        oof=checked_table(read(comparison/f'fold_{fold}/calibration_oof_covariates.csv'),samples,train)
        np.testing.assert_allclose(oof.pathology_risk,original.pathology_risk,rtol=1e-12,atol=1e-12)
        np.testing.assert_allclose(oof.ct_A,original.ct_risk,rtol=1e-12,atol=1e-12)
        calibration_path=source/f'fold_{fold}/calibration.yaml'
        calibration=yaml.safe_load(calibration_path.read_text())
        provenance[str(calibration_path.resolve())]=file_digest(calibration_path)
        for modality in ['pathology','ct']:
            epochs=int(np.median([row['selected_epoch'] for row in calibration['inner_models']
                                  if row['modality']==modality]))
            path=Path(config['checkpoint_root'])/f'fold_{fold}/outer/{modality}/checkpoint_metadata.yaml'
            metadata=yaml.safe_load(path.read_text());provenance[str(path.resolve())]=file_digest(path)
            if (calibration['selected_epochs'][modality]!=epochs
                    or metadata['selection']!='fixed_epochs_from_inner_training'
                    or metadata['selected_epoch']!=epochs or metadata['epochs_completed']!=epochs):
                raise ValueError('Source outer neural epochs were not fixed from inner STOP selection')
        for inner,fit,stop,held in partitions:
            ids=samples.iloc[held].case_id
            if not original.loc[ids,'inner_fold'].eq(inner).all() or not oof.loc[ids,'inner_fold'].eq(inner).all():
                raise ValueError('OOF rows do not match the patient-excluded inner model')
            for modality in ['pathology','ct']:
                record=next(row for row in calibration['inner_models'] if row['inner_fold']==inner and row['modality']==modality)
                if record['scaler']['n_fit']!=len(fit) or record['selection']!='inner_stop_cindex':
                    raise ValueError('Neural OOF scaler or STOP selection differs')
                checkpoint=Path(config['checkpoint_root'])/f'fold_{fold}/inner_{inner}/{modality}'
                history=read(checkpoint/'training_history.csv')
                if int(history.loc[history.val_cindex.idxmax(),'epoch'])!=record['selected_epoch']:
                    raise ValueError('Source neural epoch did not follow STOP history')
                np.testing.assert_allclose(original.loc[ids,modality+'_risk'],
                    standardize_risk(original.loc[ids,'raw_'+modality+'_risk'],record['scaler']),rtol=1e-12,atol=1e-12)
        ct_oof=checked_table(read(comparison/f'fold_{fold}/ct_fixed_inner_oof.csv'),samples,train)
        np.testing.assert_allclose(oof.ct_C,ct_oof.ct_risk,rtol=1e-12,atol=1e-12)
        ct_meta_path=comparison/f'fold_{fold}/ct_risk_provenance.json'
        ct_meta=json.loads(ct_meta_path.read_text());provenance[str(ct_meta_path.resolve())]=file_digest(ct_meta_path)
        if (ct_meta['weight']!=.2 or len(ct_meta['inner_fit_scalers'])!=3
                or ct_meta['epochs']!=int(np.median(ct_meta['inner_STOP_epochs']))):
            raise ValueError('Source C must use fixed ALL-Shape lambda0.2')
        for inner,fit,stop,held in partitions:
            ids=samples.iloc[held].case_id
            scaler=ct_meta['inner_fit_scalers'][inner]
            if scaler['n_fit']!=len(fit) or not ct_oof.loc[ids,'inner_fold'].eq(inner).all():
                raise ValueError('CT C OOF membership/scaler does not match source FIT')
            np.testing.assert_allclose(ct_oof.loc[ids,'ct_risk'],
                standardize_risk(ct_oof.loc[ids,'raw_ct_risk'],scaler),rtol=1e-12,atol=1e-12)
        frames={}
        for split,indices in [('train',train),('val',val)]:
            base=checked_table(read(source/f'late_fusion/fold_{fold}/{split}_predictions.csv'),samples,indices)
            c=checked_table(read(comparison/f'fold_{fold}/ct_fixed_{split}_predictions.csv'),samples,indices)
            np.testing.assert_allclose(c.ct_risk,standardize_risk(c.risk_score,
                ct_meta['outer_train_scaler']),rtol=1e-12,atol=1e-12)
            base['ct_A'],base['ct_C']=base.ct_risk,c.ct_risk
            for modality in ['pathology','ct']:
                raw=checked_table(read(source/f'{modality}/fold_{fold}/{split}_predictions.csv'),samples,indices)
                np.testing.assert_allclose(base['raw_'+modality+'_risk'],raw.risk_score,rtol=1e-12,atol=1e-12)
                np.testing.assert_allclose(base[modality+'_risk'],standardize_risk(raw.risk_score,
                    calibration['outer_scalers'][modality]),rtol=1e-12,atol=1e-12)
            frames[split]=base
        references={name:{split:checked_table(read(comparison/name/f'fold_{fold}/{split}_predictions.csv'),
            samples,indices) for split,indices in [('train',train),('val',val)]} for name in ['M0','M1','M2']}
        bundles.append(dict(partitions=partitions,oof=oof,references=references,**frames))
    for path in [Path(args.cohort_csv),feature_path,feature_path.parent/'run_config.yaml',
                 feature_path.parent/'extraction_manifest.json',source/'run_config.yaml',comparison/'run_config.yaml',
                 comparison/'completed.json']:
        provenance[str(path.resolve())]=file_digest(path)
    print('Source audit passed:721patients/251events,strict PA/A/C OOF and five outer folds; no neural training.',flush=True)
    return samples,features,bundles,provenance


def run(args):
    if args.cox_ridge!=1.:
        raise ValueError('This matched diagnostic fixes CT ridge1 and unpenalized PA')
    samples,x,bundles,provenance=load_sources(args)
    if args.audit_only:return
    torch.set_num_threads(2)
    root=Path(args.results_root)
    root.mkdir(parents=True,exist_ok=False)
    config=dict(vars(args),seed=123,radiomics_penalty_grid=[.01,.1,1.,10.,100.,1000.],
        pathology='Same strict nested original-v7 ABMIL risks as test15, outer epochs1/14/1/14/23',
        diagnostic_only=True,neural_training=False,pa_penalty=0.,source_hashes=provenance,
        scope='Full Radiomics scalar risk M3; no feature-group search, no teacher/student modification')
    (root/'run_config.yaml').write_text(yaml.safe_dump(config,allow_unicode=True))
    metrics={name:[] for name in ['M0','M1','M2','M3']};predictions={name:[] for name in metrics}
    for fold,bundle in enumerate(bundles):
        radio=radiomics_risks(x,samples,fold,bundle['partitions'],root/f'ct/fold_{fold}')
        oof=bundle['oof'].copy();oof['radiomics_risk']=radio['oof'].radiomics_risk
        np.testing.assert_array_equal(oof.inner_fold,radio['oof'].inner_fold)
        oof.to_csv(root/f'ct/fold_{fold}/fusion_oof_covariates.csv')
        heads=fit_heads(oof,args.cox_ridge)
        for name,head in heads.items():
            folder=root/name/f'fold_{fold}';folder.mkdir(parents=True)
            joblib.dump(head,folder/'head.joblib')
            frames={}
            for split in ['train','val']:
                base=bundle[split]
                covariate=(None if name=='M0' else radio[split].loc[base.index,['radiomics_risk']].to_numpy()
                           if name=='M3' else base[['ct_A' if name=='M1' else 'ct_C']].to_numpy())
                frame=base[['dfs.month','dfs.status']].copy()
                frame['risk_score']=head.predict(base.pathology_risk,covariate)
                if name!='M3':
                    reference=bundle['references'][name][split]
                    np.testing.assert_allclose(frame.risk_score,reference.risk_score,rtol=1e-10,atol=1e-10)
                frames[split]=frame.reset_index()
            frames['train'].to_csv(folder/'train_predictions.csv',index=False)
            extra=analyze_fold(frames['train'],frames['val'],folder)
            losses={split+'_loss':float(cox_loss(*[torch.tensor(frame[key].to_numpy(),dtype=torch.float64)
                for key in ['risk_score','dfs.month','dfs.status']])) for split,frame in frames.items()}
            metrics[name].append(dict(fold=fold,**extra,**losses));predictions[name].append(frames['val'])
            (folder/'head_parameters.json').write_text(json.dumps(dict(coefficients=head.model.coef_.tolist(),
                pa_oof_mean=head.pa_scaler.mean_.tolist(),pa_oof_scale=head.pa_scaler.scale_.tolist(),
                ct_oof_mean=None if head.ct_pipeline is None else head.ct_pipeline.named_steps['scale'].mean_.tolist(),
                ct_oof_scale=None if head.ct_pipeline is None else head.ct_pipeline.named_steps['scale'].scale_.tolist(),
                ridge_ct=1.,ridge_pa=0.,n_oof=len(oof)),indent=2))
            print(f'{name} Outer{fold}: Val C-index={extra["val_cindex"]:.4f}',flush=True)
    for name,rows in metrics.items():save_cv_results(rows,root/name)
    report(root,metrics,predictions)
    for path,checksum in provenance.items():
        if file_digest(path)!=checksum:raise ValueError(f'Source changed during diagnosis: {path}')
    (root/'completed.json').write_text(json.dumps(dict(patients=721,events=251,outer_folds=5,
        radiomics_base_fits=20,radiomics_penalty_cv_fits=360,fusion_heads=20,
        source_hashes_unchanged=True,neural_training=False,source_M0_M1_M2_reproduced=True),indent=2))


def report(root,metrics,predictions):
    summary=pd.DataFrame([dict(model=name,**{key+'_'+stat:float(getattr(pd.DataFrame(rows)[key],stat)())
        for key in ['train_cindex','val_cindex','val_loss','auc_36m','auc_60m','gap'] for stat in ['mean','std']})
        for name,rows in metrics.items()])
    summary.to_csv(root/'summary.csv',index=False)
    pd.DataFrame([dict(fold=fold,**{name+'_val_cindex':metrics[name][fold]['val_cindex'] for name in metrics},
        delta_M3_M0=metrics['M3'][fold]['val_cindex']-metrics['M0'][fold]['val_cindex'],
        delta_M3_M2=metrics['M3'][fold]['val_cindex']-metrics['M2'][fold]['val_cindex'])
        for fold in range(5)]).to_csv(root/'per_fold_comparison.csv',index=False)
    comparisons=[]
    for candidate,control in [('M1','M0'),('M2','M0'),('M3','M0'),('M3','M1'),('M3','M2')]:
        pairs=[];corrected=broken=0
        for current,reference in zip(predictions[candidate],predictions[control]):
            frame=current.copy();reference=reference.set_index('case_id').loc[current.case_id]
            frame['reference']=reference.risk_score.to_numpy();pairs.append(frame)
            t=frame['dfs.month'].to_numpy();e=frame['dfs.status'].to_numpy().astype(bool)
            valid=e[:,None]&((t[:,None]<t[None,:])|((t[:,None]==t[None,:])&~e[None,:]))
            r=frame.risk_score.to_numpy();s=frame.reference.to_numpy()
            corrected+=int((valid&((r[:,None]-r[None,:])>1e-8)&((s[:,None]-s[None,:])< -1e-8)).sum())
            broken+=int((valid&((s[:,None]-s[None,:])>1e-8)&((r[:,None]-r[None,:])< -1e-8)).sum())
        result=paired_comparison(pairs,seed=123,reference_columns={'reference':'reference'})['fusion_minus_reference']
        comparisons.append(dict(model=candidate,reference=control,**result,
            positive_folds=sum(a['val_cindex']>b['val_cindex'] for a,b in zip(metrics[candidate],metrics[control])),
            corrected_pairs=corrected,broken_pairs=broken))
    pd.DataFrame(comparisons).to_csv(root/'paired_comparisons.csv',index=False)
    lines=['# Full Radiomics risk conditional increment','',
        'Same721 patients/251 events, outer5 and inner3 roles as test15. PA/A/C risks remain fixed. '
        'M0=PA; M1=PA+original CT; M2=PA+ALL-Shape CT(lambda0.2); M3=PA+Original107 Radiomics Risk. '
        'All heads use fixed CT ridge1, no PA penalty and free slopes.','',
        '| Model | Train C-index | Val C-index | AUC36 | AUC60 | Val Cox NLL |','|---|---|---|---|---|---|']
    for row in summary.to_dict('records'):
        lines.append('| '+row['model']+' | '+' | '.join(f'{row[k+"_mean"]:.4f} ± {row[k+"_std"]:.4f}'
            for k in ['train_cindex','val_cindex','auc_36m','auc_60m','val_loss'])+' |')
    lines+=['','| Contrast | Delta C-index | Conditional95% interval | Positive folds | Corrected/broken |',
        '|---|---|---|---|---|']
    for row in comparisons:
        lines.append(f'| {row["model"]}-{row["reference"]} | {row["delta_cindex"]:+.4f} | '
            f'[{row["lower"]:+.4f}, {row["upper"]:+.4f}] | {row["positive_folds"]}/5 | '
            f'{row["corrected_pairs"]}/{row["broken_pairs"]} |')
    lines+=['','Radiomics OOF base fitting and all CV tuning exclude HELD and outer validation. '
        'Inner base models fit FIT+STOP; each filter/scaler is refitted within each tuning fold. '
        'Risk normalization uses base fitting predictions; head scalers use only OOF covariates. '
        'Outer fitting/penalty selection uses outer training only. M0/M1/M2 reproduce prior exports.',
        '', 'Bootstrap2000, fixed folds/patient resampling, conditional on saved models; excludes retraining, '
        'earlier selection and multiple-comparison uncertainty. Patient pairs are not independent.',
        '', 'The strict PA baseline has historical outer epochs1/14/1/14/23; two short refits remain a limitation. '
        'Historical PA0.7528 used outer stopping and is not a matched benchmark. '
        'A small scalar-risk increment does not exclude complementary high-dimensional information. '
        'Radiomics uses complete tumor masks/current NIfTI intensities, not only ROI64. '
        'This is diagnostic reuse of predictions, not warm-start/frozen unimodal teacher training. '
        'No neural training, feature-group search or Brier-score analysis was performed.']
    (root/'report.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    experiment=Path(__file__).resolve().parent;test15=experiment.parent/'15_pathology_query_ct_tokens'
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cohort_csv',default=str(experiment.parents[1]/'data/eagle/top25/all_label_roi64.csv'))
    parser.add_argument('--features_csv',default=str(experiment/'output/features.csv'))
    parser.add_argument('--source_run',default=str(test15/'results/nested_risk_increment'))
    parser.add_argument('--comparison_run',default=str(test15/'results/shape_risk_increment'))
    parser.add_argument('--results_root',default=str(experiment/'conditional_risk_increment'))
    parser.add_argument('--cox_ridge',type=float,default=1.)
    parser.add_argument('--audit_only',action='store_true')
    with threadpool_limits(limits=2):run(parser.parse_args())
