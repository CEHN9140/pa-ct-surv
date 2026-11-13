"""Public MedicalNet -> cross-fitted Shape -> pathology-conditional Cox."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))

import joblib
import numpy as np
import pandas as pd
import torch
from sksurv.metrics import concordance_index_censored
from threadpoolctl import threadpool_limits

from ct_phenotype_increment import save_result
from feature_increment import file_hash
from increment_cox import IncrementCox,select_increment
from loss import cox_loss
from phenotype_followup import load_inputs
from phenotype_probe import evaluate_probe
from risk_calibration import paired_comparison
from shape_crossfit import crossfit_shape
from utils import save_cv_results


def saved_subpartitions(feature_root,fold,inner,train_ids):
    roles=pd.read_csv(Path(feature_root)/f'cache/fold_{fold}/inner_{inner}/nested_pa_roles.csv',dtype={'case_id':str})
    result=[]
    for _,rows in roles.groupby('subfold'):
        held=set(rows.loc[rows.role.eq('oof'),'case_id'])
        result.append(dict(train_ids=[x for x in train_ids if x not in held],val_ids=[x for x in train_ids if x in held]))
    return result


def save_probe_context(folder,result):
    folder=Path(folder);folder.mkdir(parents=True)
    result['train_pred'].rename_axis('case_id').to_csv(folder/'crossfit_shapes.csv')
    result['val_pred'].rename_axis('case_id').to_csv(folder/'predicted_shapes.csv')
    joblib.dump(result['head'],folder/'shape_probe.joblib')
    (folder/'crossfit_roles.json').write_text(json.dumps(result['roles'],indent=2))
    (folder/'probe_selection.json').write_text(json.dumps(dict(final=result['selection'],
        oof=[r['selection'] for r in result['roles']['oof']],n_train=len(result['train_pred']),
        n_held=len(result['val_pred']),source='fixed public512; nested target-wise MSE tuning'),indent=2))


def read_reference(folder,fold,bundle,provenance):
    folder=Path(folder);frames={}
    metric_path=folder/'metrics.csv';provenance[str(metric_path)]=file_hash(metric_path)
    rows=pd.read_csv(metric_path);row=rows.loc[rows.fold.astype(str).eq(str(fold))].iloc[0].to_dict();row['fold']=fold
    for split in ['train','val']:
        path=folder/f'fold_{fold}/{split}_predictions.csv'
        frame=pd.read_csv(path,dtype={'case_id':str});truth=bundle[split]
        if frame.case_id.duplicated().any() or set(frame.case_id)!=set(truth.index):raise ValueError('Reference case mismatch')
        np.testing.assert_allclose(frame['dfs.month'],truth.loc[frame.case_id,'dfs.month'])
        np.testing.assert_array_equal(frame['dfs.status'],truth.loc[frame.case_id,'dfs.status'])
        ci=concordance_index_censored(frame['dfs.status'].astype(bool),frame['dfs.month'],frame.risk_score)[0]
        np.testing.assert_allclose(ci,row[split+'_cindex'],rtol=0,atol=1e-12)
        row[split+'_loss']=float(cox_loss(*[torch.tensor(frame[k].to_numpy(),dtype=torch.float64)
            for k in ['risk_score','dfs.month','dfs.status']]))
        frames[split]=frame;provenance[str(path)]=file_hash(path)
    return row,frames


def write_report(root,summary,comparison,probe_metrics):
    lines=['# Cross-fitted public MedicalNet Shape increment','',
        '721 patients,251 DFS events,original five outer folds. Public-pretrained CT features only; no neural fits/extraction.',
        '', '| Model | Train C-index | Val C-index | AUC36 | AUC60 | Val Cox NLL |',
        '|---|---|---|---|---|---|']
    for row in summary:
        cells=[f'{row[k+"_mean"]:.4f} ± {row[k+"_std"]:.4f}' for k in ['train_cindex','val_cindex','auc_36m','auc_60m','val_loss']]
        lines.append('| '+row['model']+' | '+' | '.join(cells)+' |')
    for label in ['PA','true_shape','direct512']:
        value=comparison['fusion_minus_'+label]
        lines.extend(['',f'Predicted Shape minus {label}: ΔC={value["delta_cindex"]:+.4f}, '
            f'95% paired interval[{value["lower"]:+.4f},{value["upper"]:+.4f}], '
            f'{comparison["positive_folds"][label]}/5 positive folds.'])
    lines.extend(['','## Actual outer Shape decoder performance','', '| Target | R² | MAE |','|---|---|---|'])
    for target,rows in pd.DataFrame(probe_metrics).groupby('target',sort=False):
        lines.append(f'| {target} | {rows.r2.mean():.4f} ± {rows.r2.std():.4f} | {rows.mae.mean():.4f} ± {rows.mae.std():.4f} |')
    lines.extend(['','## Validation boundaries','',
        '- Cox fitting uses honest PA OOF risks and nested OOF predicted Shape. Each OOF patient is excluded from Probe fitting AND alpha tuning. Inner Cox validation is excluded from its entire upstream Probe fitting/tuning. Outer validation never tunes a decoder or Cox head.',
        '- Public CT vectors are fixed across all patients and have never used cohort DFS labels. Probe tuning is target-wise three-fold MSE, nested within each fitting subset. Saved PA sub-OOF partitions produce inner-training Shape OOF; source three inner partitions produce outer-training Shape OOF.',
        '- Cox ridge is selected using the unchanged three inner PA calibration contexts. Scalars/features are standardized only inside each fitting partition. Final validation inference uses CT vectors plus pathology risk; no validation Mask/Shape is read by Probe/Cox prediction.',
        '- The existing CT vectors were extracted from tumor-mask-localized ROI64. No extra Shape vector is needed at Probe inference, but this diagnostic does not remove the upstream localization/mask requirement.',
        '- Train predictions use the final fitted Probe and saved final PA predictor only for descriptive training metrics/KM cutoffs; they are NOT the Cox fitting inputs. Explicit OOF fitting covariates and all nested roles are saved locally for audit.',
        '- True Shape is a mask-assisted reference, not a theoretical upper bound. Direct512 uses the existing nested PCA16/32+ridge pipeline, not every possible512-D readout. Predicted Shape additionally benefits from training-mask phenotype supervision; a difference is not proof of a defective original head.',
        '- Shape volume is log(volume/mm3), diameter is in mm, sphericity dimensionless. Probe MAE follows those units. Phenotype recovery and DFS gain remain distinct questions. Actual stage3 Probe R² is reported because its strict nested tuning can differ from the earlier standalone Probe.',
        '- Mean/sample SD are over five folds. Patient-paired Bootstrap2000 conditions on fitted models and excludes retraining/tuning and prior exploration uncertainty. Multiple repeated analyses are exploratory.',
        '- DFS-finetuned stage3 is not run: it requires further inner CT encoders to avoid inherited DFS-label leakage. No trained baseline initializes a teacher/student; the formal joint teacher -> CT student method is unchanged.'])
    (root/'report.md').write_text('\n'.join(lines)+'\n')


def run(args):
    root=Path(args.results_root)
    if root.exists():raise FileExistsError(f'Refusing to overwrite {root}')
    if np.__version__!='1.26.4':raise RuntimeError('Use the isolated phenotype runtime (NumPy1.26.4)')
    config,all_targets,units,bundles,contexts,public,unused,provenance=load_inputs(args)
    del unused
    targets=all_targets.iloc[:,:3]
    source=Path(args.phenotype_run);feature=Path(args.feature_run)
    locations={'PA':source/'M0','true_shape':source/'pa_plus_ct/Shape','direct512':feature/'M2_public'}
    metrics={k:[] for k in locations};predictions={k:[] for k in locations}
    for fold,bundle in bundles.items():
        for label,folder in locations.items():
            row,frames=read_reference(folder,fold,bundle,provenance)
            metrics[label].append(row);predictions[label].append(frames['val'])
    root.mkdir(parents=True)
    (root/'run_config.json').write_text(json.dumps(dict(args=vars(args),patients=721,events=251,ct_source='public only',
        targets=targets.columns.tolist(),units=units[:3],seed=config['seed'],ridge_grid=config['ridge_grid'],
        probe_tuning='target-wise KFold3 MSE within EACH fitting subset',numpy_version=np.__version__,
        references={k:str(v) for k,v in locations.items()},source_hashes=provenance,neural_fits=0,feature_extractions=0),indent=2))
    metrics['predicted_shape']=[];predictions['predicted_shape']=[];probes=[]
    for fold,bundle in bundles.items():
        calibration=[]
        for inner,c in enumerate(contexts[fold]):
            split=saved_subpartitions(feature,fold,inner,c['train_ids'])
            result=crossfit_shape(public,targets,c['train_ids'],c['val_ids'],split,config['ridge_grid'],config['seed'])
            save_probe_context(root/f'cache/fold_{fold}/inner_{inner}',result)
            calibration.append(dict(c,train_ct=result['train_pred'].loc[c['train_ids']].to_numpy(),
                                      val_ct=result['val_pred'].loc[c['val_ids']].to_numpy()))
            print(f'Outer{fold} inner{inner}: nested Shape OOF {len(c["train_ids"])} training / {len(c["val_ids"])} held',flush=True)
        selected=select_increment(calibration,dimensions=[None],penalties=config['ridge_grid'])
        oof=bundle['oof'];train_ids=oof.index.tolist();val_ids=bundle['val'].index.tolist()
        splits=[dict(train_ids=oof.index[~oof.inner_fold.eq(i)].tolist(),val_ids=oof.index[oof.inner_fold.eq(i)].tolist()) for i in range(3)]
        result=crossfit_shape(public,targets,train_ids,val_ids,splits,config['ridge_grid'],config['seed'])
        save_probe_context(root/f'cache/fold_{fold}/outer',result)
        design=result['train_pred'].loc[train_ids]
        head=IncrementCox(ridge=selected['ridge']).fit(oof.pathology_risk,design.to_numpy(),oof['dfs.month'],oof['dfs.status'])
        fitting=design.copy();fitting['pathology_risk']=oof.pathology_risk
        fitting['dfs.month']=oof['dfs.month'];fitting['dfs.status']=oof['dfs.status']
        fitting.rename_axis('case_id').to_csv(root/f'cache/fold_{fold}/outer/cox_training_covariates.csv')
        frames={}
        for split in ['train','val']:
            frame=bundle[split]
            shape=result['head'].predict(public.loc[frame.index].to_numpy())
            if split=='val':np.testing.assert_allclose(shape,result['val_pred'].loc[frame.index].to_numpy(),rtol=1e-12,atol=1e-12)
            frames[split]=frame[['dfs.month','dfs.status']].assign(risk_score=head.predict(frame.pathology_risk,shape)).reset_index()
        row,pred=save_result('predicted_shape',fold,frames,root,dict(selection=selected,columns=targets.columns.tolist(),
            coefficients=head.model.coef_.tolist(),pathology_penalty=0.,ct_penalty=selected['ridge'],
            fitting_inputs='PA OOF plus nested predicted Shape OOF; saved in cache/fold_i/outer'),head)
        metrics['predicted_shape'].append(row);predictions['predicted_shape'].append(pred)
        scores=evaluate_probe(targets.loc[train_ids],targets.loc[val_ids],result['val_pred'].loc[val_ids],targets.columns.tolist())
        probes.extend(dict(fold=fold,**r) for r in scores)
    for label,rows in metrics.items():save_cv_results(rows,root/label)
    paired=[]
    for fold,frame in enumerate(predictions['predicted_shape']):
        frame=frame.copy()
        for label in locations:
            frame[label]=predictions[label][fold].set_index('case_id').loc[frame.case_id,'risk_score'].to_numpy()
        paired.append(frame)
    comparison=paired_comparison(paired,seed=config['seed'],reference_columns={k:k for k in locations})
    new=pd.DataFrame(metrics['predicted_shape'])
    comparison['positive_folds']={k:int((new.val_cindex>pd.DataFrame(metrics[k]).val_cindex).sum()) for k in locations}
    summary=[];fold_rows=[]
    for label in metrics:
        frame=pd.DataFrame(metrics[label])
        for key in ['train_cindex','val_cindex','val_loss','auc_36m','auc_60m']:
            if not np.isfinite(frame[key]).all():raise ValueError('Nonfinite stage3 survival metric')
        summary.append(dict(model=label,**{key+'_'+stat:float(getattr(frame[key],stat)())
            for key in ['train_cindex','val_cindex','val_loss','auc_36m','auc_60m'] for stat in ['mean','std']}))
    for fold in range(5):
        row=dict(fold=fold,**{k:metrics[k][fold]['val_cindex'] for k in metrics})
        row.update({f'delta_vs_{k}':row['predicted_shape']-row[k] for k in locations});fold_rows.append(row)
    pd.DataFrame(fold_rows).to_csv(root/'fold_comparisons.csv',index=False)
    pd.DataFrame(summary).to_csv(root/'summary.csv',index=False)
    pd.DataFrame(probes).to_csv(root/'shape_probe_metrics.csv',index=False)
    (root/'paired_comparison.json').write_text(json.dumps(comparison,indent=2))
    write_report(root,summary,comparison,probes)
    for path,expected in provenance.items():
        if file_hash(path)!=expected:raise ValueError(f'Source changed during stage3: {path}')
    (root/'completed.json').write_text(json.dumps(dict(patients=721,events=251,outer_folds=list(range(5)),
        ct_source='public only',nested_probe_contexts=20,cox_heads=5,source_hashes_unchanged=True,
        neural_fits=0,feature_extractions=0,finetuned_stage3=False),indent=2))
    print('Public predicted-Shape stage3 complete; no DFS encoder or teacher training started.',flush=True)


if __name__=='__main__':
    experiment=Path(__file__).resolve().parent
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phenotype_run',default=str(experiment/'results/ct_phenotype_increment'))
    parser.add_argument('--feature_run',default=str(experiment/'results/feature_increment'))
    parser.add_argument('--results_root',default=str(experiment/'results/predicted_shape_increment'))
    with threadpool_limits(limits=2):run(parser.parse_args())
