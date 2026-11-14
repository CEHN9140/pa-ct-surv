"""Linear Shape and size-conditioned sphere recovery after CT auxiliary training."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,r2_score

from phenotype_probe import fit_probe,select_probe,evaluate_probe
from dataset import cv_splits


def conditional_sphere(train_shapes,held_shapes):
    train,held=np.asarray(train_shapes,dtype=float),np.asarray(held_shapes,dtype=float)
    if train.shape[1]!=3 or held.shape[1]!=3 or not np.isfinite(train).all() or not np.isfinite(held).all():
        raise ValueError('Require finite V,D,S arrays for conditional recovery')
    nuisance=LinearRegression().fit(train[:,:2],train[:,2])
    return nuisance,(train[:,2]-nuisance.predict(train[:,:2]))[:,None],(held[:,2]-nuisance.predict(held[:,:2]))[:,None]


def run_shape_probes(contexts,train_features,val_features,train_shapes,val_shapes,penalties,destination):
    destination=Path(destination);destination.mkdir(parents=True)
    raw=[];residual=[]
    for c in contexts:
        nuisance,a,b=conditional_sphere(c['train_shapes'],c['val_shapes'])
        common=dict(train_ids=c['train_ids'],val_ids=c['val_ids'],train_ct=c['train_ct'],val_ct=c['val_ct'])
        raw.append(dict(common,train_targets=c['train_shapes'],val_targets=c['val_shapes']))
        residual.append(dict(common,train_targets=a,val_targets=b))
    names=['log_volume_mm3','diameter_mm','sphericity']
    raw_choice=select_probe(raw,penalties,names)
    head=fit_probe(train_features,train_shapes,raw_choice['alphas']);prediction=head.predict(val_features)
    raw_metrics=evaluate_probe(train_shapes,val_shapes,prediction,names)
    nuisance,a,b=conditional_sphere(train_shapes,val_shapes)
    residual_choice=select_probe(residual,penalties,['sphere_residual_given_true_VD'])
    residual_head=fit_probe(train_features,a,residual_choice['alphas']);residual_pred=residual_head.predict(val_features).reshape(-1,1)
    residual_metrics=evaluate_probe(a,b,residual_pred,['sphere_residual_given_true_VD'])[0]
    oracle=nuisance.predict(np.asarray(val_shapes)[:,:2]);reconstructed=oracle+residual_pred[:,0]
    residual_metrics.update(oracle_VD_r2=float(r2_score(np.asarray(val_shapes)[:,2],oracle)),
        reconstructed_sphere_r2=float(r2_score(np.asarray(val_shapes)[:,2],reconstructed)),
        reconstructed_sphere_mae=float(mean_absolute_error(np.asarray(val_shapes)[:,2],reconstructed)))
    joblib.dump(dict(raw=head,residual=residual_head,nuisance=nuisance),destination/'probes.joblib')
    (destination/'selection.json').write_text(json.dumps(dict(raw=raw_choice,residual=residual_choice,
        nuisance_fit='outer training true V,D -> S only; inner nuisance fits each inner training part',
        conditional_metric='residual trueS-g(trueVD); uses held trueVD ONLY for diagnostic labels/oracle comparator'),indent=2))
    return raw_metrics+[residual_metrics],prediction,residual_pred


def evaluate_selected_probes(dataset,shapes,partitions,selected_features,baseline_features,args,root):
    records=[]
    samples=dataset.samples
    ids=(samples.pa_id+'|'+samples.ct_id).to_numpy()
    feature_config=yaml.safe_load((Path(args.feature_run)/'run_config.yaml').read_text())
    for label,features in [('A',baseline_features),('aux',selected_features)]:
        for fold,(training,validation) in enumerate(cv_splits(samples)):
            contexts=[]
            for number,fit,stop,held in partitions[fold]:
                train_ids=ids[np.sort(np.r_[fit,stop])].tolist();held_ids=ids[held].tolist()
                x=features[fold][f'inner_{number}']
                if set(train_ids)&set(held_ids) or not(set(train_ids)|set(held_ids))<=set(ids[training]):
                    raise ValueError('Shape probe fitting/held isolation mismatch')
                contexts.append(dict(train_ids=train_ids,val_ids=held_ids,
                    train_ct=x.loc[train_ids].to_numpy(),val_ct=x.loc[held_ids].to_numpy(),
                    train_shapes=shapes.loc[train_ids].to_numpy(),val_shapes=shapes.loc[held_ids].to_numpy()))
            train_ids=ids[training];val_ids=ids[validation];x=features[fold]['outer']
            folder=Path(root)/f'probes/{label}/fold_{fold}'
            scores,prediction,residual_pred=run_shape_probes(contexts,x.loc[train_ids].to_numpy(),x.loc[val_ids].to_numpy(),
                shapes.loc[train_ids].to_numpy(),shapes.loc[val_ids].to_numpy(),feature_config['ridge_grid'],folder)
            out=pd.DataFrame(dict(case_id=val_ids,pred_sphere_residual=residual_pred[:,0]))
            for i,column in enumerate(shapes.columns):out['true_'+column]=shapes.loc[val_ids,column].to_numpy();out['pred_'+column]=prediction[:,i]
            out.to_csv(folder/'probe_predictions.csv',index=False)
            records.extend(dict(source=label,fold=fold,**r) for r in scores)
            print(f'Outer Probe {label} fold{fold} | '+', '.join(f'{r["target"]}:R2={r["r2"]:.3f}' for r in scores),flush=True)
    frame=pd.DataFrame(records);frame.to_csv(Path(root)/'probe_fold_metrics.csv',index=False)
    rows=[]
    for (source,target),data in frame.groupby(['source','target'],sort=False):
        rows.append(dict(source=source,target=target,**{key+'_'+stat:float(getattr(data[key],stat)())
            for key in ['r2','mae','rmse','baseline_r2','baseline_mae','baseline_rmse'] for stat in ['mean','std']}))
    pd.DataFrame(rows).to_csv(Path(root)/'probe_summary.csv',index=False)
    return rows


def write_aux_report(root,results,comparison,probe_rows,selections,args):
    baseline=pd.read_csv(Path(args.source_run)/'ct/metrics.csv').iloc[:5];aux=pd.DataFrame(results)
    lines=['# CT-only auxiliary Shape supervision','',
        f'Target={args.target};721 patients/251 events,original outer5/inner3 roles. Public-only fresh initialization for every new fit.',
        '', '| Model | Train C-index | Val C-index | AUC36 | AUC60 |','|---|---|---|---|---|']
    for name,frame in [('A Cox baseline',baseline),('Auxiliary selected by inner HELD',aux)]:
        cells=[f'{frame[k].mean():.4f} ± {frame[k].std():.4f}' for k in ['train_cindex','val_cindex','auc_36m','auc_60m']]
        lines.append('| '+name+' | '+' | '.join(cells)+' |')
    interval=comparison['fusion_minus_A']
    lines.extend(['',f'ΔC vs A={interval["delta_cindex"]:+.4f},95% paired interval[{interval["lower"]:+.4f},{interval["upper"]:+.4f}], '
        f'{comparison["positive_folds"]}/5 positive folds.',
        '', 'Selected lambda / epochs by outer fold: '+', '.join(f'{i}:{s["weight"]:g}/{s["epochs"]}' for i,s in enumerate(selections))+'.',
        '', '## Linear recovery on outer validation','', '| Source | Target | R² | MAE |','|---|---|---|---|'])
    for r in probe_rows:
        lines.append(f'| {r["source"]} | {r["target"]} | {r["r2_mean"]:.4f} ± {r["r2_std"]:.4f} | {r["mae_mean"]:.4f} ± {r["mae_std"]:.4f} |')
    lines.extend(['','## Boundaries','',
        '- STOP C-index selects each epoch; separate inner HELD C-index selects lambda0/.05/.2. Exact ties prefer smaller lambda. Outer epochs are floor median chosen-candidate STOP epochs; outer patients never tune anything.',
        '- A/lambda0 reuses matched completed control outputs and representation caches; this is not auxiliary initialization or warm-start. Nonzero candidates and outer refits start only from public MedicalNet/new heads. All backbone, risk and auxiliary parameters train; BN running statistics remain fixed.',
        '- Original risk head512->256->ReLU->Dropout.25->1 is preserved. Extra head512->K has no dropout and its construction preserves baseline RNG. True batch32+tail,optimizer/LRs,seed and inputs match A.',
        '- Shape targets use only neural FIT population mean/std; outer refits use only outertrain statistics. These buffers are checkpointed. MSE averages patients and targets: at the same lambda each C target receives one third of the B per-target weight. B/C is a supervision-configuration comparison,not a causal isolation of two extra targets.',
        '- A and selected lambda0 have no trained auxiliary head: direct auxiliary-head MSE/R²/MAE are N/A. Nonzero direct-head errors are inverse-transformed using checkpoint FIT scalers. Probes are separately fitted readouts for all groups.',
        '- Residual target is S-g(logV,D),where unpenalized LINEAR g is fitted on the corresponding Probe training patients only. It is not complete conditional independence from VD; nonlinear size relationships can remain and CT features themselves were not residualized. Residual R² and reconstructed-S R² have different denominators and are reported separately in fold metrics.',
        '- Oracle VD baseline/reconstruction uses held trueVD only for diagnostic labels/comparators,not deployed risk prediction. Risk inference uses CT alone on the existing mask-localized ROI64; training Shape targets come from masks.',
        '- Lambda and Probe alpha reuse outer-training inner partitions for tuning; inner Probe errors are not independent validation. Each inner readout uses one appropriate encoder basis; the final readout uses its outer encoder. Outer Probe results measure recovery,not DFS success.',
        '- Target choice follows earlier exploration on this same cohort; nested CV here does not erase historical selection bias. Paired Bootstrap2000 conditions on fitted models and excludes retraining/prior exploration uncertainty. Independent confirmation is still needed.',
        '- No teacher/student or formal method change was launched. A better Shape R² alone is not success: the primary outcome is paired CT-only DFS improvement.'])
    (Path(root)/'report.md').write_text('\n'.join(lines)+'\n')
