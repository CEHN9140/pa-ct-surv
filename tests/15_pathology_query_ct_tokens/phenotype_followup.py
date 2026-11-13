"""Shape-adjusted First-order increment and cached MedicalNet linear probes."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import r2_score
from sksurv.metrics import concordance_index_censored
from threadpoolctl import threadpool_limits

from ct_phenotype_features import feature_groups
from ct_phenotype_increment import load_cached_contexts,save_result
from dataset import CT_Dataset
from feature_increment import file_hash,load_bundle
from increment_cox import IncrementCox,select_increment
from phenotype_probe import fit_probe,select_probe,evaluate_probe
from risk_calibration import paired_comparison
from utils import save_cv_results


def phenotype_targets():
    groups=feature_groups()
    return groups['Shape']+groups['FirstOrder'],['ln(volume/mm3)','mm','dimensionless']+['clipped-intensity units']*5+['bits']


def read_feature_cache(path,expected_ids,cohort_hash,public,provenance):
    path=Path(path)
    if not path.is_file():raise FileNotFoundError(f'Cached CT features required; extraction/training disabled: {path}')
    metadata=json.loads(path.with_suffix('.json').read_text())
    kind='public only' if public else 'fold-restricted DFS encoder'
    if (metadata['cohort_sha256']!=cohort_hash or metadata['source']!=kind or metadata['feature_dim']!=512
            or metadata['patients']!=len(expected_ids)
            or metadata['computation']!='layer4 global average; before risk head; eval; no L2; BN unchanged'):
        raise ValueError('CT feature-cache provenance mismatch')
    checkpoint=Path(metadata['checkpoint'])
    if file_hash(checkpoint)!=metadata['checkpoint_sha256']:raise ValueError('CT feature checkpoint hash mismatch')
    with np.load(path,allow_pickle=False) as cache:
        ids=cache['case_id'].astype(str);x=cache['features']
        if list(ids)!=list(expected_ids) or len(set(ids))!=len(ids):raise ValueError('CT feature-cache patient identity mismatch')
        if x.shape!=(len(ids),512) or not np.isfinite(x).all():raise ValueError('Require finite CT features [N,512]')
        result=pd.DataFrame(x,index=ids)
    for item in [path,path.with_suffix('.json'),checkpoint]:provenance[str(item)]=file_hash(item)
    return result


def validate_ct_roles(fold,bundle,contexts,source):
    path=Path(source)/f'fold_{fold}/inner_split_roles.csv'
    roles=pd.read_csv(path,dtype={'case_id':str})
    outer=set(bundle['val'].index);training=set(bundle['oof'].index)
    if sorted(roles.inner_fold.unique())!=list(range(3)):raise ValueError('Expected three source CT inner folds')
    for inner,c in enumerate(contexts):
        rows=roles[roles.inner_fold.eq(inner)]
        groups={role:set(frame.case_id) for role,frame in rows.groupby('role')}
        held=set(bundle['oof'].index[bundle['oof'].inner_fold.eq(inner)])
        if (set(groups)!={'fit','stop','oof'} or rows.case_id.duplicated().any()
                or set(rows.case_id)!=training or set(rows.case_id)&outer or groups['oof']!=held
                or groups['oof']!=set(c['val_ids']) or groups['fit']|groups['stop']!=set(c['train_ids'])):
            raise ValueError('CT encoder fit/stop/held isolation mismatch')
    return path


def load_inputs(args):
    phenotype=Path(args.phenotype_run);feature=Path(args.feature_run)
    if not(phenotype/'completed.json').is_file() or not(feature/'completed.json').is_file():
        raise ValueError('Require completed source diagnostics')
    settings=json.loads((phenotype/'run_config.json').read_text())
    config=yaml.safe_load((feature/'run_config.yaml').read_text());source=Path(config['source_run'])
    if Path(settings['args']['feature_run']).resolve()!=feature.resolve():raise ValueError('Phenotype/CT feature source mismatch')
    if Path(settings['args']['source_run']).resolve()!=source.resolve():raise ValueError('Nested source mismatch')
    provenance={}
    for path,expected in settings['source_hashes'].items():
        actual=file_hash(path)
        if actual!=expected:raise ValueError(f'Source changed since phenotype extraction: {path}')
        provenance[path]=actual
    samples=CT_Dataset(config['data_dir'],roi_size=64).samples
    if len(samples)!=721 or int(samples.event.sum())!=251:raise ValueError('Require721 patients/251 events')
    ids=(samples.pa_id+'|'+samples.ct_id).tolist()
    targets,units=phenotype_targets()
    table=pd.read_csv(phenotype/'ct_phenotypes.csv',dtype={'case_id':str}).set_index('case_id')
    if (not table.index.is_unique or set(table.index)!=set(ids) or table.shape!=(721,27)
            or not np.isfinite(table.to_numpy()).all()):raise ValueError('Phenotype cohort/values mismatch')
    for path in [phenotype/'ct_phenotypes.csv',phenotype/'run_config.json',phenotype/'completed.json',feature/'completed.json']:
        provenance[str(path)]=file_hash(path)
    for branch in ['M0','pa_plus_ct/Shape']:
        path=phenotype/branch/'metrics.csv';provenance[str(path)]=file_hash(path)
    public=read_feature_cache(feature/'cache/public_ct_features.npz',ids,config['cohort_sha256'],True,provenance)
    public_meta=json.loads((feature/'cache/public_ct_features.json').read_text())
    if Path(public_meta['checkpoint']).resolve()!=Path(config['ct_pretrained_path']).resolve():raise ValueError('Public checkpoint source mismatch')
    source_config=yaml.safe_load((source/'run_config.yaml').read_text())
    if public_meta['checkpoint_sha256']!=source_config['public_pretrained_sha256']:
        raise ValueError('Public checkpoint differs from the original nested source')
    checkpoint_root=Path(source_config['checkpoint_root'])
    bundles={};contexts={};finetuned={}
    for fold in range(5):
        bundle=load_bundle(fold,source,samples);bundles[fold]=bundle
        for relative in [f'fold_{fold}/inner_oof_predictions.csv',
                         f'late_fusion/fold_{fold}/train_predictions.csv',f'late_fusion/fold_{fold}/val_predictions.csv']:
            path=source/relative;provenance[str(path)]=file_hash(path)
        contexts[fold]=load_cached_contexts(fold,bundle,feature)
        rolefile=validate_ct_roles(fold,bundle,contexts[fold],source);provenance[str(rolefile)]=file_hash(rolefile)
        finetuned[fold]={}
        for stage in ['inner_0','inner_1','inner_2','outer']:
            expected=ids if stage=='outer' else [x for x in ids if x in bundle['oof'].index]
            path=feature/f'cache/finetuned/fold_{fold}/{stage}.npz'
            meta=json.loads(path.with_suffix('.json').read_text())
            expected_checkpoint=checkpoint_root/f'fold_{fold}'/stage/'ct/best_model.pth'
            if Path(meta['checkpoint']).resolve()!=expected_checkpoint.resolve():raise ValueError('Fold-restricted checkpoint path mismatch')
            finetuned[fold][stage]=read_feature_cache(path,expected,config['cohort_sha256'],False,provenance)
        for split in ['train','val']:
            for branch in ['M0','pa_plus_ct/Shape']:
                path=phenotype/branch/f'fold_{fold}/{split}_predictions.csv'
                frame=pd.read_csv(path,dtype={'case_id':str}).set_index('case_id')
                truth=bundle[split]
                if not frame.index.is_unique or set(frame.index)!=set(truth.index):raise ValueError('Reference prediction identity mismatch')
                np.testing.assert_allclose(frame['dfs.month'],truth.loc[frame.index,'dfs.month'])
                np.testing.assert_array_equal(frame['dfs.status'],truth.loc[frame.index,'dfs.status'])
                provenance[str(path)]=file_hash(path)
    print('Audited721 patients,21 CT feature caches and15 isolated CT/PA inner contexts; no extraction or neural fits.',flush=True)
    return config,table[targets],units,bundles,contexts,public,finetuned,provenance


def prognosis_followup(root,args,config,targets,bundles,contexts):
    rows=[];predictions=[];references={'PA':[],'PA_Shape':[]}
    for fold,bundle in bundles.items():
        inner=[dict(c,train_ct=targets.loc[c['train_ids']].to_numpy(),val_ct=targets.loc[c['val_ids']].to_numpy())
               for c in contexts[fold]]
        selected=select_increment(inner,dimensions=[None],penalties=config['ridge_grid'])
        oof=bundle['oof']
        head=IncrementCox(ridge=selected['ridge']).fit(oof.pathology_risk,targets.loc[oof.index].to_numpy(),oof['dfs.month'],oof['dfs.status'])
        frames={split:frame[['dfs.month','dfs.status']].assign(risk_score=head.predict(frame.pathology_risk,targets.loc[frame.index].to_numpy())).reset_index()
                for split,frame in [('train',bundle['train']),('val',bundle['val'])]}
        row,pred=save_result('pa_shape_firstorder',fold,frames,root,dict(columns=targets.columns.tolist(),selection=selected,
            coefficients=head.model.coef_.tolist(),pathology_penalty=0.,ct_penalty=selected['ridge'],n_fit=len(oof)),head)
        rows.append(row);paired=pred.copy()
        for label,branch in [('PA','M0'),('PA_Shape','pa_plus_ct/Shape')]:
            ref=pd.read_csv(Path(args.phenotype_run)/branch/f'fold_{fold}/val_predictions.csv',dtype={'case_id':str}).set_index('case_id').loc[pred.case_id]
            paired[label]=ref.risk_score.to_numpy()
            references[label].append(float(concordance_index_censored(ref['dfs.status'].astype(bool),ref['dfs.month'],ref.risk_score)[0]))
        predictions.append(paired)
    save_cv_results(rows,root/'pa_shape_firstorder')
    comparison=paired_comparison(predictions,seed=config['seed'],reference_columns={'PA':'PA','PA_Shape':'PA_Shape'})
    vals=[r['val_cindex'] for r in rows]
    comparison['positive_folds']={key:int(np.sum(np.asarray(vals)>values)) for key,values in references.items()}
    comparison['folds']=[dict(fold=fold,pa=references['PA'][fold],pa_shape=references['PA_Shape'][fold],
        pa_shape_firstorder=vals[fold],delta_vs_pa=vals[fold]-references['PA'][fold],
        delta_vs_shape=vals[fold]-references['PA_Shape'][fold]) for fold in range(5)]
    pd.DataFrame(comparison['folds']).to_csv(root/'prognosis_fold_comparisons.csv',index=False)
    primary=comparison['fusion_minus_PA_Shape']
    comparison['exploratory_signal_vs_shape']=(primary['delta_cindex']>0 and primary['lower']>0 and comparison['positive_folds']['PA_Shape']>=4)
    (root/'pa_shape_firstorder/paired_comparison.json').write_text(json.dumps(comparison,indent=2))
    table=[]
    for label,branch in [('PA','M0'),('PA_Shape','pa_plus_ct/Shape'),('PA_Shape_FirstOrder',None)]:
        data=pd.DataFrame(rows) if branch is None else pd.read_csv(Path(args.phenotype_run)/branch/'metrics.csv').iloc[:5]
        for key in ['train_cindex','val_cindex','val_loss','auc_36m','auc_60m']:
            if not np.isfinite(data[key]).all():raise ValueError('Nonfinite prognosis metric')
        table.append(dict(model=label,**{key+'_'+stat:float(getattr(data[key],stat)())
            for key in ['train_cindex','val_cindex','val_loss','auc_36m','auc_60m'] for stat in ['mean','std']}))
    pd.DataFrame(table).to_csv(root/'prognosis_summary.csv',index=False)
    return table,comparison


def probe_followup(root,config,targets,units,bundles,contexts,public,finetuned):
    records=[];summary=[]
    for source in ['public','finetuned']:
        outer_predictions=[]
        for fold,bundle in bundles.items():
            inner=[]
            for number,c in enumerate(contexts[fold]):
                features=public if source=='public' else finetuned[fold][f'inner_{number}']
                inner.append(dict(train_ids=c['train_ids'],val_ids=c['val_ids'],
                    train_ct=features.loc[c['train_ids']].to_numpy(),val_ct=features.loc[c['val_ids']].to_numpy(),
                    train_targets=targets.loc[c['train_ids']].to_numpy(),val_targets=targets.loc[c['val_ids']].to_numpy()))
            selected=select_probe(inner,config['ridge_grid'],targets.columns.tolist())
            features=public if source=='public' else finetuned[fold]['outer']
            train_ids=bundle['train'].index;val_ids=bundle['val'].index
            ytrain=targets.loc[train_ids].to_numpy();truth=targets.loc[val_ids].to_numpy()
            head=fit_probe(features.loc[train_ids].to_numpy(),ytrain,selected['alphas'])
            prediction=head.predict(features.loc[val_ids].to_numpy())
            scores=evaluate_probe(ytrain,truth,prediction,targets.columns.tolist())
            folder=root/f'probes/{source}/fold_{fold}';folder.mkdir(parents=True)
            joblib.dump(head,folder/'probe.joblib')
            (folder/'selection.json').write_text(json.dumps(dict(selection=selected,training_target_means=ytrain.mean(0).tolist(),
                target_units=units,source=source,fold=fold,no_dfs_labels_in_probe=True),indent=2))
            out=pd.DataFrame({'case_id':val_ids})
            for i,column in enumerate(targets):
                out['true_'+column]=truth[:,i];out['pred_'+column]=prediction[:,i]
                out['baseline_'+column]=ytrain[:,i].mean()
                records.append(dict(source=source,fold=fold,units=units[i],ridge=selected['alphas'][i],**scores[i]))
            out.to_csv(folder/'probe_predictions.csv',index=False);outer_predictions.append(out)
            pd.DataFrame(scores).to_csv(folder/'probe_metrics.csv',index=False)
            print(f'Probe {source} fold{fold}: '+', '.join(f'{r["target"]} R2={r["r2"]:.3f}' for r in scores),flush=True)
        pooled=pd.concat(outer_predictions,ignore_index=True)
        if len(pooled)!=721 or not pooled.case_id.is_unique or set(pooled.case_id)!=set(targets.index):raise ValueError('Probe outer OOF coverage mismatch')
        for column in targets:
            rows=pd.DataFrame([r for r in records if r['source']==source and r['target']==column])
            summary.append(dict(source=source,target=column,units=rows.units.iloc[0],
                **{key+'_'+stat:float(getattr(rows[key],stat)()) for key in ['r2','mae','rmse','baseline_r2','baseline_mae','baseline_rmse'] for stat in ['mean','std']},
                pooled_outer_r2=float(r2_score(pooled['true_'+column],pooled['pred_'+column]))))
    pd.DataFrame(records).to_csv(root/'probe_fold_metrics.csv',index=False)
    pd.DataFrame(summary).to_csv(root/'probe_summary.csv',index=False)
    return summary


def write_report(root,prognosis,comparison,probes):
    lines=['# Shape-adjusted First-order and MedicalNet phenotype decoding','',
        '721 patients / 251 events / original five outer folds; cache-only diagnostics. No neural training or stage3 DFS fits.',
        '', '| Prognosis model | Train C-index | Val C-index | AUC36 | AUC60 | Val Cox NLL |',
        '|---|---|---|---|---|---|']
    for row in prognosis:
        values=[f'{row[k+"_mean"]:.4f} ± {row[k+"_std"]:.4f}' for k in ['train_cindex','val_cindex','auc_36m','auc_60m','val_loss']]
        lines.append('| '+row['model']+' | '+' | '.join(values)+' |')
    for key,title in [('PA_Shape','PA+Shape'),('PA','PA only')]:
        value=comparison['fusion_minus_'+key]
        lines.extend(['',f'PA+Shape+FirstOrder minus {title}: ΔC={value["delta_cindex"]:+.4f}, '
            f'95% paired interval[{value["lower"]:+.4f},{value["upper"]:+.4f}]; '
            f'{comparison["positive_folds"][key]}/5 positive folds.'])
    lines.extend(['','## Linear probes','',
        '| Target | Units | Public R² | Finetuned R² | Public MAE | Finetuned MAE |',
        '|---|---|---|---|---|---|'])
    for target in phenotype_targets()[0]:
        a=next(r for r in probes if r['target']==target and r['source']=='public')
        b=next(r for r in probes if r['target']==target and r['source']=='finetuned')
        lines.append(f'| {target} | {a["units"]} | {a["r2_mean"]:.4f} ± {a["r2_std"]:.4f} | '
            f'{b["r2_mean"]:.4f} ± {b["r2_std"]:.4f} | {a["mae_mean"]:.4f} ± {a["mae_std"]:.4f} | '
            f'{b["mae_mean"]:.4f} ± {b["mae_std"]:.4f} |')
    lines.extend(['','## Protocol and interpretation','',
        '- Probe: train-only feature scaling, float64 linear Ridge with intercept, target-wise inner MSE tuning. Each inner fold uses its own DFS encoder; outer validation is never used for selection. No concatenation of features from different trained encoders into one regression basis.',
        '- Probe scores use outer-validation patients only. Tables report five-fold means/sample SD; pooled outer R² is separately labeled. Baselines predict each outer training target mean. No survival outcome enters probe regression or its tuning; the finetuned encoders previously used DFS labels inside their appropriate training folds.',
        '- Low linear R² does not prove absence of nonlinear encoding. High R² does not establish DFS benefit or prove a risk head failed. No threshold for good decoding is selected from these results.',
        '- Primary prognosis contrast is PA+Shape+FirstOrder versus PA+Shape. Re-selected ridge applies to all CT terms; this evaluates a feature bundle, not a causal or independently identified coefficient effect.',
        '- All phenotype labels require masks. Probe inference uses only cached CT features. Density units are recovered clipped-input proxies, not verified raw HU. Shape volume is log-transformed, matching the earlier Cox reference.',
        '- Multiple analyses remain exploratory. Patient-paired Bootstrap conditions on fitted models and excludes retraining, tuning and earlier exploration uncertainty.',
        '- Stage3 (cross-fitted predicted phenotypes -> DFS Cox) is intentionally not run. The formal method remains a jointly trained pathology–CT teacher -> CT-only student; these cached baseline representations are diagnostics only.'])
    (root/'report.md').write_text('\n'.join(lines)+'\n')


def run(args):
    root=Path(args.results_root)
    if root.exists():raise FileExistsError(f'Refusing to overwrite {root}')
    if np.__version__!='1.26.4':raise RuntimeError('Use the existing isolated phenotype runtime (NumPy1.26.4)')
    config,targets,units,bundles,contexts,public,finetuned,provenance=load_inputs(args)
    root.mkdir(parents=True)
    (root/'run_config.json').write_text(json.dumps(dict(args=vars(args),patients=721,events=251,targets=targets.columns.tolist(),
        units=units,ridge_grid=config['ridge_grid'],seed=config['seed'],numpy_version=np.__version__,
        source_hashes=provenance,neural_fits=0,feature_extractions=0,stage3=False),indent=2))
    prognosis,comparison=prognosis_followup(root,args,config,targets,bundles,contexts)
    probes=probe_followup(root,config,targets,units,bundles,contexts,public,finetuned)
    write_report(root,prognosis,comparison,probes)
    for path,expected in provenance.items():
        if file_hash(path)!=expected:raise ValueError(f'Source changed: {path}')
    (root/'completed.json').write_text(json.dumps(dict(patients=721,events=251,folds=list(range(5)),
        prognosis_models=['PA+Shape+FirstOrder'],probe_sources=['public','finetuned'],probe_targets=9,probe_fold_target_metrics=90,
        neural_fits=0,feature_extractions=0,stage3=False,source_hashes_unchanged=True),indent=2))
    print('Shape+First-order and MedicalNet probes complete; stage3 and teacher training were not started.',flush=True)


if __name__=='__main__':
    experiment=Path(__file__).resolve().parent
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phenotype_run',default=str(experiment/'results/ct_phenotype_increment'))
    parser.add_argument('--feature_run',default=str(experiment/'results/feature_increment'))
    parser.add_argument('--results_root',default=str(experiment/'results/phenotype_followup'))
    with threadpool_limits(limits=2):run(parser.parse_args())
