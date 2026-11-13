"""Stage1 only: nested Cox ablations of true and already cross-fitted Shape."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))

import joblib
import numpy as np
import pandas as pd
import yaml
from threadpoolctl import threadpool_limits

from ct_phenotype_increment import load_cached_contexts,save_result
from dataset import CT_Dataset
from feature_increment import file_hash,load_bundle
from increment_cox import IncrementCox,select_increment
from phenotype_followup import read_feature_cache
from predicted_shape_increment import read_reference
from risk_calibration import paired_comparison
from utils import save_cv_results


def shape_groups():
    v,d,s='log_volume_mm3','diameter_mm','sphericity'
    return dict(V=[v],D=[d],S=[s],VD=[v,d],VS=[v,s],DS=[d,s],VDS=[v,d,s])


def conditional_contrasts():
    return [('VDS','DS','V'),('VDS','VS','D'),('VDS','VD','S'),('VD','V','D'),('VD','D','V')]


def verify_reference_risk(current,reference):
    current,reference=np.asarray(current,dtype=float),np.asarray(reference,dtype=float)
    # Refitting Cox at tol=1e-9 can differ by a few1e-9 after CSV roundtrip.
    # Require BOTH tiny absolute error and identical pair ordering/ties.
    np.testing.assert_allclose(current,reference,rtol=0,atol=1e-8)
    tags=[]
    for risk in [current,reference]:
        delta=risk[:,None]-risk[None,:]
        tags.append(np.where(abs(delta)<=1e-8,0,np.sign(delta)))
    np.testing.assert_array_equal(*tags)
    return float(np.max(abs(current-reference)))


def read_shape_context(folder,train_ids,val_ids,outer_ids,provenance):
    folder=Path(folder);train,val=set(train_ids),set(val_ids)
    if train&val or train&set(outer_ids):raise ValueError('Shape context patient overlap')
    frames=[]
    for filename,expected in [('crossfit_shapes.csv',train_ids),('predicted_shapes.csv',val_ids)]:
        path=folder/filename
        frame=pd.read_csv(path,dtype={'case_id':str}).set_index('case_id')
        if (not frame.index.is_unique or len(set(expected))!=len(expected) or set(frame.index)!=set(expected)
                or frame.columns.tolist()!=shape_groups()['VDS']):raise ValueError('Shape cache identity/column mismatch')
        if not np.isfinite(frame.to_numpy()).all():raise ValueError('Require finite saved Shape predictions')
        frames.append(frame.loc[expected]);provenance[str(path)]=file_hash(path)
    path=folder/'crossfit_roles.json';roles=json.loads(path.read_text());counts={x:0 for x in train}
    for step in roles['oof']+[roles['final']]:
        a,b=step['train_ids'],step['val_ids'];sa,sb=set(a),set(b)
        if len(sa)!=len(a) or len(sb)!=len(b) or sa&sb or not sa<=train:
            raise ValueError('Invalid saved Shape fitting partition')
        if step is roles['final']:
            if sa!=train or sb!=val:raise ValueError('Invalid final Probe partition')
        else:
            if sa|sb!=train:raise ValueError('Invalid Shape OOF partition')
            for patient in b:counts[patient]+=1
        for split in step['tuning_roles']:
            fit,held=split['train_ids'],split['val_ids']
            if set(fit)&set(held) or set(fit)|set(held)!=sa or len(set(fit))+len(set(held))!=len(fit)+len(held):
                raise ValueError('Invalid nested Probe tuning partition')
    if not counts or set(counts.values())!={1}:raise ValueError('Incomplete saved Shape OOF partition')
    provenance[str(path)]=file_hash(path)
    return tuple(frames)


def load_ablation_inputs(args):
    source=Path(args.predicted_shape_run)
    done=json.loads((source/'completed.json').read_text());review=json.loads((source/'completion_review.json').read_text())
    if not review['passed'] or done['patients']!=721 or done['events']!=251 or done['ct_source']!='public only':
        raise ValueError('Require verified complete public predicted-Shape stage3')
    settings=json.loads((source/'run_config.json').read_text());provenance=dict(settings['source_hashes'])
    for path,sha in provenance.items():
        if file_hash(path)!=sha:raise ValueError(f'Stage3 source changed: {path}')
    feature=Path(settings['args']['feature_run']);phenotype=Path(settings['args']['phenotype_run'])
    config=yaml.safe_load((feature/'run_config.yaml').read_text());nested=Path(config['source_run'])
    samples=CT_Dataset(config['data_dir'],roi_size=64).samples
    if len(samples)!=721 or int(samples.event.sum())!=251:raise ValueError('Require721 patients/251 events')
    ids=(samples.pa_id+'|'+samples.ct_id).tolist()
    shapes=pd.read_csv(phenotype/'ct_phenotypes.csv',dtype={'case_id':str}).set_index('case_id')[shape_groups()['VDS']]
    if not shapes.index.is_unique or set(shapes.index)!=set(ids) or not np.isfinite(shapes).all().all():
        raise ValueError('True Shape identity/values mismatch')
    public=read_feature_cache(feature/'cache/public_ct_features.npz',ids,config['cohort_sha256'],True,provenance)
    bundles={};contexts={};predicted={};references={key:[] for key in ['PA','true','predicted']}
    locations=dict(PA=Path(settings['references']['PA']),true=Path(settings['references']['true_shape']),predicted=source/'predicted_shape')
    for fold in range(5):
        bundle=load_bundle(fold,nested,samples);bundles[fold]=bundle
        contexts[fold]=load_cached_contexts(fold,bundle,feature);predicted[fold]={}
        stages=[(f'inner_{i}',c['train_ids'],c['val_ids']) for i,c in enumerate(contexts[fold])]
        stages.append(('outer',bundle['oof'].index.tolist(),bundle['val'].index.tolist()))
        for stage,train_ids,val_ids in stages:
            folder=source/f'cache/fold_{fold}'/stage
            fitting,held=read_shape_context(folder,train_ids,val_ids,bundle['val'].index.tolist(),provenance)
            predicted[fold][stage]=dict(fit=fitting,held=held)
        # Only descriptive train inference requires the saved final Probe;
        # Cox fitting always uses the crossfit table loaded above.
        path=source/f'cache/fold_{fold}/outer/shape_probe.joblib'
        head=joblib.load(path);provenance[str(path)]=file_hash(path)
        training=bundle['train'].index;validation=bundle['val'].index
        predicted[fold]['train_eval']=pd.DataFrame(head.predict(public.loc[training].to_numpy()),index=training,columns=shapes.columns)
        np.testing.assert_allclose(head.predict(public.loc[validation].to_numpy()),predicted[fold]['outer']['held'],rtol=1e-11,atol=1e-11)
        for key,location in locations.items():
            row,frames=read_reference(location,fold,bundle,provenance)
            references[key].append(dict(metrics=row,frames=frames))
            if key!='PA':
                path=location/f'fold_{fold}/head_parameters.json';provenance[str(path)]=file_hash(path)
    for path in [source/'run_config.json',source/'completed.json',source/'completion_review.json',phenotype/'ct_phenotypes.csv']:
        provenance[str(path)]=file_hash(path)
    print('Verified721 patients,20 saved Shape contexts and all15 reference folds; no Probe/encoder fitting.',flush=True)
    return config,shapes,bundles,contexts,predicted,references,provenance,locations


def compare_models(current,references,predictions,metrics,seed):
    frames=[]
    for fold,frame in enumerate(predictions[current]):
        frame=frame.copy()
        for label,key in references.items():
            frame[label]=predictions[key][fold].set_index('case_id').loc[frame.case_id,'risk_score'].to_numpy()
        frames.append(frame)
    result=paired_comparison(frames,seed=seed,reference_columns={k:k for k in references})
    current_scores=np.array([r['val_cindex'] for r in metrics[current]])
    result['positive_folds']={label:int(np.sum(current_scores>np.array([r['val_cindex'] for r in metrics[key]]))) for label,key in references.items()}
    return result


def write_report(root,summary,conditional,correlations,baseline):
    lines=['# Stage1 Shape ablation','',
        '721 patients / 251 events / original five outer folds. Stage1 only; no neural/Probe fitting or auxiliary supervision.',
        '',f'PA reference Val C-index: {baseline.val_cindex.mean():.4f} ± {baseline.val_cindex.std():.4f}.',
        '', 'V=log(volume/mm3), D=maximum3D diameter(mm), S=sphericity. VD also is leave-S-out; no duplicate model.']
    for source in ['true','predicted']:
        lines.extend(['',f'## {source} Shape','',
            '| Group | Val C-index | AUC36 | AUC60 | ΔC vs PA [nominal95% interval] | Positive folds |',
            '|---|---|---|---|---|---|'])
        for r in summary:
            if r['source']!=source:continue
            lines.append(f'| {r["group"]} | {r["val_cindex_mean"]:.4f} ± {r["val_cindex_std"]:.4f} | '
                f'{r["auc_36m_mean"]:.4f} ± {r["auc_36m_std"]:.4f} | {r["auc_60m_mean"]:.4f} ± {r["auc_60m_std"]:.4f} | '
                f'{r["delta_vs_PA"]:+.4f} [{r["lower_vs_PA"]:+.4f},{r["upper_vs_PA"]:+.4f}] | {r["positive_vs_PA"]}/5 |')
    lines.extend(['','## Conditional additions (current minus reduced model)','',
        '| Source | Added | Existing | ΔC [nominal95% interval] | Positive folds |','|---|---|---|---|---|'])
    for r in conditional:
        lines.append(f'| {r["source"]} | {r["added"]} | {r["conditioning"]} | '
            f'{r["delta_cindex"]:+.4f} [{r["lower"]:+.4f},{r["upper"]:+.4f}] | {r["positive_folds"]}/5 |')
    lines.extend(['','## Descriptive correlations','', '| Source | Pair | Pearson | Spearman |','|---|---|---|---|'])
    for r in correlations:lines.append(f'| {r["source"]} | {r["pair"]} | {r["pearson"]:.4f} | {r["spearman"]:.4f} |')
    lines.extend(['','## Interpretation boundaries','',
        '- True and predicted feature subsets answer different questions: phenotype value versus value remaining after decoding. A null predicted-only result does not establish absence of true-phenotype value.',
        '- Conditional increment is relative to the current scalar pathology risk predictor, not all information in the WSI. A positive result does not establish an independent biological mechanism.',
        '- All subsets refit/reselect their own Cox ridge within the same saved inner contexts. Leave-one-out differences measure entire predictive pipelines; correlated features and retuning prevent interpreting them as unique causal contributions.',
        '- No features or auxiliary targets are selected automatically from outer results. No CT-only, teacher or student auxiliary-supervision experiment is started. This is repeated exploratory analysis, not independent confirmation.',
        '- Intervals bootstrap paired patients within each fixed outer fold2000 times; they are nominal, not adjusted for multiple comparisons, and exclude network/Probe retraining, tuning and prior exploration uncertainty.',
        '- True Shape uses tumor masks. Predicted Shape uses public512 vectors with training-mask phenotype supervision. Existing CT input is mask-localized ROI64; this does not make the entire pipeline mask-free.',
        '- Cox fits PA OOF plus saved Shape OOF. Full-Probe own-training predictions are used only for descriptive train metrics/KM cutoffs, never Cox fitting. Source PA/true-VDS/predicted-VDS patient predictions remain unchanged.',
        '- Full VDS parity requires the same selected ridge, risk error at most1e-8 and identical pair ordering/ties. A preflight true-VDS fold1 refit differed by about3e-9 from solver roundoff at Cox tolerance1e-9; applying the saved old head to current covariates differed by only4e-16.',
        '- Means/sample SD are across five folds. Full train/val metrics,NLL,HR,KM,paired subset/full/matching-true comparisons and per-fold changes accompany this report.'])
    (root/'report.md').write_text('\n'.join(lines)+'\n')


def run(args):
    root=Path(args.results_root)
    if root.exists():raise FileExistsError(f'Refusing to overwrite {root}')
    if np.__version__!='1.26.4':raise RuntimeError('Use the existing isolated phenotype runtime')
    config,shapes,bundles,contexts,predicted,references,provenance,locations=load_ablation_inputs(args)
    root.mkdir(parents=True);groups=shape_groups()
    (root/'run_config.json').write_text(json.dumps(dict(args=vars(args),patients=721,events=251,groups=groups,
        sources=['true','predicted'],conditional_contrasts=conditional_contrasts(),seed=config['seed'],
        ridge_grid=config['ridge_grid'],source_hashes=provenance,neural_fits=0,probe_fits=0,auxiliary_training=False),indent=2))
    metrics={'PA':[r['metrics'] for r in references['PA']]};predictions={'PA':[r['frames']['val'] for r in references['PA']]}
    parity=[]
    for source in ['true','predicted']:
        for group,columns in groups.items():
            key=source+'/'+group;metrics[key]=[];predictions[key]=[]
            for fold,bundle in bundles.items():
                inner=[]
                for i,c in enumerate(contexts[fold]):
                    a=shapes if source=='true' else predicted[fold][f'inner_{i}']['fit']
                    b=shapes if source=='true' else predicted[fold][f'inner_{i}']['held']
                    inner.append(dict(c,train_ct=a.loc[c['train_ids'],columns].to_numpy(),val_ct=b.loc[c['val_ids'],columns].to_numpy()))
                selected=select_increment(inner,dimensions=[None],penalties=config['ridge_grid'])
                oof=bundle['oof'];fit=shapes if source=='true' else predicted[fold]['outer']['fit']
                head=IncrementCox(ridge=selected['ridge']).fit(oof.pathology_risk,fit.loc[oof.index,columns].to_numpy(),oof['dfs.month'],oof['dfs.status'])
                frames={}
                for split in ['train','val']:
                    frame=bundle[split]
                    value=shapes if source=='true' else predicted[fold]['train_eval' if split=='train' else 'outer']
                    if source=='predicted' and split=='val':value=value['held']
                    risk=head.predict(frame.pathology_risk,value.loc[frame.index,columns].to_numpy())
                    frames[split]=frame[['dfs.month','dfs.status']].assign(risk_score=risk).reset_index()
                if group=='VDS':
                    old_choice=json.loads((locations[source]/f'fold_{fold}/head_parameters.json').read_text())['selection']['ridge']
                    if selected['ridge']!=old_choice:raise ValueError('Full-Shape reference ridge mismatch')
                    errors={}
                    for split in ['train','val']:
                        old=references[source][fold]['frames'][split].set_index('case_id').loc[frames[split].case_id]
                        errors[split]=verify_reference_risk(frames[split].risk_score,old.risk_score)
                    parity.append(dict(source=source,fold=fold,ridge=old_choice,max_abs_errors=errors,pair_order_identical=True))
                metadata=dict(source=source,group=group,columns=columns,selection=selected,coefficients=head.model.coef_.tolist(),
                    pathology_penalty=0.,ct_penalty=selected['ridge'],fitting_inputs='true Shape or saved nested OOF Shape; PA OOF',probe_fits=0)
                row,pred=save_result(key,fold,frames,root,metadata,head)
                metrics[key].append(row);predictions[key].append(pred)
    for key,rows in metrics.items():save_cv_results(rows,root/key)
    summary=[];folds=[]
    for source in ['true','predicted']:
        for group in groups:
            key=source+'/'+group;refs=dict(PA='PA',full=source+'/VDS')
            if source=='predicted':refs['matching_true']='true/'+group
            comparison=compare_models(key,refs,predictions,metrics,config['seed'])
            (root/key/'paired_comparison.json').write_text(json.dumps(comparison,indent=2))
            frame=pd.DataFrame(metrics[key]);row=dict(source=source,group=group)
            for column in ['train_cindex','val_cindex','train_loss','val_loss','auc_36m','auc_60m']:
                if not np.isfinite(frame[column]).all():raise ValueError('Nonfinite ablation metric')
                row.update({column+'_'+stat:float(getattr(frame[column],stat)()) for stat in ['mean','std']})
            for label in refs:
                value=comparison['fusion_minus_'+label]
                row.update({f'delta_vs_{label}':value['delta_cindex'],f'lower_vs_{label}':value['lower'],
                            f'upper_vs_{label}':value['upper'],f'positive_vs_{label}':comparison['positive_folds'][label]})
            summary.append(row)
            for fold,r in enumerate(metrics[key]):
                folds.append(dict(source=source,group=group,**r,delta_vs_PA=r['val_cindex']-metrics['PA'][fold]['val_cindex'],
                    delta_vs_full=r['val_cindex']-metrics[source+'/VDS'][fold]['val_cindex']))
    conditional=[]
    for source in ['true','predicted']:
        for larger,smaller,added in conditional_contrasts():
            a,b=source+'/'+larger,source+'/'+smaller
            comp=compare_models(a,{'reduced':b},predictions,metrics,config['seed']);value=comp['fusion_minus_reduced']
            conditional.append(dict(source=source,added=added,conditioning=smaller,larger=larger,
                delta_cindex=value['delta_cindex'],lower=value['lower'],upper=value['upper'],positive_folds=comp['positive_folds']['reduced'],
                delta_auc36=float(np.mean([x['auc_36m']-y['auc_36m'] for x,y in zip(metrics[a],metrics[b])])),
                delta_auc60=float(np.mean([x['auc_60m']-y['auc_60m'] for x,y in zip(metrics[a],metrics[b])]))))
    correlations=[]
    decoded=pd.concat([predicted[i]['outer']['held'] for i in range(5)])
    if not decoded.index.is_unique or set(decoded.index)!=set(shapes.index):raise ValueError('Predicted outer Shape coverage mismatch')
    for source,table in [('true',shapes),('predicted',decoded)]:
        for a,b in [('V','D'),('V','S'),('D','S')]:
            x,y=table[groups[a][0]],table[groups[b][0]]
            correlations.append(dict(source=source,pair=a+'-'+b,pearson=float(x.corr(y)),spearman=float(x.corr(y,method='spearman'))))
    pd.DataFrame(summary).to_csv(root/'summary.csv',index=False)
    pd.DataFrame(folds).to_csv(root/'fold_metrics.csv',index=False)
    pd.DataFrame(conditional).to_csv(root/'conditional_additions.csv',index=False)
    pd.DataFrame(correlations).to_csv(root/'feature_correlations.csv',index=False)
    write_report(root,summary,conditional,correlations,pd.DataFrame(metrics['PA']))
    for path,sha in provenance.items():
        if file_hash(path)!=sha:raise ValueError(f'Source changed during ablation: {path}')
    (root/'completed.json').write_text(json.dumps(dict(patients=721,events=251,groups=list(groups),sources=['true','predicted'],
        model_fold_outputs=70,full_reference_parity=parity,neural_fits=0,probe_fits=0,auxiliary_training=False,
        source_hashes_unchanged=True),indent=2))
    print('Shape stage1 ablation complete; no auxiliary-supervision training started.',flush=True)


if __name__=='__main__':
    experiment=Path(__file__).resolve().parent
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predicted_shape_run',default=str(experiment/'results/predicted_shape_increment'))
    parser.add_argument('--results_root',default=str(experiment/'results/shape_ablation'))
    with threadpool_limits(limits=2):run(parser.parse_args())
