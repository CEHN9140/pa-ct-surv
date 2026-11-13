"""Experiment1: predefined ROI phenotypes conditional on honest pathology risks."""

import argparse
import hashlib
import json
import logging
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))

import joblib
import numpy as np
import pandas as pd
import radiomics
import SimpleITK as sitk
import torch
import yaml
from sklearn.exceptions import ConvergenceWarning
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
from sksurv.util import Surv
from threadpoolctl import threadpool_limits

from ct_phenotype_features import feature_groups,extract_case
from dataset import CT_Dataset
from feature_increment import load_bundle
from increment_cox import IncrementCox,select_increment
from loss import cox_loss
from risk_calibration import paired_comparison
from utils import save_cv_results
from visualization.survival_plots import analyze_fold


def fit_ct_head(x,time,event,penalty):
    head=Pipeline([('scale',StandardScaler()),('cox',CoxPHSurvivalAnalysis(alpha=penalty,
                   ties='breslow',n_iter=200,tol=1e-9))])
    with warnings.catch_warnings():
        warnings.simplefilter('error',ConvergenceWarning)
        head.fit(np.asarray(x,dtype=np.float64),Surv.from_arrays(np.asarray(event).astype(bool),np.asarray(time,dtype=float)))
    if not np.isfinite(head.named_steps['cox'].coef_).all():raise ValueError('Nonfinite CT-only coefficients')
    return head


def select_ct_penalty(contexts,penalties):
    for context in contexts:
        if set(context['train_ids'])&set(context['val_ids']):raise ValueError('CT-only tuning patient overlap')
    candidates=[]
    for penalty in penalties:
        scores=[];failures=[]
        for number,c in enumerate(contexts):
            try:
                model=fit_ct_head(c['train_ct'],c['train_time'],c['train_event'],penalty)
                risk=model.predict(c['val_ct'])
                scores.append(float(concordance_index_censored(c['val_event'].astype(bool),c['val_time'],risk)[0]))
            except (ValueError,ArithmeticError,ConvergenceWarning,np.linalg.LinAlgError) as exc:
                failures.append(f'inner{number}: {type(exc).__name__}: {exc}')
        candidates.append(dict(ridge=float(penalty),scores=scores,failures=failures,
                               mean_cindex=None if failures else float(np.mean(scores))))
    valid=[x for x in candidates if x['mean_cindex'] is not None]
    if not valid:raise RuntimeError(f'No valid CT-only inner candidates: {candidates}')
    best=max(valid,key=lambda x:(x['mean_cindex'],x['ridge']))
    return dict(ridge=best['ridge'],inner_mean_cindex=best['mean_cindex'],candidates=candidates,
                selection='mean inner C-index; exact tie favors stronger ridge')


def load_cached_contexts(fold,bundle,feature_root):
    oof=bundle['oof'];contexts=[];outer=set(bundle['val'].index)
    for inner in sorted(oof.inner_fold.astype(int).unique()):
        train=oof.index[~oof.inner_fold.eq(inner)].tolist();held=oof.index[oof.inner_fold.eq(inner)].tolist()
        root=Path(feature_root)/f'cache/fold_{fold}/inner_{inner}'
        cache=root/'pathology_train_oof.csv'
        if not cache.is_file():raise FileNotFoundError(f'Honest PA cache required; neural retraining disabled: {cache}')
        pa=pd.read_csv(cache,dtype={'case_id':str}).set_index('case_id')
        roles=pd.read_csv(root/'nested_pa_roles.csv',dtype={'case_id':str})
        if not pa.index.is_unique or set(pa.index)!=set(train) or set(roles.case_id)&(outer|set(held)):
            raise ValueError('Cached PA calibration predictions/roles leak or differ from inner training')
        all_oof=[]
        for sub,rows in roles.groupby('subfold'):
            groups={role:set(frame.case_id) for role,frame in rows.groupby('role')}
            if (set(groups)!={'fit','stop','oof'} or rows.case_id.duplicated().any()
                    or set(rows.case_id)!=set(train)):
                raise ValueError('Cached PA fit/stop/OOF roles not disjoint within training')
            all_oof.extend(groups['oof'])
        if len(all_oof)!=len(train) or set(all_oof)!=set(train):raise ValueError('PA sub-OOF coverage differs')
        p=pa.loc[train,'pathology_risk'].to_numpy()
        if not np.isfinite(p).all():raise ValueError('Nonfinite cached PA OOF risk')
        contexts.append(dict(train_ids=train,val_ids=held,train_pa=p,
            val_pa=oof.loc[held,'pathology_risk'].to_numpy(),
            train_time=oof.loc[train,'dfs.month'].to_numpy(),train_event=oof.loc[train,'dfs.status'].to_numpy(),
            val_time=oof.loc[held,'dfs.month'].to_numpy(),val_event=oof.loc[held,'dfs.status'].to_numpy()))
    return contexts


def save_result(name,fold,frames,root,metadata,head=None):
    target=Path(root)/name/f'fold_{fold}';target.mkdir(parents=True)
    if head is not None:joblib.dump(head,target/'cox_head.joblib')
    (target/'head_parameters.json').write_text(json.dumps(metadata,indent=2))
    frames['train'].to_csv(target/'train_predictions.csv',index=False)
    metrics=analyze_fold(frames['train'],frames['val'],target)
    for split,frame in frames.items():
        metrics[split+'_loss']=float(cox_loss(*[torch.tensor(frame[key].to_numpy(),dtype=torch.float64)
            for key in ['risk_score','dfs.month','dfs.status']]))
    print(f'{name} fold{fold} | Train={metrics["train_cindex"]:.4f} | Val={metrics["val_cindex"]:.4f} | '
          f'Val Cox NLL={metrics["val_loss"]:.4f}',flush=True)
    return dict(fold=fold,**metrics),frames['val']


def write_report(root,table,baseline,coverage):
    lines=['# Predefined CT phenotype increment: exploratory diagnostic','',
        '721 patients, 251 events, unchanged five outer folds. No neural models were retrained; no MedicalNet probes were run.',
        '',f'Pathology reference C-index: {baseline.val_cindex.mean():.4f} ± {baseline.val_cindex.std():.4f}.',
        '', '| Phenotype | CT-only C-index | PA+CT C-index | Δ vs PA (95% paired interval) | Positive folds | AUC36 | AUC60 |',
        '|---|---|---|---|---|---|---|']
    for row in table:
        lines.append(f'| {row["phenotype"]} | {row["ct_only_cindex_mean"]:.4f} ± {row["ct_only_cindex_std"]:.4f} | '
            f'{row["pa_ct_cindex_mean"]:.4f} ± {row["pa_ct_cindex_std"]:.4f} | '
            f'{row["delta_cindex"]:+.4f} [{row["delta_ci_lower"]:+.4f}, {row["delta_ci_upper"]:+.4f}] | '
            f'{row["positive_folds"]}/5 | {row["auc36_mean"]:.4f} ± {row["auc36_std"]:.4f} | '
            f'{row["auc60_mean"]:.4f} ± {row["auc60_std"]:.4f} |')
    combo=next(row for row in table if row['phenotype']=='Combined')
    lines.extend(['',f'Combined minus Shape: {combo["delta_vs_shape"]:+.4f} '
        f'[{combo["delta_vs_shape_lower"]:+.4f}, {combo["delta_vs_shape_upper"]:+.4f}].',
        '',f'Peritumoral 5 mm coverage: {int(coverage.peri5_coverage.ge(1-1e-12).sum())}/721 complete; '
        f'minimum {coverage.peri5_coverage.min():.4%}. All patients retained; coverage is QC only.',
        '', '## Interpretation limits','',
        '- Means and sample SD are across five folds. Paired intervals bootstrap patients within each fixed validation fold, conditional on fitted models; they exclude retraining, tuning and prior exploration uncertainty.',
        '- All groups depend on tumor masks. Shape reuses native-mask geometry; intensity and texture use only the saved ROI64 and valid observed shell, excluding padding. This is mask-assisted information, not a demonstrated capability of an unmasked encoder.',
        '- Recovered integer intensities are clipped-input proxies. Upstream intensity calibration is unknown; inverse normalization cannot recover the original raw HU.',
        '- Combined versus Shape tests the added feature bundle after retaining Shape. Other groups versus Shape are replacement comparisons, not independent effects adjusted for Shape.',
        '- Conditional value is relative to the current pathology risk predictor, not all WSI information. Multiple predefined comparisons remain exploratory; a null result does not prove absence of other CT information.',
        '- Validation Cox partial NLL is event-normalized Breslow loss, not calibrated survival probability error. Full NLL, AUC, per-fold metrics, ridge selection and pairwise comparisons are saved in the accompanying tables/JSON.',
        '- The formal method remains a jointly trained pathology–CT teacher and CT-only student. No downstream unimodal initialization is introduced.'])
    (Path(root)/'report.md').write_text('\n'.join(lines)+'\n')


def run(args):
    if np.__version__!='1.26.4':
        raise RuntimeError('This diagnostic requires its isolated NumPy1.26.4 runtime; use run_ct_phenotype_increment.sh. UNI is unchanged.')
    root=Path(args.results_root);feature_root=Path(args.feature_run);source=Path(args.source_run)
    if root.exists():raise FileExistsError(f'Refusing to overwrite {root}')
    if not(feature_root/'completed.json').is_file() or not(source/'completed.json').is_file():
        raise ValueError('Require complete feature-increment and nested-risk sources')
    config=yaml.safe_load((feature_root/'run_config.yaml').read_text())
    if Path(config['source_run']).resolve()!=source.resolve():raise ValueError('Nested-risk source differs')
    audit=Path(args.input_audit)
    audited=json.loads((audit/'completed.json').read_text())
    if not audited['passed'] or audited['patients']!=721:raise ValueError('Require passed721-case input audit')
    dataset=CT_Dataset(config['data_dir'],roi_size=64);samples=dataset.samples
    if len(samples)!=721 or int(samples.event.sum())!=251:raise ValueError('Require721 patients/251 events')
    def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    provenance={str(path):digest(path) for path in [feature_root/'run_config.yaml',source/'run_config.yaml',
        Path(config['data_dir'])/'all_label_roi64.csv',feature_root/'cache/shape_features.csv',
        Path(args.preprocess_log),audit/'run_config.json']}
    if digest(Path(config['data_dir'])/'all_label_roi64.csv')!=config['cohort_sha256']:
        raise ValueError('Source cohort changed')
    if digest(source/'run_config.yaml')!=config['source_config_sha256']:raise ValueError('Source nested config changed')
    audit_config=json.loads((audit/'run_config.json').read_text())
    if audit_config['sha256']['cohort']!=config['cohort_sha256'] or audit_config['sha256']['preprocess_log']!=digest(args.preprocess_log):
        raise ValueError('Input audit provenance changed')
    bundles={};contexts={}
    for fold in range(5):
        bundles[fold]=load_bundle(fold,source,samples)
        contexts[fold]=load_cached_contexts(fold,bundles[fold],feature_root)
        for inner in range(3):
            for filename in ['pathology_train_oof.csv','nested_pa_roles.csv']:
                path=feature_root/f'cache/fold_{fold}/inner_{inner}'/filename
                provenance[str(path)]=digest(path)
    root.mkdir(parents=True)
    groups=feature_groups()
    settings=dict(experiment='CT phenotypes conditional increment; experiment1 only',args=vars(args),
        patients=721,events=251,groups=groups,pyradiomics_version=radiomics.__version__,
        numpy_version=np.__version__,runtime_python=sys.executable,ridge_grid=config['ridge_grid'],
        extraction=dict(image_type='Original',binWidth=25,distances=[1],symmetricalGLCM=True,weightingNorm=None,
            force2D=False,normalize=False,resampling=None,intensity='round(1400*saved_ROI-1000); clipped proxy, upstream unknown'),
        coverage='observed valid5mm outside-mask shell / full geometric shell; QC not predictor',
        source_hashes=provenance,neural_fits=0,medicalnet_probes=False,feature_selection='predefined only; no outcome filtering')
    (root/'run_config.json').write_text(json.dumps(settings,indent=2))
    logs=pd.read_csv(args.preprocess_log,dtype={'ct_id':str}).set_index('ct_id')
    audit_rows=pd.read_csv(audit/'ct_input_cases.csv',dtype={'ct_id':str}).set_index('ct_id')
    jobs=[(row,logs.loc[row['ct_id']].to_dict(),audit_rows.loc[row['ct_id']].to_dict()) for row in samples.to_dict('records')]
    outputs=[];started=time.monotonic()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for number,result in enumerate(pool.map(extract_case,jobs),1):
            outputs.append(result)
            if number%25==0 or number==721:print(f'Phenotype extraction {number}/721 | {time.monotonic()-started:.1f}s',flush=True)
    failed=[x for x in outputs if x['error'] is not None]
    (root/'phenotype_failed_cases.json').write_text(json.dumps(failed,indent=2))
    if failed:raise RuntimeError(f'{len(failed)} phenotype extractions failed; no patients silently dropped')
    features=pd.DataFrame([dict(case_id=x['case_id'],**x['features']) for x in outputs]).set_index('case_id')
    coverage=pd.DataFrame([dict(case_id=x['case_id'],**x['coverage']) for x in outputs]).set_index('case_id')
    shapes=pd.read_csv(feature_root/'cache/shape_features.csv',dtype={'case_id':str}).set_index('case_id')
    if not features.index.is_unique or set(features.index)!=set(shapes.index):raise ValueError('Phenotype/Shape cohort mismatch')
    features['log_volume_mm3']=np.log(shapes.loc[features.index,'volume_mm3'])
    features['diameter_mm']=shapes.loc[features.index,'diameter_mm'];features['sphericity']=shapes.loc[features.index,'sphericity']
    features=features[groups['Combined']]
    if features.shape!=(721,27) or not np.isfinite(features).all().all():raise ValueError('Incomplete predefined phenotype table')
    features.to_csv(root/'ct_phenotypes.csv');coverage.to_csv(root/'phenotype_coverage.csv')
    metrics={};predictions={};shape_checks=[]
    for fold in range(5):
        bundle=bundles[fold];oof=bundle['oof'];training=bundle['train'];validation=bundle['val']
        m0={split:pd.read_csv(feature_root/f'M0/fold_{fold}/{split}_predictions.csv',dtype={'case_id':str}) for split in ['train','val']}
        for split,frame in m0.items():
            truth=bundle[split].loc[frame.case_id]
            if set(frame.case_id)!=set(bundle[split].index):raise ValueError('M0 reference cohort mismatch')
            np.testing.assert_allclose(frame['dfs.month'],truth['dfs.month']);np.testing.assert_array_equal(frame['dfs.status'],truth['dfs.status'])
        row,pred=save_result('M0',fold,m0,root,dict(source='unchanged completed M0 calibrated pathology risk',source_run=str(feature_root)))
        metrics.setdefault('M0',[]).append(row);predictions.setdefault('M0',[]).append(pred)
        for group,columns in groups.items():
            inner=[dict(c,train_ct=features.loc[c['train_ids'],columns].to_numpy(),
                         val_ct=features.loc[c['val_ids'],columns].to_numpy()) for c in contexts[fold]]
            for conditional in [False,True]:
                if conditional:
                    selected=select_increment(inner,dimensions=[None],penalties=config['ridge_grid'])
                    head=IncrementCox(ridge=selected['ridge']).fit(oof.pathology_risk,features.loc[oof.index,columns].to_numpy(),oof['dfs.month'],oof['dfs.status'])
                    coefficients=head.model.coef_.tolist()
                else:
                    selected=select_ct_penalty(inner,config['ridge_grid'])
                    head=fit_ct_head(features.loc[oof.index,columns].to_numpy(),oof['dfs.month'],oof['dfs.status'],selected['ridge'])
                    coefficients=head.named_steps['cox'].coef_.tolist()
                name=('pa_plus_ct/' if conditional else 'ct_only/')+group;frames={}
                for split,frame in [('train',training),('val',validation)]:
                    x=features.loc[frame.index,columns].to_numpy()
                    risk=head.predict(frame.pathology_risk.to_numpy(),x) if conditional else head.predict(x)
                    frames[split]=frame[['dfs.month','dfs.status']].assign(risk_score=risk).reset_index()
                if conditional and group=='Shape':
                    for split in ['train','val']:
                        old=pd.read_csv(feature_root/f'M4_shape/fold_{fold}/{split}_predictions.csv').set_index('case_id').loc[frames[split].case_id]
                        np.testing.assert_allclose(frames[split].risk_score,old.risk_score,rtol=1e-9,atol=1e-10)
                    shape_checks.append(fold)
                info=dict(group=group,conditional=conditional,columns=columns,selection=selected,
                    coefficients=coefficients,n_fit=len(oof),pathology_penalty=0. if conditional else None,
                    ct_penalty=selected['ridge'],transform_fit='outer training only; hyperparameters via honest inner calibration')
                row,pred=save_result(name,fold,frames,root,info,head)
                metrics.setdefault(name,[]).append(row);predictions.setdefault(name,[]).append(pred)
    table=[]
    for name,rows in metrics.items():save_cv_results(rows,root/name)
    for group in groups:
        name='pa_plus_ct/'+group;combined=[]
        for fold,frame in enumerate(predictions[name]):
            frame=frame.copy()
            for key,reference in [('baseline_pa','M0'),('baseline_shape','pa_plus_ct/Shape')]:
                frame[key]=predictions[reference][fold].set_index('case_id').loc[frame.case_id,'risk_score'].to_numpy()
            combined.append(frame)
        paired=paired_comparison(combined,seed=config['seed'],reference_columns={'M0':'baseline_pa','Shape':'baseline_shape'})
        cond=pd.DataFrame(metrics[name]);ct=pd.DataFrame(metrics['ct_only/'+group]);base=pd.DataFrame(metrics['M0'])
        positive=int((cond.val_cindex>base.val_cindex).sum());interval=paired['fusion_minus_M0']
        paired.update(positive_folds_vs_M0=positive,positive_folds_vs_Shape=int((cond.val_cindex>pd.DataFrame(metrics['pa_plus_ct/Shape']).val_cindex).sum()),
            exploratory_signal_vs_M0=interval['delta_cindex']>0 and interval['lower']>0 and positive>=4,
            mean_delta_val_cox_nll=float((cond.val_loss-base.val_loss).mean()))
        (root/name/'comparison_to_references.json').write_text(json.dumps(paired,indent=2))
        table.append(dict(phenotype=group,ct_only_cindex_mean=float(ct.val_cindex.mean()),ct_only_cindex_std=float(ct.val_cindex.std()),
            pa_ct_cindex_mean=float(cond.val_cindex.mean()),pa_ct_cindex_std=float(cond.val_cindex.std()),
            delta_cindex=interval['delta_cindex'],delta_ci_lower=interval['lower'],delta_ci_upper=interval['upper'],positive_folds=positive,
            val_cox_nll_mean=float(cond.val_loss.mean()),val_cox_nll_std=float(cond.val_loss.std()),
            delta_val_cox_nll=paired['mean_delta_val_cox_nll'],
            auc36_mean=float(cond.auc_36m.mean()),auc36_std=float(cond.auc_36m.std()),
            auc60_mean=float(cond.auc_60m.mean()),auc60_std=float(cond.auc_60m.std()),
            delta_vs_shape=paired['fusion_minus_Shape']['delta_cindex'],
            delta_vs_shape_lower=paired['fusion_minus_Shape']['lower'],delta_vs_shape_upper=paired['fusion_minus_Shape']['upper']))
    pd.DataFrame(table).to_csv(root/'phenotype_summary.csv',index=False)
    write_report(root,table,pd.DataFrame(metrics['M0']),coverage)
    for path,expected in provenance.items():
        if digest(path)!=expected:raise ValueError(f'Source changed during diagnostic: {path}')
    (root/'completed.json').write_text(json.dumps(dict(patients=721,events=251,features=27,groups=list(groups),
        folds=list(range(5)),source_M0_unchanged=True,source_Shape_predictions_reproduced=shape_checks,
        neural_fits=0,medicalnet_probes=False,source_hashes_unchanged=True),indent=2))
    print(pd.DataFrame(table).to_string(index=False),flush=True)
    print('Predefined CT phenotype experiment1 complete; no experiment2 or teacher training started.',flush=True)


if __name__=='__main__':
    experiment=Path(__file__).resolve().parent
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source_run',default=str(experiment/'results/nested_risk_increment'))
    parser.add_argument('--feature_run',default=str(experiment/'results/feature_increment'))
    parser.add_argument('--input_audit',default=str(experiment/'results/ct_input_audit'))
    parser.add_argument('--preprocess_log',default='/home/gly001/cqj/data/lung_cancer/ct/processed_ct_64/preprocess_log.csv')
    parser.add_argument('--results_root',default=str(experiment/'results/ct_phenotype_increment'))
    parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args()
    if args.workers<1:parser.error('workers must be positive')
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1);torch.set_num_threads(2)
    radiomics.setVerbosity(logging.WARNING)
    with threadpool_limits(limits=2):run(args)
