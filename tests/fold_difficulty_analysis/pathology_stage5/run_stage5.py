"""Stage5: five-fold diagnostic MeanPool + nested Ridge-Cox on raw UNI."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import joblib
import h5py
import numpy as np
import pandas as pd
import scipy
import sklearn
import sksurv
from scipy.stats import spearmanr
from sksurv.metrics import concordance_index_censored,cumulative_dynamic_auc
from sksurv.util import Surv

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(Path(__file__).parent.parent/'pathology_stage4'))

from ridge import pool_feature,fit_outer
from pairs import pair_changes,summarize_changes
from diagnose_late import read_predictions


def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(8*1024*1024),b''): digest.update(block)
    return digest.hexdigest()


def evaluate_predictions(train,val,train_risk,val_risk):
    row={}
    for name,part,risk in [('train',train,train_risk),('val',val,val_risk)]:
        row[name+'_cindex']=float(concordance_index_censored(part.event.astype(bool),part.time,risk)[0])
    row['gap']=row['train_cindex']-row['val_cindex']
    y_train=Surv.from_arrays(train.event.astype(bool),train.time)
    y_val=Surv.from_arrays(val.event.astype(bool),val.time)
    for t in (36.,60.):
        key=f'auc_{t:g}m'
        row[key]=np.nan
        row[key+'_status']='ok'
        try:
            if not ((val.time<=t)&val.event.eq(1)).any() or not (val.time>t).any():
                raise ValueError('No cases or controls at this horizon')
            auc=cumulative_dynamic_auc(y_train,y_val,val_risk,[t])[0][0]
            if not np.isfinite(auc): raise ValueError('Nonfinite IPCW AUC')
            row[key]=float(auc)
        except ValueError as exc:
            row[key+'_status']=str(exc)
    return row


def compare_predictions(patients,ridge_risk,reference_risk):
    pairs,errors,summary=pair_changes(patients.time.to_numpy(),patients.event.to_numpy(),
                                      np.asarray(reference_risk),np.asarray(ridge_risk))
    shared=(pairs.best_credit==0)&(pairs.last_credit==0)
    union=(pairs.best_credit==0)|(pairs.last_credit==0)
    rho=spearmanr(reference_risk,ridge_risk).statistic if np.std(reference_risk)>0 and np.std(ridge_risk)>0 else np.nan
    row=dict(reference_cindex=summary['best_cindex'],ridge_cindex=summary['last_cindex'],
             delta_cindex=summary['delta_cindex'],spearman=float(rho),pairs=len(pairs),
             both_wrong=int(shared.sum()),wrong_union=int(union.sum()),
             wrong_jaccard=float(shared.sum()/union.sum()) if union.any() else np.nan,
             reference_wrong=int((pairs.best_credit==0).sum()),ridge_wrong=int((pairs.last_credit==0).sum()),
             reference_wrong_ridge_correct=summary['corrected_full'],
             reference_correct_ridge_wrong=summary['broken_full'],
             gained_credit=summary['gained_credit'],lost_credit=summary['lost_credit'],
             both_tied=int(((pairs.best_credit==.5)&(pairs.last_credit==.5)).sum()))
    strata=[]
    for (kind,bin_name),part in pairs.groupby(['pair_type','event_time_bin']):
        subshared=(part.best_credit==0)&(part.last_credit==0)
        subunion=(part.best_credit==0)|(part.last_credit==0)
        change=summarize_changes(part)
        strata.append(dict(pair_type=kind,event_time_bin=bin_name,pairs=len(part),
                            reference_cindex=change['best_cindex'],ridge_cindex=change['last_cindex'],
                            delta_cindex_contribution=change['net_credit']/len(pairs),
                            both_wrong=int(subshared.sum()),wrong_union=int(subunion.sum()),
                            wrong_jaccard=float(subshared.sum()/subunion.sum()) if subunion.any() else np.nan))
    for name,mask in [('shared_wrong_pairs',shared),('reference_only_wrong_pairs',(pairs.best_credit==0)&(pairs.last_credit!=0)),
                      ('ridge_only_wrong_pairs',(pairs.last_credit==0)&(pairs.best_credit!=0)),
                      ('reference_tied_pairs',pairs.best_credit==.5),('ridge_tied_pairs',pairs.last_credit==.5)]:
        errors[name]=np.bincount(np.r_[pairs.i,pairs.j],weights=np.r_[mask,mask],minlength=len(patients)).astype(int)
    errors=errors.rename(columns={'error_rate_best':'reference_error_rate','error_rate_last':'ridge_error_rate',
                                  'error_credit_best':'reference_error_credit','error_credit_last':'ridge_error_credit'})
    errors=errors.assign(case_id=patients.pa_id.to_numpy(),event=patients.event.to_numpy(),time=patients.time.to_numpy(),
                         reference_risk=reference_risk,ridge_risk=ridge_risk)
    return row,pd.DataFrame(strata),errors


def run(csv,output,reference,seed=123):
    output=Path(output)
    if output.exists() and any(output.iterdir()): raise FileExistsError('Use a fresh --output; do not overwrite earlier results')
    private=output/'private';private.mkdir(parents=True,exist_ok=True)
    status=dict(status='running',pid=os.getpid(),stage='mean_pooling',seed=seed)
    (output/'status.json').write_text(json.dumps(status,indent=2))
    sources={}
    code_paths=[Path(__file__),Path(__file__).with_name('ridge.py'),Path(__file__).with_name('report_stage5.py'),
                Path(__file__).parent.parent/'pathology_stage4/pairs.py']
    code_hashes={str(p.relative_to(ROOT)):sha256(p) for p in code_paths}
    def track(path):
        path=Path(path).resolve()
        value=sha256(path)
        if str(path) in sources and sources[str(path)]!=value: raise RuntimeError('Source changed during read')
        sources[str(path)]=value
        return path
    try:
        csv=track(csv)
        original=pd.read_csv(csv,dtype={'pa_id':str,'ct_id':str})
        if len(original)!=721 or original.event.sum()!=251 or original.pa_id.duplicated().any() or set(original.split)!=set(range(5)):
            raise ValueError('Expected unchanged 721-patient/251-event five-fold cohort')
        if not original.event.isin([0,1]).all() or not np.isfinite(original.time).all() or (original.time<0).any():
            raise ValueError('Invalid labels')
        # Match the previously used time representation; Cox calculations remain float64.
        cohort=original.copy();cohort['time']=cohort.time.to_numpy(np.float32).astype(float)
        means,counts,features=[],[],{}
        print('Stage5: all raw UNI1024 patches -> MeanPool -> nested StandardScaler/Ridge-Cox',flush=True)
        for i,path in enumerate(cohort.pa_path):
            before=sha256(path)
            mean,count=pool_feature(path)
            means.append(mean);counts.append(count);features[path]=before
            if (i+1)%100==0 or i+1==len(cohort): print(f'Mean pooled {i+1}/{len(cohort)} full WSI',flush=True)
        x=np.stack(means)
        np.savez_compressed(private/'patient_features.npz',features=x,case_ids=cohort.pa_id.to_numpy(dtype=str),patch_counts=counts)
        (private/'feature_provenance.json').write_text(json.dumps(features,indent=2))
        reference=Path(reference)
        reference_metrics=pd.read_csv(track(reference/'metrics.csv'))
        import yaml
        ref_config=yaml.safe_load(track(reference/'run_config.yaml').read_text())
        if ref_config['cohort_sha256']!=sources[str(csv)] or ref_config['feature_encoder']!='uni_v1' or ref_config['feature_dim']!=1024 \
                or ref_config['pa_model']!='abmil' or ref_config['max_updates']!=180:
            raise ValueError('Historical ABMIL source does not match raw UNI cohort/features')
        rows,cv_rows,comparisons,strata,profiles,oof,fold_metadata=[],[],[],[],[],[],[]
        for fold in range(5):
            train=np.flatnonzero(cohort.split.ne(fold));val=np.flatnonzero(cohort.split.eq(fold))
            fit_patients,held_patients=cohort.iloc[train],cohort.iloc[val]
            y=Surv.from_arrays(fit_patients.event.astype(bool),fit_patients.time)
            status.update(stage='ridge_fitting',active_fold=fold)
            (output/'status.json').write_text(json.dumps(status,indent=2))
            print(f'Fold {fold}: Train={len(train)}, events={int(fit_patients.event.sum())}; Val={len(val)} | 18 inner fits + 1 outer fit',flush=True)
            start=time.monotonic()
            model,alpha,cv,roles=fit_outer(x[train],y,seed)
            cv_rows.extend(dict(fold=fold,**r) for r in cv)
            saved=private/f'fold_{fold}';saved.mkdir()
            joblib.dump(model,saved/'model.joblib')
            pd.DataFrame(dict(case_id=fit_patients.pa_id.to_numpy(),inner_fold=roles)).to_csv(saved/'inner_roles.csv',index=False)
            a,b=model.predict(x[train]),model.predict(x[val])
            reloaded=joblib.load(saved/'model.joblib')
            if not np.array_equal(reloaded.predict(x[val]),b): raise RuntimeError('Model reload changes predictions')
            metrics=evaluate_predictions(fit_patients,held_patients,a,b)
            row=dict(fold=fold,alpha=alpha,n_train=len(train),n_val=len(val),n_event_train=int(y['event'].sum()),**metrics)
            rows.append(row)
            for part,patients,risk in [('train',fit_patients,a),('val',held_patients,b)]:
                predictions=pd.DataFrame({'case_id':patients.pa_id.to_numpy(),'dfs.month':patients.time.to_numpy(),
                                           'dfs.status':patients.event.to_numpy(),'risk_score':risk})
                predictions.to_csv(saved/f'{part}_predictions.csv',index=False)
                if part=='val': oof.append(predictions.assign(fold=fold))
            baseline=reference_metrics.loc[reference_metrics.size_label.eq('full')&reference_metrics.fold.eq(fold)&reference_metrics['repeat'].eq(0)]
            if len(baseline)!=1: raise ValueError('Missing unique full-pool ABMIL reference')
            baseline=baseline.iloc[0]
            for phase,directory in [('best',reference/'full_repeat0'/f'fold_{fold}'),('last',reference/'full_repeat0'/f'fold_{fold}'/'last')]:
                for part,patients,risk in [('train',fit_patients,a),('val',held_patients,b)]:
                    historical=read_predictions(track(directory/f'{part}_predictions.csv'),patients)
                    r=historical.risk_score.to_numpy(float)
                    comp,sub,patient_errors=compare_predictions(patients,risk,r)
                    field=('last_' if phase=='last' else '')+part+'_cindex'
                    if not np.isclose(comp['reference_cindex'],baseline[field],atol=1e-12,rtol=0):
                        raise RuntimeError('Reference predictions do not reproduce historical C-index')
                    context=dict(fold=fold,split=part,reference=phase)
                    comparisons.append(dict(**context,**comp))
                    strata.append(sub.assign(**context))
                    profiles.append(patient_errors.assign(**context))
            pd.DataFrame(rows).to_csv(output/'metrics.csv',index=False)
            pd.DataFrame(cv_rows).to_csv(output/'inner_cv.csv',index=False)
            fold_metadata.append(dict(fold=fold,alpha=alpha,elapsed_seconds=time.monotonic()-start,
                                      coefficient_norm=float(np.linalg.norm(model.named_steps['cox'].coef_)),
                                      scaler_fit_patients=len(train),model_roundtrip_exact=True))
            status.update(stage='ridge_fitting',completed_folds=fold+1)
            (output/'status.json').write_text(json.dumps(status,indent=2))
            print(f"Fold {fold} completed | alpha={alpha:g} | Train C-index={metrics['train_cindex']:.6f} | "
                  f"Val C-index={metrics['val_cindex']:.6f} | Gap={metrics['gap']:.6f} | "
                  f"AUC36={metrics['auc_36m']:.6f} | AUC60={metrics['auc_60m']:.6f} | {time.monotonic()-start:.1f}s",flush=True)
        oof=pd.concat(oof,ignore_index=True)
        if len(oof)!=721 or oof.case_id.duplicated().any() or set(oof.case_id)!=set(cohort.pa_id):
            raise RuntimeError('Invalid OOF coverage')
        oof.to_csv(private/'oof_predictions.csv',index=False)
        pd.concat(profiles,ignore_index=True).to_csv(private/'patient_error_comparisons.csv',index=False)
        comparisons=pd.DataFrame(comparisons);strata=pd.concat(strata,ignore_index=True)
        comparisons.to_csv(output/'prediction_comparison.csv',index=False)
        strata.to_csv(output/'error_overlap_strata.csv',index=False)
        for path,value in features.items():
            if sha256(path)!=value: raise RuntimeError('An original feature file changed')
        if any(sha256(path)!=value for path,value in sources.items()): raise RuntimeError('A source CSV changed')
        from report_stage5 import report_and_plots
        cv_frame=pd.DataFrame(cv_rows)
        report_and_plots(output,pd.DataFrame(rows),comparisons,strata,reference_metrics,cv_frame)
        if any(sha256(ROOT/path)!=value for path,value in code_hashes.items()):
            raise RuntimeError('Experiment code changed during fitting')
        validation=dict(status='completed',n_patients=721,n_events=251,n_patches=int(sum(counts)),
                        feature_dim=1024,feature_matrix_sha256=hashlib.sha256(x.tobytes()).hexdigest(),
                        feature_source_bundle_sha256=hashlib.sha256(json.dumps(features,sort_keys=True).encode()).hexdigest(),
                        source_hashes_unchanged=True,source_sha256=sources,inner_fit_attempts=len(cv_rows),
                        inner_converged_fits=int(cv_frame.converged.sum()),inner_failed_fits=int((~cv_frame.converged).sum()),
                        final_outer_fits=5,numerical_protocol_amendment='After first FIT-only alpha0.1 overflow, invalid candidates require all3 inner fits; no OuterVal inspected, grid unchanged',
                        seed=seed,alphas=[.1,1,10,100,1000,10000],n_iter=1000,tol=1e-9,ties='breslow',
                        software=dict(numpy=np.__version__,scikit_survival=sksurv.__version__,
                                      scikit_learn=sklearn.__version__,scipy=scipy.__version__,h5py=h5py.__version__),
                        folds=fold_metadata,neural_training_started=False,
                        code_sha256=code_hashes)
        (output/'validation.json').write_text(json.dumps(validation,indent=2))
        status.update(status='completed',stage='finished')
        print('Stage5 completed five outer folds and 90 inner fit attempts; report and plots saved.',flush=True)
    except Exception as exc:
        status.update(status='failed',error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        (output/'status.json').write_text(json.dumps(status,indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv',type=Path,default=ROOT/'data/trident/features_uni_v1/all_label_roi64.csv')
    parser.add_argument('--output',type=Path,default=Path(__file__).parent/'results')
    parser.add_argument('--reference',type=Path,default=ROOT/'tests/10_pathology_learning_curve/results/uni_abmil_cox_dropout0.25_updates180_eval9_coxbs64_lr0.0001_wd0.0005_seed123')
    parser.add_argument('--seed',type=int,default=123)
    args=parser.parse_args()
    run(args.csv,args.output,args.reference,args.seed)


if __name__=='__main__': main()
