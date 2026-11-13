import sys,json,warnings,hashlib
from pathlib import Path
sys.path.insert(0,'/home/gly001/cqj/pa_ct_surv/tests/15_pathology_query_ct_tokens')
sys.path.insert(0,'/home/gly001/cqj/pa_ct_surv')
import numpy as np
import pandas as pd
import joblib,yaml
from threadpoolctl import threadpool_limits
from shape_increment import volume_probe
from dataset import CT_Dataset
from feature_increment import load_bundle,file_hash
from visualization.survival_plots import fold_metrics

root=Path('/home/gly001/cqj/pa_ct_surv/tests/15_pathology_query_ct_tokens/results/feature_increment')
config=yaml.safe_load((root/'run_config.yaml').read_text())
source=Path(config['source_run'])
destination=root/'volume_probe_numeric_review'
destination.mkdir(exist_ok=True)
samples=CT_Dataset(config['data_dir'],roi_size=64).samples
assert file_hash(Path(config['data_dir'])/'all_label_roi64.csv')==config['cohort_sha256']
assert file_hash(source/'run_config.yaml')==config['source_config_sha256']
shapes=pd.read_csv(root/'cache/shape_features.csv',dtype={'case_id':str}).set_index('case_id')
assert len(shapes)==721 and shapes.index.is_unique

def features(path):
    with np.load(path,allow_pickle=False) as cache:
        f=pd.DataFrame(cache['features'],index=cache['case_id'])
    assert f.shape[1]==512 and f.index.is_unique and np.isfinite(f).all().all()
    metadata=json.loads(path.with_suffix('.json').read_text())
    assert metadata['cohort_sha256']==config['cohort_sha256']
    assert file_hash(metadata['checkpoint'])==metadata['checkpoint_sha256']
    return f

public=features(root/'cache/public_ct_features.npz')
models=json.loads((root/'completed.json').read_text())['models']
audit=[]; probes=[]; selection=[]; captured=[]
with threadpool_limits(limits=2),warnings.catch_warnings(record=True) as captured:
    warnings.simplefilter('always')
    for fold in range(5):
        bundle=load_bundle(fold,source,samples)
        oof=bundle['oof']; outer_val=set(bundle['val'].index)
        fine=features(root/f'cache/finetuned/fold_{fold}/outer.npz')
        for name in models:
            target=root/name/f'fold_{fold}'
            head=joblib.load(target/'head.joblib')
            parameters=json.loads((target/'head_parameters.json').read_text())
            assert parameters['pathology_scaler_fit_n']==len(oof)
            if name!='M0':assert parameters['ct_scaler_fit_n']==len(oof)
            np.testing.assert_allclose(head.pa_scaler.mean_,oof.pathology_risk.mean(),rtol=0,atol=1e-12)
            if name.startswith('M2'):
                assert head.ct_pipeline.named_steps['pca'].n_samples_==len(oof)
                assert head.ct_pipeline.named_steps['pca'].components_.dtype==np.dtype('float64')
            frames={}
            for split in ['train','val']:
                frame=pd.read_csv(target/f'{split}_predictions.csv',dtype={'case_id':str}).set_index('case_id')
                np.testing.assert_array_equal(frame.index,bundle[split].index)
                np.testing.assert_allclose(frame['dfs.month'],bundle[split]['dfs.month'])
                np.testing.assert_array_equal(frame['dfs.status'],bundle[split]['dfs.status'])
                x=None
                if name=='M1':x=bundle[split].ct_risk.to_numpy()[:,None]
                elif name.startswith('M2'):x=(public if name=='M2_public' else fine).loc[frame.index].to_numpy()
                elif name in ['M3_volume','M4_shape']:
                    s=shapes.loc[frame.index]
                    x=np.log(s.volume_mm3.to_numpy())[:,None]
                    if name=='M4_shape':x=np.column_stack([x,s.diameter_mm,s.sphericity])
                prediction=head.predict(bundle[split].pathology_risk.to_numpy(),x)
                np.testing.assert_allclose(prediction,frame.risk_score,rtol=1e-9,atol=1e-10)
                frames[split]=frame
            _,actual=fold_metrics(frames['train'],frames['val'],(36.,60.))
            stored=pd.read_csv(root/name/'metrics.csv').set_index('fold').loc[str(fold)]
            for key in ['train_cindex','val_cindex','auc_36m','auc_60m','HR_high_vs_low','logrank_p']:
                np.testing.assert_allclose(actual[key],float(stored[key]),rtol=1e-9,atol=1e-12)
            audit.append(dict(model=name,fold=fold,n_train=len(oof),n_val=len(bundle['val']),
                              prediction_reload_and_metrics_match=True))
            choice=parameters['selection']
            selection.append(dict(model=name,fold=fold,dim=choice['dim'],ridge=choice['ridge'],
                                  beta_p=float(head.model.coef_[0])))
        for ct_source,final in [('public',public),('finetuned',fine)]:
            contexts=[]
            for inner in sorted(oof.inner_fold.astype(int).unique()):
                training=oof.index[~oof.inner_fold.eq(inner)].tolist()
                held=oof.index[oof.inner_fold.eq(inner)].tolist()
                cache=root/f'cache/fold_{fold}/inner_{inner}'
                pa=pd.read_csv(cache/'pathology_train_oof.csv',dtype={'case_id':str}).set_index('case_id')
                assert set(pa.index)==set(training) and pa.pathology_risk.notna().all()
                roles=pd.read_csv(cache/'nested_pa_roles.csv',dtype={'case_id':str})
                assert not set(roles.case_id)&(outer_val|set(held))
                for sub,subset in roles.groupby('subfold'):
                    groups={r:set(g.case_id) for r,g in subset.groupby('role')}
                    assert set.union(*groups.values())==set(training)
                    assert sum(map(len,groups.values()))==len(training)
                selected_features=public if ct_source=='public' else features(root/f'cache/finetuned/fold_{fold}/inner_{inner}.npz')
                contexts.append(dict(train_ids=training,val_ids=held,
                    train_ct=selected_features.loc[training].to_numpy(),val_ct=selected_features.loc[held].to_numpy(),
                    train_volume=np.log(shapes.loc[training,'volume_mm3'].to_numpy()),
                    val_volume=np.log(shapes.loc[held,'volume_mm3'].to_numpy())))
            result=volume_probe(contexts,final.loc[oof.index].to_numpy(),np.log(shapes.loc[oof.index,'volume_mm3']),
                final.loc[bundle['val'].index].to_numpy(),np.log(shapes.loc[bundle['val'].index,'volume_mm3']),
                penalties=config['ridge_grid'])
            target=destination/f'{ct_source}/fold_{fold}';target.mkdir(parents=True,exist_ok=True)
            assert result['model'].solver=='svd' and result['model'].coef_.dtype==np.dtype('float64')
            joblib.dump(dict(transform=result.pop('transform'),model=result.pop('model')),target/'probe.joblib')
            pred=result.pop('prediction')
            pd.DataFrame(dict(case_id=bundle['val'].index,predicted_log_volume=pred,
                volume_mm3=shapes.loc[bundle['val'].index,'volume_mm3'].to_numpy())).to_csv(target/'volume_predictions.csv',index=False)
            old=pd.read_csv(root/f'volume_probe_{ct_source}/fold_{fold}/volume_predictions.csv').set_index('case_id')
            deviation=float(np.max(np.abs(pred-old.loc[bundle['val'].index,'predicted_log_volume'])))
            (target/'metrics.json').write_text(json.dumps(result,indent=2))
            probes.append(dict(source=ct_source,fold=fold,**{k:v for k,v in result.items() if k not in ['selected','candidates']},
                ridge=result['selected']['ridge'],max_abs_prediction_change=deviation))
            print(ct_source,fold,'R2',round(result['r2_log_volume'],6),'max_pred_change',deviation,flush=True)
numeric=pd.DataFrame(probes);numeric.to_csv(destination/'metrics.csv',index=False)
numeric.groupby('source').agg(['mean','std']).to_csv(destination/'summary.csv')
pd.DataFrame(selection).to_csv(root/'review_selected_parameters.csv',index=False)
warnings_list=[str(w.message) for w in captured if not issubclass(w.category,DeprecationWarning)]
assert not warnings_list,warnings_list
audit_report=dict(patients=len(samples),events=int(samples.event.sum()),completed_models=models,
    folds=5,head_checks=audit,all_metric_and_prediction_checks_passed=True,
    cohort_and_source_config_hashes_unchanged=True,all_cached_encoder_hashes_unchanged=True,
    sub_oof_roles_disjoint_from_outer_and_calibration_validation=True,
    volume_numeric_review=dict(dtype='float64',solver='svd',numerical_warnings=warnings_list,
                              metric_dependency_deprecations=sum(issubclass(w.category,DeprecationWarning) for w in captured),
                              original_outputs_preserved=True,neural_retraining=False),
    caveat='Conditional model-fixed bootstrap; exploratory internal CV, not independent confirmation.')
(root/'review_audit.json').write_text(json.dumps(audit_report,indent=2))
print(numeric.groupby('source')[['r2_log_volume','r2_volume','max_abs_prediction_change']].agg(['mean','std']).to_string(),flush=True)
print('ALL AUDITS PASSED',flush=True)
