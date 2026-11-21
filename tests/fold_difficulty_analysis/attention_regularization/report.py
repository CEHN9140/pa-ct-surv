"""Matched inner-selection comparison, audited final risk metrics and attention control."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
import yaml
from sksurv.metrics import concordance_index_censored
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from protocol import CANDIDATES,partition,choose

BASE=Path(__file__).resolve().parent;ROOT=BASE.parents[2]


def summarize(base=BASE):
    base=Path(base);output=base/'results';output.mkdir(exist_ok=True)
    source=pd.read_csv(ROOT/'data/trident/features_uni_v1/all_label_roi64.csv',dtype={'pa_id':str})
    final=[];selection=[];all_inner=[];chosen_curves=[];audits=[];predictions={k:[] for k in ['baseline','entropy','neff']}
    for fold in range(5):
        folder=base/f'fold_{fold}';status=json.loads((folder/'status.json').read_text())
        if status['status']!='completed' or status['completed_inner']!=7 or status['completed_outer']!=3:raise ValueError('Incomplete nested fold')
        fit,stop,train,val=partition(source,fold)
        roles=pd.read_csv(folder/'private/inner_roles.csv',dtype={'pa_id':str})
        if set(roles[roles.role.eq('fit')].pa_id)!=set(source.iloc[fit].pa_id) or set(roles[roles.role.eq('stop')].pa_id)!=set(source.iloc[stop].pa_id):raise ValueError('Inner roles changed')
        records={}
        for config in CANDIDATES:
            directory=folder/'inner'/config['key'];audit=json.loads((directory/'audit.json').read_text())
            if audit['config']!=config or audit['phase']!='inner' or audit['fit_patients']!=len(fit) or audit['stop_patients']!=len(stop):raise ValueError('Inner audit differs')
            curve=pd.read_csv(directory/'epoch_metrics.csv')
            if not np.isclose(curve[curve.split.eq('stop')].cindex.max(),audit['selection_cindex'],rtol=0,atol=1e-12):raise ValueError('Selection score differs')
            records[config['key']]=audit;audits.append(audit)
            curve['fold']=fold;curve['candidate']=config['key'];all_inner.append(curve)
        for family in ['baseline','entropy','neff']:
            config,inner=choose(CANDIDATES,records,family);directory=folder/'outer'/family
            metadata=json.loads((directory/'audit.json').read_text());completed=json.loads((directory/'completed.json').read_text())
            if metadata['config']!=config or metadata['selected_epoch']!=inner['selected_epoch'] or metadata['last_epoch']!=inner['selected_epoch']:raise ValueError('Outer refit selection mismatch')
            if metadata['stop_patients'] or not completed['checkpoint_reload_exact'] or completed['outer_val_passes']!=1:raise ValueError('Outer protocol violation')
            if metadata['patient_exposures']!=len(train)*metadata['last_epoch']:raise ValueError('Lost patients')
            checkpoint_meta=yaml.safe_load((directory/'checkpoint_metadata.yaml').read_text())
            if checkpoint_meta['best_epoch']!=inner['selected_epoch'] or checkpoint_meta['last_epoch']!=inner['selected_epoch'] or checkpoint_meta['outer_validation_used']:
                raise ValueError('Final checkpoint metadata differs')
            best_state=torch.load(directory/'best_model.pth',map_location='cpu',weights_only=True)
            last_state=torch.load(directory/'last_model.pth',map_location='cpu',weights_only=True)
            if set(best_state)!=set(last_state) or not all(torch.equal(best_state[k],last_state[k]) for k in best_state):raise ValueError('Outer checkpoint not the fixed-budget last state')
            if not 0<=metadata['regularizer_active_patient_exposures']<=metadata['patient_exposures']:raise ValueError('Invalid activity count')

            audits.append(metadata)
            metrics=pd.read_csv(directory/'metrics.csv').iloc[0].to_dict();metrics.update(fold=fold,family=family,selected_key=config['key'],selected_weight=config['weight'],selected_epoch=inner['selected_epoch'])
            metrics['active_training_fraction']=metadata['regularizer_active_patient_exposures']/metadata['patient_exposures']
            for split,indices in [('train',train),('val',val)]:
                p=pd.read_csv(directory/f'results/{split}_predictions.csv',dtype={'case_id':str}).set_index('case_id')
                expected=source.iloc[indices].set_index('pa_id')
                if set(p.index)!=set(expected.index):raise ValueError('Prediction cohort changed')
                p=p.loc[expected.index]
                np.testing.assert_array_equal(p['dfs.status'],expected.event)
                np.testing.assert_allclose(p['dfs.month'],expected.time.astype(np.float32),rtol=0,atol=1e-5)
                c=concordance_index_censored(p['dfs.status'].to_numpy(bool),p['dfs.month'].to_numpy(np.float32),p.risk_score.to_numpy(np.float32))[0]
                if abs(c-metrics[split+'_cindex'])>1e-12:raise ValueError('C-index does not reproduce')
                if split=='val':predictions[family].append(p)
            metrics['gap']=metrics['train_cindex']-metrics['val_cindex'];final.append(metrics)
            selection.append(dict(fold=fold,family=family,selected_key=config['key'],weight=config['weight'],q_min=config.get('q_min',np.nan),
                selected_epoch=inner['selected_epoch'],inner_stop_cindex=inner['selection_cindex'],active_training_fraction=metrics['active_training_fraction']))
            selected_curve=pd.read_csv(folder/'inner'/config['key']/'epoch_metrics.csv');selected_curve['fold']=fold;selected_curve['family']=family;chosen_curves.append(selected_curve)
    if len(audits)!=50 or len({row['initial_state_sha256'] for row in audits})!=1:raise ValueError('Incorrect neural fit/init count')
    frame=pd.DataFrame(final);frame.to_csv(output/'fold_metrics.csv',index=False)
    pd.DataFrame(selection).to_csv(output/'selected_parameters.csv',index=False)
    pd.concat(all_inner,ignore_index=True).to_csv(output/'inner_candidate_curves.csv',index=False)
    curves=pd.concat(chosen_curves,ignore_index=True);curves.to_csv(output/'selected_inner_curves.csv',index=False)
    pd.DataFrame([{k:v for k,v in row.items() if k!='config'} for row in audits]).to_csv(output/'fit_audit.csv',index=False)
    summary=[]
    for family,rows in frame.groupby('family'):
        record=dict(family=family,regularized_folds=int(rows.selected_weight.gt(0).sum()))
        for key in ['train_cindex','val_cindex','gap','auc_36m','auc_60m','val_mean_q','val_mean_entropy','val_mean_top128_mass','active_training_fraction']:
            record[key+'_mean']=float(rows[key].mean(skipna=False));record[key+'_std']=float(rows[key].std(skipna=False))
        summary.append(record)
    pd.DataFrame(summary).to_csv(output/'summary.csv',index=False)
    rng=np.random.default_rng(123);draws={k:[] for k in ['entropy','neff']}
    for repeat in range(2000):
        scores={k:[] for k in ['baseline','entropy','neff']}
        for fold in range(5):
            basepred=predictions['baseline'][fold]
            for attempt in range(1000):
                chosen=rng.integers(len(basepred),size=len(basepred))
                e=basepred['dfs.status'].to_numpy(bool)[chosen];t=basepred['dfs.month'].to_numpy(np.float32)[chosen]
                if any(e[i] and np.any((t>t[i])|((t==t[i])&~e)) for i in range(len(t))):break
            else:raise ValueError('Unable to draw comparable survival pairs')
            for family in scores:
                p=predictions[family][fold].loc[basepred.index]
                scores[family].append(concordance_index_censored(p['dfs.status'].to_numpy(bool)[chosen],p['dfs.month'].to_numpy(np.float32)[chosen],p.risk_score.to_numpy(np.float32)[chosen])[0])
        for family in draws:draws[family].append(float(np.mean(scores[family])-np.mean(scores['baseline'])))
    contrasts=[];pivot=frame.pivot(index='fold',columns='family',values='val_cindex')
    for family,values in draws.items():
        delta=pivot[family]-pivot.baseline
        contrasts.append(dict(family=family,mean_delta=float(delta.mean()),positive_folds=int(delta.gt(0).sum()),
            interval_low=float(np.quantile(values,.025)),interval_high=float(np.quantile(values,.975)),bootstrap_repeats=2000))
    pd.DataFrame(contrasts).to_csv(output/'paired_comparisons.csv',index=False)
    with plt.rc_context({'font.size':10,'axes.labelweight':'bold'}):
        fig,axes=plt.subplots(2,5,figsize=(19,8))
        for fold in range(5):
            for family in ['baseline','entropy','neff']:
                rows=curves[curves.fold.eq(fold)&curves.family.eq(family)&curves.split.eq('stop')]
                axes[0,fold].plot(rows.epoch,rows.cindex,label=family,linewidth=2)
                axes[1,fold].plot(rows.epoch,rows.mean_q,linewidth=2)
            axes[0,fold].set_title(f'Fold {fold} | inner STOP');axes[0,fold].set_ylabel('STOP C-index')
            axes[1,fold].set(xlabel='Epoch',ylabel='STOP mean Neff/N')
            for ax in axes[:,fold]:ax.grid(alpha=.2)
        axes[0,0].legend();fig.tight_layout();fig.savefig(output/'inner_stability.png',dpi=180);plt.close(fig)
    lines=['# Attention regularization: matched inner-selection survival comparison','',
        'Completed35 inner fits and15 freshly initialized outer refits. Raw UNI/full patches, original ABMIL/Cox32/Adam1e-4/WD5e-4/dropout.25/seed123. Each saved outer fold is unchanged. Fixed event-stratified80/20 FIT/STOP inside Outer Train selects candidate and Epoch; Outer Val is evaluated only at the final fixed-budget checkpoint. Single inner holdout, not threefold inner CV.',
        '', 'This implements the user-specified normalized entropy loss inspired by [AEM MICCAI2025](https://papers.miccai.org/miccai-2025/0056-Paper5183.html), not an exact classification/Cosine Weight Annealing reproduction. No EMA, architecture changes, input L2 or patch sampling.',
        '', '| Method | Train C-index | Val C-index | AUC36m | AUC60m | Regularized folds |','|---|---:|---:|---:|---:|---:|']
    for row in summary:
        lines.append('| '+row['family']+' | '+' | '.join(f"{row[k+'_mean']:.4f} ± {row[k+'_std']:.4f}" for k in ['train_cindex','val_cindex','auc_36m','auc_60m'])+f" | {row['regularized_folds']}/5 |")
    lines+=['','## Outer-fold incremental performance','', '| Method vs matched baseline | Mean delta C | Positive folds | Conditional95% interval |','|---|---:|---:|---:|']
    for r in contrasts:lines.append(f"| {r['family']} | {r['mean_delta']:+.4f} | {r['positive_folds']}/5 | [{r['interval_low']:+.4f}, {r['interval_high']:+.4f}] |")
    lines+=['','Intervals use2000 paired patient resamples within each validation fold, conditional on fitted models. They omit refitting/selection and previous repeated exploration and are unadjusted for two regularized-family comparisons. This remains exploratory internal validation, not an independent external test.',
        '', '## Actual control strength','', '| Family | Mean Outer Val Neff/N | Mean Outer Val normalized entropy | Mean training active-penalty fraction |','|---|---:|---:|---:|']
    for r in summary:lines.append(f"| {r['family']} | {r['val_mean_q_mean']:.4f} | {r['val_mean_entropy_mean']:.4f} | {r['active_training_fraction_mean']:.4f} |")
    lines+=['','A family can select zero regularization. A nonzero coefficient with little active exposure is not proof the intended constraint materially changed optimization. Larger q/reduced Train-Val gap alone is not success; DFS ranking/AUC must improve relative to the freshly fitted matched baseline.',
        '', 'Selected inner curves are selected on STOP and therefore descriptive, not unbiased evidence of an effect. The independent outer final comparison evaluates the full selection procedure. Historical best scores selected on Outer Val are a different protocol and must not be used as the matched comparator.',
        '', 'All50 audits share the same initial parameter hash. Inner/outer patient roles, fixed outer Epoch, exposure counts and30 final Train/Val C-indices were independently checked from saved artifacts. No project-trained unimodal model initializes any fit. Private patient mappings/predictions and all checkpoints stay local. No guarantee of benefit to a future multimodal teacher or CT student is established here.']
    (output/'report.md').write_text('\n'.join(lines)+'\n')
    (output/'validation.json').write_text(json.dumps(dict(passed=True,inner_fits=35,outer_fits=15,neural_fits=50,outer_validation_passes_per_model=1,
        source_roles_verified=True,fixed_epoch_refit_verified=True,initialization_equal=True,final_cindices_recomputed=True,private_data_excluded=True),indent=2))


if __name__=='__main__':summarize()
