"""Conditional descriptive report; no tissue diagnoses or causal attribution claims."""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sksurv.metrics import concordance_index_censored
from attribution import pair_changes
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE=Path(__file__).resolve().parent
QUALITY=['white_fraction','black_fraction','tissue_fraction','laplacian_variance','od_mean','saturation_mean','od_r','od_g','od_b','valid_bounds_fraction']


def summarize(smoke=False):
    private=BASE/('validation/smoke' if smoke else 'private');out=private/'report' if smoke else BASE/'results';out.mkdir(exist_ok=True)
    if not json.loads((private/'quality_completed.json').read_text())['passed']:raise ValueError('Native QC not complete')
    patients=pd.read_csv(private/'patients.csv',dtype={'pa_id':str});qc=pd.read_csv(private/'patch_quality.csv',dtype={'pa_id':str})
    selected=pd.read_csv(private/'selected_patches.csv',dtype={'pa_id':str});pairs=pd.read_csv(private/'pairs.csv',dtype={'event_pa_id':str,'other_pa_id':str})
    if len(patients)!= (4 if smoke else 721) or patients.pa_id.duplicated().any() or len(qc)!=len(selected):raise ValueError('Incomplete cohort')
    # Independently reconstruct each comparable pair and each patient's involvement.
    for fold,frame in patients.groupby('fold',sort=True):
        frame=frame.reset_index(drop=True);stored=pairs[pairs.fold.eq(fold)].reset_index(drop=True)
        recomputed,roles=pair_changes(frame.risk_early_attention,frame.risk_late_attention,frame.time,frame.event)
        temporal,temporal_roles=pair_changes(frame.actual_epoch5_risk,frame.actual_late_risk,frame.time,frame.event)
        best_temporal,_=pair_changes(frame.actual_best_risk,frame.actual_late_risk,frame.time,frame.event)
        for key in recomputed:
            np.testing.assert_allclose(stored[key],recomputed[key],rtol=1e-10,atol=1e-9,equal_nan=True)
        np.testing.assert_array_equal(stored.event_pa_id,frame.pa_id.to_numpy()[recomputed.event_index])
        np.testing.assert_array_equal(stored.other_pa_id,frame.pa_id.to_numpy()[recomputed.other_index])
        np.testing.assert_allclose(stored.temporal_credit_change,temporal.credit_change,rtol=0,atol=0)
        np.testing.assert_allclose(stored.best_to_late_credit_change,best_temporal.credit_change,rtol=0,atol=0)
        for key in ['comparable_degree','pair_credit_change','mean_pair_credit_change','broken_pairs','corrected_pairs']:
            np.testing.assert_allclose(frame[key],roles[key],rtol=0,atol=1e-12,equal_nan=True)
        for key in ['pair_credit_change','mean_pair_credit_change','broken_pairs','corrected_pairs']:
            np.testing.assert_allclose(frame['temporal_'+key],temporal_roles[key],rtol=0,atol=1e-12,equal_nan=True)
    key=['pa_id','fold','group','patch_index']
    if selected.duplicated(key).any() or qc.duplicated(key).any():raise ValueError('Duplicate selected/QC patch key')
    if set(map(tuple,selected[key].to_numpy()))!=set(map(tuple,qc[key].to_numpy())):raise ValueError('Selected/QC patch keys differ')
    summaries=[];rng=np.random.default_rng(123)
    groupmeans=qc.groupby(['pa_id','fold','group'])[QUALITY].mean().reset_index()
    contrasts=groupmeans.pivot(index=['pa_id','fold'],columns='group',values=QUALITY)
    for metric in QUALITY:
        delta=contrasts[metric]['high_abs_ig']-contrasts[metric]['low_abs_ig'];valid=delta.dropna().to_numpy()
        draws=np.mean(rng.choice(valid,(2000,len(valid))),axis=1) if len(valid) else np.full(2000,np.nan)
        summaries.append(dict(metric=metric,patients_with_both_values=len(valid),high_mean=float(contrasts[metric]['high_abs_ig'].mean()),
            low_mean=float(contrasts[metric]['low_abs_ig'].mean()),mean_within_wsi_difference=float(np.mean(valid)) if len(valid) else np.nan,
            interval_low=float(np.quantile(draws,.025)),interval_high=float(np.quantile(draws,.975))))
    pd.DataFrame(summaries).to_csv(out/'quality_contrasts.csv',index=False)
    difference=pd.DataFrame({metric:contrasts[metric]['high_abs_ig']-contrasts[metric]['low_abs_ig'] for metric in QUALITY}).reset_index()
    difference=patients.merge(difference,on=['pa_id','fold'],validate='one_to_one');difference.to_csv(private/'patient_quality_contrasts.csv',index=False)
    associations=[]
    for fold,frame in [('all',difference),*list(difference.groupby('fold'))]:
        for metric in QUALITY:
            for outcome in ['mean_pair_credit_change','temporal_mean_pair_credit_change']:
                values=frame[[metric,outcome]].dropna()
                rho=spearmanr(values[metric],values[outcome]).statistic if len(values)>3 and values.nunique().min()>1 else np.nan
                associations.append(dict(fold=fold,metric=metric,outcome=outcome,patients=len(values),spearman_with_pair_credit_change=float(rho)))
    pd.DataFrame(associations).to_csv(out/'quality_pair_associations.csv',index=False)
    # Record each selected patch's signed change to each changed comparable-pair margin.
    contribution_rows=[];lookup={str(k):v for k,v in selected[selected.group.eq('high_abs_ig')].groupby('pa_id')}
    for pair_id,pair in pairs[pairs.credit_change.ne(0)].iterrows():
        for patient,orientation in [(str(pair.event_pa_id),1),(str(pair.other_pa_id),-1)]:
            for patch in lookup[patient].itertuples():
                contribution_rows.append(dict(fold=int(pair.fold),pair_id=pair_id,event_pa_id=pair.event_pa_id,other_pa_id=pair.other_pa_id,
                    credit_change=pair.credit_change,pa_id=patient,patch_index=int(patch.patch_index),risk_contribution=patch.contribution,
                    pair_margin_contribution=orientation*patch.contribution))
    pd.DataFrame(contribution_rows,columns=['fold','pair_id','event_pa_id','other_pa_id','credit_change','pa_id','patch_index','risk_contribution','pair_margin_contribution']).to_csv(private/'changed_pair_high_impact_patches.csv',index=False)
    np.testing.assert_allclose(pairs.event_ig_sum-pairs.other_ig_sum,pairs.margin_delta,atol=1e-9,rtol=1e-9)
    # Re-read all private per-patch arrays to verify completeness independently.
    for row in patients.itertuples():
        with np.load(private/f'fold_{row.fold}/{row.pa_id}/attribution.npz',allow_pickle=False) as f:
            c=f['contributions'];assert len(c)==row.n_patches
            if abs(float(f['best_contributions'].sum())-row.best_to_late_delta)>1e-8*max(1.,abs(row.best_to_late_delta)):raise ValueError('Best-reference completeness fails')
            if abs(float(c.sum())-row.risk_delta)>1e-8*max(1.,abs(row.risk_delta)):raise ValueError('Saved-patch completeness fails')
    folds=[]
    for fold,frame in patients.groupby('fold'):
        if smoke:continue
        scores={name:float(concordance_index_censored(frame.event.to_numpy(bool),frame.time,frame[name])[0]) for name in ['actual_epoch5_risk','actual_epoch10_risk','actual_best_risk','actual_late_risk','risk_early_attention','best_attention_risk','risk_late_attention']}
        p=pairs[pairs.fold.eq(fold)]
        if abs(scores['risk_late_attention']-scores['risk_early_attention']-p.credit_change.mean())>1e-12:raise ValueError('Recomputed pairs do not agree with C-index')
        folds.append(dict(fold=int(fold),patients=len(frame),**scores,attention_delta_c=float(p.credit_change.mean()),
            corrected_pairs=int(p.credit_change.gt(0).sum()),broken_pairs=int(p.credit_change.lt(0).sum()),
            temporal_broken_pairs=int(p.temporal_credit_change.lt(0).sum()),
            temporal_and_attention_broken=int((p.credit_change.lt(0)&p.temporal_credit_change.lt(0)).sum()),
            mean_abs_risk_change=float(frame.risk_delta.abs().mean()),max_completeness_error=float(frame.completeness_error.max())))
    pd.DataFrame(folds).to_csv(out/'fold_metrics.csv',index=False)
    # Conditional paired survival bootstrap, keeping fold roles and both predictions paired.
    boot=[]
    if not smoke:
        for repeat in range(1000):
            scores=[]
            for fold,frame in patients.groupby('fold'):
                e=frame.event.to_numpy(bool);t=frame.time.to_numpy(float)
                for attempt in range(1000):
                    ix=rng.integers(len(frame),size=len(frame))
                    if any(e[ix][j] and np.any((t[ix]>t[ix][j])|((t[ix]==t[ix][j])&~e[ix])) for j in range(len(ix))):break
                else:raise ValueError('No comparable bootstrap sample')
                a=concordance_index_censored(e[ix],t[ix],frame.risk_early_attention.to_numpy()[ix])[0]
                b=concordance_index_censored(e[ix],t[ix],frame.risk_late_attention.to_numpy()[ix])[0];scores.append(b-a)
            boot.append(float(np.mean(scores)))
        pd.DataFrame([dict(contrast='late_attention_minus_epoch5_attention_fixed_late_head',mean_delta=float(np.mean([r['attention_delta_c'] for r in folds])),
            interval_low=float(np.quantile(boot,.025)),interval_high=float(np.quantile(boot,.975)),repeats=1000)]).to_csv(out/'paired_interval.csv',index=False)
    with plt.rc_context({'font.size':12,'axes.labelweight':'bold','axes.titleweight':'bold'}):
        fig,axes=plt.subplots(1,3,figsize=(17,5))
        axes[0].hist(patients.risk_delta,bins=25,color='#31799e');axes[0].set(xlabel='Attention-only risk change',ylabel='Patients')
        axes[1].scatter(difference.white_fraction,difference.mean_pair_credit_change,s=16,alpha=.6)
        axes[1].set(xlabel='Within-WSI white fraction: high minus low IG',ylabel='Mean pair-credit change')
        s=pd.DataFrame(summaries).set_index('metric').loc[['white_fraction','black_fraction','tissue_fraction']]
        axes[2].errorbar(s.mean_within_wsi_difference,np.arange(len(s)),xerr=np.vstack([np.maximum(0,s.mean_within_wsi_difference-s.interval_low),np.maximum(0,s.interval_high-s.mean_within_wsi_difference)]),fmt='o',capsize=4)
        axes[2].set_yticks(np.arange(len(s)),s.index);axes[2].set(xlabel='Within-WSI technical contrast')
        for ax in axes:ax.grid(alpha=.2)
        fig.tight_layout();fig.savefig(out/'attribution_quality_overview.png',dpi=180);plt.close(fig)
    lines=['# Frozen attention-risk attribution and native-patch quality diagnostic','',
        'No new neural training. Original Raw UNI full bags and bs32 five-fold last checkpoints are frozen. Epoch5 attention is the reference. All outputs are exploratory, conditional on models originally early-stopped using outer validation.',
        '', '## What is decomposed','',
        'For each patient, freeze late projected patch vectors and risk head; interpolate normalized Epoch5 to late attention. The eval head is Linear/ReLU/Dropout(identity)/Linear. Exact integration of each ReLU active fraction yields patch IG. All patch contributions sum to the counterfactual attention-only risk change. This is not patch deletion, a unique causal attribution, or attribution of the entire actual early-to-late prediction change.',
        '', 'Actual Epoch5/Epoch10/best/late risks and best-attention-to-late IG are recorded separately. patients.best_to_late_delta denotes frozen best-attention-to-late risk, while pairs.best_to_late_credit_change denotes actual best-model-to-late credit; these are different comparisons. All comparisons use comparable survival pairs including event/censor at the same time and half-credit for risk ties. Both endpoint contributions are needed: pair-margin delta equals event-patient summed IG minus comparator summed IG. A positive risk contribution can help or harm depending on the paired patient and event ordering.',
        '', '| Fold | Actual E5 C | Actual best C | Actual late C | Early attention / late head C | Late attention / late head C | Attention delta C | Corrected pairs | Broken pairs |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in folds:lines.append(f"| {r['fold']} | {r['actual_epoch5_risk']:.4f} | {r['actual_best_risk']:.4f} | {r['actual_late_risk']:.4f} | {r['risk_early_attention']:.4f} | {r['risk_late_attention']:.4f} | {r['attention_delta_c']:+.4f} | {r['corrected_pairs']} | {r['broken_pairs']} |")
    lines+=['', '## Matched native-patch quality','',
        'Per WSI: eight largest absolute-IG patches versus eight smallest absolute-IG controls, disjoint; fewer if necessary. Both directions of risk contribution are retained. Patches are read at native metadata-matched TRIDENT FOV, not approximated from thumbnails. Within-WSI matching controls slide-wide scanner/stain differences but not local tissue type or other patch confounders. This extreme-group contrast does not estimate WSI-wide artifact prevalence.',
        '', 'White/black fractions are technical proxies. Laplacian variance is measured inside an eroded non-background mask and also depends on tissue texture; it is not a validated blur label. Optical density/channel summaries are not H&E stain concentrations or tissue diagnoses. Blank patches may have undefined sharpness/OD: available-patient counts are explicit.',
        '', '| Proxy | Patients with both values | High-minus-low mean | Conditional95% patient interval |','|---|---:|---:|---:|']
    for r in summaries:lines.append(f"| {r['metric']} | {r['patients_with_both_values']} | {r['mean_within_wsi_difference']:+.5f} | [{r['interval_low']:+.5f}, {r['interval_high']:+.5f}] |")
    lines+=['','Quality differences and pair-harm correlations are descriptive associations, not proof technical quality caused overfitting. Multiple proxies and outcome-dependent follow-up are unadjusted; intervals omit retraining and prior exploration. Pair endpoints are dependent. No pathology diagnoses or stable DFS-related histological associations can be established by this script.',
        '', '## Blinded review','',
        'private/reviewer contains15 cases: two worst actual Epoch5-to-late mean pair-credit changes and one stable control with fewest changed comparable pairs per fold. A/B are shuffled high-impact/control groups; IDs and group decoding stay in private/review_key.csv. Blank tissue/artifact/quality fields await an actual reviewer. Selection uses these same outcomes and therefore cannot serve as independent validation. Panels show native patches and WSI locations. No automatic histology label was generated.',
        '', '## Interpretation boundary','',
        'A complete IG decomposition establishes computational accounting for the chosen frozen intervention, not biological causation. High absolute IG includes positive and negative risk changes, not just harmful patches. Absolute-IG controls differ in model influence by construction; individual patch allocations also depend on the chosen representation parameterization even when their sum is invariant. Correlated nearby patches are not independent tissue observations. If high-impact quality differences replicate across folds and relate to pair deterioration, that motivates a future predefined QC intervention; otherwise this diagnostic cannot distinguish unstable biological associations from limited DFS supervision or missing prognostic information. No new loss or network is warranted from a correlation alone.']
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    (out/'validation.json').write_text(json.dumps(dict(passed=True,patients=len(patients),neural_training=False,all_saved_ig_complete=True,comparable_pairs_recomputed=True,
        native_patch_qc=True,review_awaits_human=True,best_reference_ig_complete=True),indent=2))
    print('Report and independent recomputation complete',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');summarize(p.parse_args().smoke)
