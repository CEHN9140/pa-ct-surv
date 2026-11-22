"""Independent best-reference Train/Val reconstruction and matched-quality summaries."""
from pathlib import Path
import argparse,json,sys
import numpy as np
import pandas as pd
from sksurv.metrics import concordance_index_censored

BASE=Path(__file__).resolve().parent;PARENT=BASE.parent;ROOT=PARENT.parents[2]
sys.path.insert(0,str(PARENT))
from attribution import pair_changes
from best_tools import contextual_counts,context_scores

PROXIES=['white_fraction','black_fraction','laplacian_variance','od_mean','saturation_mean','od_r','od_g','od_b']


def summarize(smoke=False):
    private=BASE/('validation/smoke' if smoke else 'private');out=private/'report' if smoke else BASE/'results';out.mkdir(exist_ok=True)
    quality_status=json.loads((private/'quality_completed.json').read_text());assert quality_status['passed']
    cohort=pd.read_csv(ROOT/'data/trident/features_uni_v1/all_label_roi64.csv',dtype={'pa_id':str})
    patients=pd.read_csv(private/'patients.csv',dtype={'pa_id':str});pairs=pd.read_csv(private/'pairs.csv',dtype={'event_pa_id':str,'other_pa_id':str})
    if patients.duplicated(['fold','pa_id']).any() or len(patients)!=(64 if smoke else 3605):raise ValueError('Fold-patient count differs')
    records=[]
    for (fold,split),frame in patients.groupby(['fold','split'],sort=True):
        frame=frame.reset_index(drop=True);expected=cohort[cohort.split.eq(fold) if split=='val' else cohort.split.ne(fold)]
        if not smoke and set(frame.pa_id)!=set(expected.pa_id):raise ValueError('Patient roles changed')
        labels=cohort.set_index('pa_id').loc[frame.pa_id]
        np.testing.assert_array_equal(frame.event,labels.event);np.testing.assert_array_equal(frame.time.to_numpy(np.float32),labels.time.to_numpy(np.float32))
        frozen,roles=pair_changes(frame.frozen_best_attention_risk,frame.frozen_late_attention_risk,frame.time,frame.event)
        actual,actual_roles=pair_changes(frame.actual_best_risk,frame.actual_late_risk,frame.time,frame.event)
        saved=pairs[pairs.fold.eq(fold)&pairs.split.eq(split)].reset_index(drop=True)
        for key in frozen:np.testing.assert_allclose(saved[key],frozen[key],atol=1e-9,rtol=1e-9,equal_nan=True)
        np.testing.assert_array_equal(saved.actual_credit_change,actual.credit_change)
        np.testing.assert_array_equal(saved.event_pa_id,frame.pa_id.to_numpy()[frozen.event_index]);np.testing.assert_array_equal(saved.other_pa_id,frame.pa_id.to_numpy()[frozen.other_index])
        counts=contextual_counts(frozen,len(frame))
        for key in counts:np.testing.assert_allclose(frame[key],counts[key],atol=0,rtol=0)
        for key in ['comparable_degree','pair_credit_change','mean_pair_credit_change','broken_pairs','corrected_pairs']:
            np.testing.assert_allclose(frame[key],roles[key],atol=1e-12,rtol=0,equal_nan=True);np.testing.assert_allclose(frame['actual_'+key],actual_roles[key],atol=1e-12,rtol=0,equal_nan=True)
        for row in frame.itertuples():
            with np.load(private/f'fold_{fold}/{row.pa_id}/best_attribution.npz',allow_pickle=False) as f:c=f['contributions']
            if len(c)!=row.n_patches or abs(c.sum()-row.attention_risk_delta)>1e-8*max(1.,abs(row.attention_risk_delta)):raise ValueError('IG not complete')
            s=context_scores(c,row)
            np.testing.assert_allclose(s['harm_support'].sum(),row.harm_support_mass,rtol=1e-9,atol=1e-9)
            np.testing.assert_allclose(s['repair_support'].sum(),row.repair_support_mass,rtol=1e-9,atol=1e-9)
        score={key:float(concordance_index_censored(frame.event.to_numpy(bool),frame.time,frame[key])[0]) for key in ['actual_best_risk','actual_late_risk','frozen_best_attention_risk','frozen_late_attention_risk']}
        if abs(score['frozen_late_attention_risk']-score['frozen_best_attention_risk']-frozen.credit_change.mean())>1e-12:raise ValueError('C-index credit mismatch')
        records.append(dict(fold=fold,split=split,patients=len(frame),best_epoch=int(frame.best_epoch.iloc[0]),last_epoch=int(frame.last_epoch.iloc[0]),**score,
            attention_delta_c=float(frozen.credit_change.mean()),actual_delta_c=float(actual.credit_change.mean()),corrected_pairs=int(frozen.credit_change.gt(0).sum()),broken_pairs=int(frozen.credit_change.lt(0).sum())))
    metrics=pd.DataFrame(records);metrics.to_csv(out/'fold_metrics.csv',index=False)
    if not smoke:
        old=pd.read_csv(PARENT/'results/fold_metrics.csv').set_index('fold')
        current=metrics[metrics.split.eq('val')].set_index('fold')
        np.testing.assert_allclose(current.frozen_late_attention_risk,old.risk_late_attention,rtol=0,atol=1e-12)
        pd.DataFrame(dict(epoch5_reference_delta=old.attention_delta_c,best_reference_delta=current.attention_delta_c)).to_csv(out/'reference_sensitivity.csv')
    summary=[]
    for split,g in metrics.groupby('split'):
        row=dict(split=split)
        for key in ['actual_best_risk','actual_late_risk','frozen_best_attention_risk','frozen_late_attention_risk','attention_delta_c','actual_delta_c']:
            row[key+'_mean']=g[key].mean();row[key+'_std']=g[key].std()
        row['positive_attention_folds']=int(g.attention_delta_c.gt(0).sum());summary.append(row)
    pd.DataFrame(summary).to_csv(out/'summary.csv',index=False)
    rng=np.random.default_rng(123);intervals=[]
    if not smoke:
        draws={'frozen_attention':[],'actual_models':[]}
        for repeat in range(1000):
            values={k:[] for k in draws}
            for fold,frame in patients[patients.split.eq('val')].groupby('fold'):
                e=frame.event.to_numpy(bool);t=frame.time.to_numpy(float)
                for attempt in range(1000):
                    ix=rng.integers(len(frame),size=len(frame))
                    if any(e[ix][i] and np.any((t[ix]>t[ix][i])|((t[ix]==t[ix][i])&~e[ix])) for i in range(len(ix))):break
                else:raise ValueError('Bootstrap lacks comparable pairs')
                for name,a,b in [('frozen_attention','frozen_best_attention_risk','frozen_late_attention_risk'),('actual_models','actual_best_risk','actual_late_risk')]:
                    values[name].append(concordance_index_censored(e[ix],t[ix],frame[b].to_numpy()[ix])[0]-concordance_index_censored(e[ix],t[ix],frame[a].to_numpy()[ix])[0])
            for key in draws:draws[key].append(float(np.mean(values[key])))
        for name,values in draws.items():intervals.append(dict(contrast=name,interval_low=float(np.quantile(values,.025)),interval_high=float(np.quantile(values,.975)),repeats=1000))
    pd.DataFrame(intervals).to_csv(out/'val_conditional_intervals.csv',index=False)
    matches=pd.read_csv(private/'matches.csv',dtype={'pa_id':str});qc=pd.read_csv(private/'patch_quality.csv',dtype={'pa_id':str});coverage=pd.read_csv(private/'matching_coverage.csv',dtype={'pa_id':str})
    if qc.duplicated(['fold','pa_id','group','patch_index']).any():raise ValueError('Duplicated QC patch')
    for row in matches.itertuples():
        high=qc[qc.fold.eq(row.fold)&qc.pa_id.eq(row.pa_id)&qc.match_id.eq(row.match_id)&qc.group.eq('high_context')]
        low=qc[qc.fold.eq(row.fold)&qc.pa_id.eq(row.pa_id)&qc.match_id.eq(row.match_id)&qc.group.eq('matched_low')]
        if len(high)!=1 or len(low)!=1 or int(high.patch_index.iloc[0])!=row.high_index or int(low.patch_index.iloc[0])!=row.low_index:raise ValueError('Matching keys differ')
        if abs(abs(high.white_fraction.iloc[0]-low.white_fraction.iloc[0])-row.native_white_difference)>1e-12 or row.thumbnail_white_difference>.1+1e-12:raise ValueError('Matching caliper metadata differs')
        if bool(row.native_caliper_pass)!=(row.native_white_difference<=.1+1e-12):raise ValueError('Native flag differs')
    if len(qc)!=2*len(matches):raise ValueError('Missing quality endpoint')
    agg=coverage.groupby(['split','basis']).agg(observations=('pa_id','size'),requested_high=('candidates','sum'),thumbnail_matches=('matches','sum'),native_caliper_pass=('native_caliper_pass','sum')).reset_index()
    agg.to_csv(out/'matching_coverage.csv',index=False)
    allowed=matches[matches.basis.eq('broken_pair_margin')&matches.native_caliper_pass].copy()
    diffs=[]
    for row in allowed.itertuples():
        subset=qc[qc.fold.eq(row.fold)&qc.pa_id.eq(row.pa_id)&qc.match_id.eq(row.match_id)].set_index('group')
        diffs.append(dict(fold=row.fold,split=row.split,pa_id=row.pa_id,match_id=row.match_id,**{key:subset.loc['high_context',key]-subset.loc['matched_low',key] for key in PROXIES}))
    differences=pd.DataFrame(diffs,columns=['fold','split','pa_id','match_id',*PROXIES]);differences.to_csv(private/'quality_pair_differences.csv',index=False)
    qsummary=[]
    patientmeans=differences.groupby(['fold','split','pa_id'])[PROXIES].mean().reset_index()
    for (fold,split),g in patientmeans.groupby(['fold','split']):
        for key in PROXIES:qsummary.append(dict(fold=fold,split=split,metric=key,patient_observations=int(g[key].notna().sum()),mean_matched_difference=g[key].mean()))
    pd.DataFrame(qsummary).to_csv(out/'quality_fold_contrasts.csv',index=False)
    intervals_q=[]
    # Val patients appear once. No bootstrap treating repeated Train observations as independent.
    g=patientmeans[patientmeans.split.eq('val')]
    for key in PROXIES:
        values=g[key].dropna().to_numpy();boot=np.mean(rng.choice(values,(2000,len(values))),axis=1) if len(values) else np.full(2000,np.nan)
        intervals_q.append(dict(metric=key,patients=len(values),mean_difference=float(np.mean(values)) if len(values) else np.nan,
            interval_low=float(np.quantile(boot,.025)),interval_high=float(np.quantile(boot,.975))))
    pd.DataFrame(intervals_q).to_csv(out/'val_quality_contrasts.csv',index=False)
    lines=['# Best-attention reference: frozen Train/Val diagnostic','',
        'No retraining. Same original721 patients, saved five folds, Raw UNI complete patches and frozen last ABMIL projection/head. All Train/Val metrics use eval mode. Best epoch was selected on Outer Val: follow-up cannot remove this selection bias and conditional bootstrap does not cover it.',
        '', '| Fold | Role | Actual best C | Actual late C | Best attention + late H/head C | Late attention + late H/head C | Attention delta C |','|---|---|---:|---:|---:|---:|---:|']
    for r in records:lines.append(f"| {r['fold']} | {r['split']} | {r['actual_best_risk']:.4f} | {r['actual_late_risk']:.4f} | {r['frozen_best_attention_risk']:.4f} | {r['frozen_late_attention_risk']:.4f} | {r['attention_delta_c']:+.4f} |")
    lines+=['','## Contextual patch interpretation','',
        'Exact best-reference straight-line attention IG sums to frozen risk difference. For an event-patient/comparator pair, margin change=sum(event IG)-sum(comparator IG). The contextual scores measure negative margin support on credit-losing pairs, its opposing mass, positive support on credit-gaining pairs, and its opposing mass. Half-credit ties weight the corresponding pair context by0.5. One patch may support both correction and damage across different pairs. These scores are not unique allocations of C-index loss, tissue importance or causal effects. Raw mass includes pair exposure counts and risk units, so it must not be compared directly across Train/Val or models; it is used only for within-WSI selection here.',
        '', '## Matching and native quality','',
        'Primary matching: up to8 positive-context-harm patches per WSI, disjoint bottom-half low-score controls, thumbnail white-fraction caliper0.1, then nearest normalized UNI cosine. UNI normalization is used only for matching, not for model input. No histology labels exist: this is not same-tissue matching. Actual native white-fraction differences are audited; primary QC contrasts use native difference<=0.1. Both matching failure rates and all screened patches are retained. White fraction is a balance variable here: small post-match white differences do not rule out background confounding.',
        '', 'Val QC covers eligible patients; Train QC covers only15 selected review examples, not the full Train distribution. Mean Train/Val C-indices are fold summaries; training patients recur across four folds and are not independent observations. No independent-patient bootstrap or hypothesis test is applied to the repeated Train sample.',
        '', 'Quality proxies are white/black fractions, tissue-masked Laplacian variance and OD/saturation summaries. They do not label blur, stain artifacts or biological tissue types. Native sharpness/OD can depend on local tissue structure; matching is incomplete control, not randomized evidence.',
        '', '## Blinded review','',
        f"Planned{quality_status['planned_review_cases']} review cases; rendered{quality_status['rendered_review_cases']}. Per fold: two most actual-best-to-late deteriorated Val patients plus a stable control; two most improved Train patients plus a stable control. Case IDs/group labels/tile order are shuffled. No-harm review cases may use abs IG fallback; these are marked and excluded from primary harm-QC summaries. Native tiles, WSI locations, matching keys and empty forms stay private; tissue labels and actual doctor review remain pending.",
        '', '## Limits','',
        'The existing fixed-Epoch5 and best references are both retained in reference_sensitivity.csv; a best-conditioned sign pattern is not independent of how the peak was selected. Best-conditioned 5/5 negative Val changes are descriptive, not independent validation or proof attention is the sole cause. A Train improvement/Val decline contrast supports fitted-sample specificity under this intervention, not a proven disease mechanism. Counterfactual path components depend on the chosen endpoint network and representation. Technical QC associations cannot identify stable histological DFS relations; actual pathology review and an independent evaluation would still be needed. No new loss, encoder adaptation or formal method change follows automatically.',
        '', f'Verification: all{len(patients)} saved IG decompositions, separate Train/Val comparable pairs, patient roles, context counts/masses and matching endpoints are independently recomputed. Prior validation IG is checked against the original best_contributions before reuse. All checkpoint/source files are read-only.']
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    (out/'validation.json').write_text(json.dumps(dict(passed=True,fold_patient_observations=len(patients),training=False,all_saved_ig_complete=True,train_val_pairs_recomputed=True,
        context_counts_verified=True,native_match_endpoints_verified=True,actual_histology_matching=False,review_pending=True),indent=2))
    print('Best-reference report and independent checks complete',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');summarize(p.parse_args().smoke)
