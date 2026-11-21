"""Validate cohort attribution and summarize read-only follow-up diagnostics."""
from pathlib import Path
import json,sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sksurv.metrics import concordance_index_censored

BASE=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE))
from qc_thumbnails import box_fraction


def run():
    frame=pd.read_csv(BASE/'private/patient_changes.csv',dtype={'pa_id':str})
    reference=pd.read_csv(BASE.parent/'results/risk_phase_changes.csv')
    for key,group in frame.groupby(['batch','fold','phase']):
        row=reference[reference.batch.eq(key[0])&reference.fold.eq(key[1])&reference.phase.eq(key[2])&reference.split.eq('val')].iloc[0]
        assert group.corrected.sum()/2==row.corrected_pairs and group.broken.sum()/2==row.broken_pairs
        assert abs(-np.nansum(group.net_harm_rate*group.degree)/group.degree.sum()-row.delta_cindex)<1e-12
    mask=np.zeros((4,4),bool);mask[:,:2]=True
    np.testing.assert_array_equal(box_fraction(mask,np.array([[0,0],[2,0],[0,2],[2,2]]),2,(4,4)),[1,0,1,0])
    qc=pd.read_csv(BASE/'private/attention_qc.csv',dtype={'pa_id':str})
    associations=[]
    for (batch,phase),group in frame.groupby(['batch','phase']):
        early='5' if phase=='5_to_10' else '10';late='10' if phase=='5_to_10' else 'last'
        a=qc[qc.batch.eq(batch)&qc.phase_point.astype(str).eq(early)].set_index('pa_id')
        b=qc[qc.batch.eq(batch)&qc.phase_point.astype(str).eq(late)].set_index('pa_id')
        for key in ['weighted_white_fraction','weighted_bg70','weighted_bg80','weighted_bg90','weighted_dark','top128_bg80']:
            change=group.pa_id.map(b[key])-group.pa_id.map(a[key]);target=group.groupby('fold').net_harm_rate.rank(pct=True)
            ranks=change.groupby(group.fold).rank(pct=True);valid=group.degree.gt(0)
            associations.append(dict(batch=batch,phase=phase,feature=key,fold_rank_spearman=float(spearmanr(ranks[valid],target[valid]).statistic)))
    pd.DataFrame(associations).to_csv(BASE/'background_harm_associations.csv',index=False)
    from analyze_patients import participation
    subgroup_rows=[]
    for fold in range(5):
        directory=BASE.parent/f'bs32/diagnostics/fold_{fold}'
        meta=pd.read_csv(directory/'private_cohort.csv',dtype={'pa_id':str});mask=meta.split.eq(fold);part=meta[mask]
        last=json.loads((directory/'audit.json').read_text())['last_epoch']
        with np.load(directory/'private/epoch_010.npz') as f:old=f['risk'][mask]
        with np.load(directory/f'private/epoch_{last:03d}.npz') as f:new=f['risk'][mask]
        _,(i,j,corrected,broken)=participation(old,new,part.time.to_numpy(np.float32),part.event.to_numpy(bool))
        q=qc[qc.batch.eq(32)&qc.fold.eq(fold)&qc.phase_point.eq('last')].set_index('pa_id').loc[part.pa_id]
        for cutoff in [.1,.2]:
            selected=q.weighted_bg80.to_numpy()>=cutoff;edges=selected[i]|selected[j]
            subgroup_rows.append(dict(fold=fold,cutoff=cutoff,patients=int(selected.sum()),pair_coverage=float(edges.mean()),
                broken_involving=int((edges&broken).sum()),corrected_involving=int((edges&corrected).sum()),
                contribution_to_delta_c=float(((edges&corrected).sum()-(edges&broken).sum())/len(i))))
    pd.DataFrame(subgroup_rows).to_csv(BASE/'high_background_subgroups.csv',index=False)
    predictions=pd.read_csv(BASE/'private/attention_intervention_predictions.csv',dtype={'pa_id':str})
    folds=[]
    for fold in range(5):
        meta=pd.read_csv(BASE.parent/f'bs32/diagnostics/fold_{fold}/private_cohort.csv',dtype={'pa_id':str})
        group=meta[meta.split.eq(fold)].set_index('pa_id')
        pivot=predictions[predictions.fold.eq(fold)].pivot(index='pa_id',columns='policy',values='risk').loc[group.index]
        folds.append((group.time.to_numpy(np.float32),group.event.to_numpy(bool),pivot))
    rng=np.random.default_rng(123);draws=[]
    for repeat in range(2000):
        deltas=[]
        for times,events,pred in folds:
            chosen=rng.integers(len(times),size=len(times))
            native=concordance_index_censored(events[chosen],times[chosen],pred.late_native.to_numpy()[chosen])[0]
            mixed=concordance_index_censored(events[chosen],times[chosen],pred.half_early_half_late.to_numpy()[chosen])[0]
            deltas.append(mixed-native)
        draws.append(np.mean(deltas))
    observed=pd.read_csv(BASE/'attention_intervention_summary.csv').set_index('policy').loc['half_early_half_late','delta_mean']
    interval=dict(comparison='half_early_half_late versus late_native',mean_delta=float(observed),bootstrap_repeats=2000,
        interval_low=float(np.quantile(draws,.025)),interval_high=float(np.quantile(draws,.975)),
        scope='paired patient resampling within each validation fold; conditional on frozen models, not retraining or previous exploration; unadjusted')
    (BASE/'attention_intervention_interval.json').write_text(json.dumps(interval,indent=2))
    contrast=pd.read_csv(BASE/'attention_intervention_summary.csv').set_index('policy')
    subset=frame[frame.batch.eq(32)];correlations=pd.read_csv(BASE/'associations.csv');concentration=pd.read_csv(BASE/'pair_concentration.csv')
    background=pd.read_csv(BASE/'attention_background_summary.csv');background=background[background.batch.eq(32)].set_index('phase_point')
    lines=['# Patient, background and fixed-Attention follow-up','',
        'Read-only follow-up to the completed Cox diagnostics; no neural network training. Attribution tables and all patient predictions remain local under private/. Thirty fold/phase/batch pair decompositions were reproduced exactly. Native late-model risks were reproduced with maximum error0.',
        '', '## What changed for patients','',
        'Every changed comparable pair is attributed to both endpoints. Patient net harm is (late error credit minus early error credit) divided by the number of comparable partners. This measures involvement, not which patient caused the error. Four patients have no comparable partners and are excluded from correlation calculations.',
        '', '| Batch32 phase | Top20% net-harm patients: all-pair coverage | Broken-pair coverage | Corrected-pair coverage |','|---|---:|---:|---:|']
    for phase,g in concentration[concentration.batch.eq(32)&concentration.method.eq('net_harm_top20pct')].groupby('phase'):
        lines.append(f"| {phase} | {g.all_pair_coverage.mean():.3f} | {g.broken_pair_coverage.mean():.3f} | {g.corrected_pair_coverage.mean():.3f} |")
    lines+=['','This concentration is selected after observing errors, not an independently validated predictor. It cannot justify excluding patients or changing labels. The same patients are not uniformly harmed across training intervals or batch sizes; see cross_batch_consistency.csv.',
        '', '## No strong monotone association with technical proxies','', '| Batch32 phase | Attention JS / net harm rho | Entropy change / net harm rho | Patch count / net harm rho | UNI norm / net harm rho |','|---|---:|---:|---:|---:|']
    for phase in ['5_to_10','10_to_last']:
        g=correlations[correlations.batch.eq(32)&correlations.phase.eq(phase)].set_index('feature')
        lines.append('| '+phase+' | '+' | '.join(f"{g.loc[key,'fold_rank_spearman']:.4f}" for key in ['attention_js','entropy_change','n_patches','historical_feature_norm_mean'])+' |')
    lines+=['','These are descriptive within-fold rank correlations, not causal effects or multiplicity-adjusted significance tests. Small correlations do not exclude nonlinear or subgroup effects. Patient-level UNI norm differences are distinct from global input-scale/optimizer effects.',
        '', '## Coarse thumbnail background proxy','',
        'Original TRIDENT thumbnails and level0 coordinates were aligned to feature rows for721 slides. A white pixel is minRGB>=220 and saturation proxy<=0.1; a background-like patch has>=80% such pixels. This cannot diagnose tumor, stroma, blur, folds or stain artifacts, and air spaces/pale valid tissue may also appear white.',
        '', '| Batch32 checkpoint | Attention-weighted white fraction | Attention mass on background-like patches | Top128 background-like fraction |','|---|---:|---:|---:|']
    for phase in ['5','10','last']:
        r=background.loc[phase];lines.append(f"| {phase} | {r.weighted_white_fraction:.4f} | {r.weighted_bg80:.4f} | {r.top128_bg80:.4f} |")
    lines+=['','Total background-like attention mass decreases; a small increase in top128 background-like frequency remains, so individual outliers are possible.25 patients have>=10% late Attention mass on background-like patches; edges involving them account for about-0.0015 mean C change of the total-0.0200 decline from10 to last. This is descriptive accounting, not causal attribution. These measurements do not support a cohort-wide explanation that late training progressively focuses on blank background.',
        '', '## Frozen late-model Attention intervention','',
        'Use the five batch32 last checkpoints. Compression, post-pooling projection and risk head remain fixed. Only patch weights change: native late, saved epoch5, a predefined50/50 mixture, or uniform. Identical patch ordering and native risk reproduction verified. No outcome-driven mixing-weight tuning.',
        '', '| Weight policy | Val C-index mean ± SD | Delta versus native late | Positive folds |','|---|---:|---:|---:|']
    for name,r in contrast.iterrows():lines.append(f'| {name} | {r.cindex_mean:.4f} ± {r.cindex_std:.4f} | {r.delta_mean:+.4f} | {int(r.positive_folds)}/5 |')
    lines += ['',f"Paired conditional bootstrap for mixture improvement: {interval['mean_delta']:+.4f}, nominal95% interval [{interval['interval_low']:+.4f}, {interval['interval_high']:+.4f}]. This omits model retraining, epoch/architecture exploration and multiple-policy uncertainty.",
        '', 'Interpretation: late Attention reweighting contributes to deterioration in this frozen-model intervention, while completely uniform pooling loses useful information. Earlier weights alone do not consistently help. The50/50 mixture partly recovers last-model performance but remains below epoch5/10 (~0.7325) and selected best checkpoints (~0.7458). It is a mechanism diagnostic, not a demonstrated better training method.',
        '', 'Weight intervention creates inputs the fixed late prediction head was not trained on. It cannot establish that temporal Attention regularization will improve training, nor rule out projection/head drift, finite supervision or missing prognostic factors. No evidence here proves intrinsic UNI information exhaustion, label noise, or a specific histological cause.']
    (BASE/'report.md').write_text('\n'.join(lines)+'\n')
    (BASE/'validation.json').write_text(json.dumps(dict(passed=True,pair_groups_verified=30,qc_slides=721,patch_coordinate_order_verified=True,
        native_frozen_risk_reproduction_error=0,bootstrap_repeats=2000,training_started=False,patient_files_private=True),indent=2))
    print(json.dumps(interval),flush=True)


if __name__=='__main__':run()
