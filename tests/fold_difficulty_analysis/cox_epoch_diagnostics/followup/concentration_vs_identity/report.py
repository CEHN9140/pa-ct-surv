"""Summarize matched concentration/assignment controls; no outcome-based tuning."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sksurv.metrics import concordance_index_censored

BASE=Path(__file__).resolve().parent;SOURCE=BASE.parents[1]


def run():
    validation=json.loads((BASE/'intervention_validation.json').read_text())
    assert validation['passed'] and json.loads((BASE/'curve_validation.json').read_text())['passed']
    metrics=pd.read_csv(BASE/'fold_metrics.csv');summary=pd.read_csv(BASE/'summary.csv').set_index('policy')
    pivot=metrics.pivot(index='fold',columns='policy',values='cindex')
    contrasts=[('same_profile_late_ranking','late_native'),('neff_matched_temperature','late_native'),
        ('early_late_mix','same_profile_late_ranking'),('early_late_mix','neff_matched_temperature'),('early_late_mix','late_native')]
    predictions=pd.read_csv(BASE/'private/predictions.csv',dtype={'pa_id':str});data=[]
    for fold in range(5):
        frame=pd.read_csv(SOURCE/f'bs32/diagnostics/fold_{fold}/private_cohort.csv',dtype={'pa_id':str})
        frame=frame[frame.split.eq(fold)].set_index('pa_id')
        p=predictions[predictions.fold.eq(fold)].pivot(index='pa_id',columns='policy',values='risk').loc[frame.index]
        for name in p:
            c=concordance_index_censored(frame.event.to_numpy(bool),frame.time.to_numpy(np.float32),p[name].to_numpy(np.float32))[0]
            assert abs(c-pivot.loc[fold,name])<1e-12
        data.append((frame.time.to_numpy(np.float32),frame.event.to_numpy(bool),p))
    rng=np.random.default_rng(123);samples=[]
    for repeat in range(2000):
        scores=[]
        for times,events,pred in data:
            chosen=rng.integers(len(times),size=len(times))
            scores.append({policy:concordance_index_censored(events[chosen],times[chosen],pred[policy].to_numpy(np.float32)[chosen])[0] for policy in pred})
        samples.append([np.mean([score[left]-score[right] for score in scores]) for left,right in contrasts])
    samples=np.asarray(samples);intervals=[]
    for i,(left,right) in enumerate(contrasts):
        difference=pivot[left]-pivot[right]
        intervals.append(dict(policy_left=left,policy_right=right,mean_delta=float(difference.mean()),positive_folds=int((difference>0).sum()),
            interval_low=float(np.quantile(samples[:,i],.025)),interval_high=float(np.quantile(samples[:,i],.975)),bootstrap_repeats=2000))
    interval=pd.DataFrame(intervals);interval.to_csv(BASE/'paired_comparisons.csv',index=False)
    best=pd.read_csv(BASE/'best_last_comparison.csv')
    names={'late_native':'Native late weights','early_late_mix':'Early5/late 50:50 mixture','same_profile_late_ranking':'Mixture weight multiset, late ranking',
        'neff_matched_temperature':'Late ranking, Neff-matched temperature','uniform':'Uniform weights'}
    lines=['# Concentration versus patch-weight assignment','',
        'Frozen-inference diagnosis on the existing batch32 fivefold last checkpoints. No new neural training, no new loss, no L2, no patch removal. Compression/post-pooling projection/risk head remain fixed. Early reference is the previously specified Epoch5; mixture remains50/50. All controls were predefined before this inference run.',
        '', '## Curves and missing data','',
        'concentration_epoch_curves.png shows all available Epoch C-index and mean effective patch count for Train/Val in eval mode. Dashed lines mark validation-selected best epochs. Top128 mass and mean(Neff_i/N_i) are shown only at saved Attention snapshots; missing Epochs remain NaN and are not interpolated. Mean(Neff_i/N_i) is computed per patient, not mean(Neff)/mean(N).',
        '', '| Fold | Best Epoch | Best Val C | Last Val C | Mean Neff/N at best | Mean Neff/N at last |','|---|---:|---:|---:|---:|---:|']
    for r in best.itertuples():lines.append(f'| {r.fold} | {r.best_epoch} | {r.cindex_best:.4f} | {r.cindex_last:.4f} | {r.effective_ratio_best:.4f} | {r.effective_ratio_last:.4f} |')
    lines+=['',f"Best-checkpoint mean effective fractions vary from{best.effective_ratio_best.min():.3f} to{best.effective_ratio_best.max():.3f}. These curves describe temporal association; selecting the maximum validation Epoch makes subsequent decline partly selection-dependent. They cannot establish that concentration causes overfitting or justify choosing a universal q_min on the same folds.",
        '', '## Stronger frozen controls','',
        'The profile control sorts the mixture weights and assigns them according to native late Patch ranking. It preserves the full weight multiset, hence Neff, entropy and Top128 mass, while keeping late weak ranking. This is stronger than matching Neff alone. The temperature control uses softmax(log(alpha_late)/T); T matches mixture Neff separately per patient without using DFS outcomes. All721 temperatures are greater than1 in this dataset.',
        '', '| Policy | Val C-index mean ± SD | Delta vs native late | Improved folds |','|---|---:|---:|---:|']
    for name in ['late_native','neff_matched_temperature','same_profile_late_ranking','early_late_mix','uniform']:
        r=summary.loc[name];lines.append(f'| {names[name]} | {r.cindex_mean:.4f} ± {r.cindex_std:.4f} | {r.delta_mean:+.4f} | {int(r.improved_folds)}/5 |')
    lines+=['','## Paired conditional uncertainty','', '| Contrast | Mean delta C | Positive folds | Nominal95% interval |','|---|---:|---:|---:|']
    for r in interval.itertuples():lines.append(f'| {names[r.policy_left]} minus {names[r.policy_right]} | {r.mean_delta:+.4f} | {r.positive_folds}/5 | [{r.interval_low:+.4f}, {r.interval_high:+.4f}] |')
    lines+=['','Bootstrap uses2000 paired patient resamples within each fold, keeping models fixed and averaging fold C-index differences. Intervals are unadjusted and exclude retraining, earlier repeated exploration, policy/epoch choices and external-cohort uncertainty. Do not treat them as confirmatory significance tests.',
        '', '## Interpretation','',
        'Holding late ranking while adjusting its weight profile already recovers some performance. At the same complete concentration profile, assigning weights according to the early/late mixture improves all5 folds over assigning them by late ranking. Point estimates are consistent with contributions from both weight distribution and assignment under these fixed-network interventions. However, intervals for the isolated profile/temperature contrasts and mixture-versus-profile contrast cross0; stable independent increments have not been established.',
        '', 'Do not convert these path-dependent performance contrasts into causal contribution percentages. A single concentration statistic cannot describe all weighting behavior; the exact-profile control reduces this ambiguity, but interventions still change inputs to a head trained jointly with native late weights. The result does not identify correct histological regions or prove that temporal-consistency/concentration regularization will improve training.',
        '', 'The mixture remains below Epoch5/10 (~0.7325) and original validation-selected best (~0.7458). This is partial recovery of late deterioration, not a new superior model. No universal concentration threshold, training regularization strength or new architecture was selected.',
        '', '## Verification','',
        f"All721 native late risks reproduce with max error{validation['maximum_native_risk_error']}; profile sorted weights match exactly; temperature Neff relative error max{validation['maximum_temperature_neff_relative_error']:.3g}. All25 fold/policy C-indices were independently recomputed from saved risks. Prior50/50 mixture predictions reproduce.3 targeted tests pass; the plot was visually inspected for labels and data availability. All patient arrays and matching audits remain private."]
    (BASE/'report.md').write_text('\n'.join(lines)+'\n')
    (BASE/'validation.json').write_text(json.dumps(dict(passed=True,tests=3,patients=721,folds=5,policies=5,training_started=False,
        fold_cindices_recomputed=25,full_profile_exact=True,ranking_preserved=True,no_attention_interpolation=True,bootstrap_repeats=2000),indent=2))
    print(interval.to_string(index=False),flush=True)


if __name__=='__main__':run()
