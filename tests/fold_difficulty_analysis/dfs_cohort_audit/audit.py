"""Read-only label provenance and matched four-seed DFS pair audit; no training."""
from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from sksurv.metrics import concordance_index_censored

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE.parent / 'attention_risk_attribution'))
from attribution import comparable_pairs


def ids(values):
    return values.astype(str).str.strip().str.replace(r'\.0$', '', regex=True)


def pairs_for(time, event, risk):
    time = np.asarray(time); event = np.asarray(event, bool); risk = np.asarray(risk)
    a, b = comparable_pairs(time, event)
    margin = risk[a] - risk[b]
    kind = np.where(event[b], 'EE', 'EC')
    bins = np.asarray(pd.cut(time[a], [-np.inf, 12, 24, 36, np.inf],
                             labels=['0-12', '12-24', '24-36', '>36']))
    return pd.DataFrame(dict(anchor=a, other=b, credit=np.where(abs(margin) <= 1e-8, .5, (margin > 0).astype(float)),
                             pair_type=kind, time_bin=bins, stratum=[f'{k}:{t}' for k, t in zip(kind, bins)]))


def label_sources(c, private, out):
    main = pd.read_excel('/home/gly001/cqj/data/dz/HE预后.xlsx', sheet_name='工作表1')
    main['pa_id'] = ids(main['病理号']); main['ct_id'] = ids(main['ID'])
    m = c.merge(main[['pa_id', 'ct_id', 'DFS.months', 'DFS.status']], on=['pa_id', 'ct_id'], how='left', validate='one_to_one')
    main_ok = (m['DFS.months'].notna() & np.isclose(m.time, m['DFS.months'], atol=1e-5, rtol=0) & (m.event == m['DFS.status']))
    m.loc[~main_ok].to_csv(private / 'main_source_discrepancies.csv', index=False)
    raw = pd.read_excel('/home/gly001/cqj/data/dz/预后label.xlsx', sheet_name='Sheet1')
    raw['ct_id'] = ids(raw.ID)
    ambiguous = raw.ct_id.duplicated(False)
    c[c.ct_id.isin(raw.loc[ambiguous, 'ct_id'])].to_csv(private / 'ambiguous_day_source_matches.csv', index=False)
    day = c.merge(raw.loc[~ambiguous], on='ct_id', how='inner', validate='one_to_one')
    day['path_id_agrees'] = day.pa_id == ids(day['Path ID'])
    day.loc[~day.path_id_agrees].to_csv(private / 'day_source_identity_review.csv', index=False)
    ct_matches = day.copy()
    day = day.loc[day.path_id_agrees].copy()
    day['day_month_delta'] = pd.to_numeric(day['DFS.days']) / 30 - day.time
    day['event_disagreement'] = day.event != pd.to_numeric(day['DFS.status'])
    day['time_disagreement'] = day.day_month_delta.abs() > .1
    day['death_with_dfs_censored'] = (day['OS.status'] == 1) & (day.event == 0)
    day.loc[day.event_disagreement | day.time_disagreement | day.death_with_dfs_censored].to_csv(private / 'day_source_review.csv', index=False)
    rows = []
    for fold, g in c.groupby('split'):
        d = day[day.split == fold]
        rows.append(dict(fold=int(fold), main_matched=int(main_ok[m.split == fold].sum()), n=len(g), day_unique_matches=len(d), day_ct_only_matches=int(((ct_matches.split == fold) & ~ct_matches.path_id_agrees).sum()),
                         day_ambiguous_matches=int(g.ct_id.isin(raw.loc[ambiguous, 'ct_id']).sum()),
                         day_event_disagreements=int(d.event_disagreement.sum()), day_time_disagreements=int(d.time_disagreement.sum()),
                         death_with_dfs_censored=int(d.death_with_dfs_censored.sum())))
    pd.DataFrame(rows).to_csv(out / 'source_consistency.csv', index=False)
    return pd.DataFrame(rows)


def run():
    out = HERE / 'results'; private = HERE / 'private'
    out.mkdir(exist_ok=True); private.mkdir(exist_ok=True)
    c = pd.read_csv(ROOT / 'data/trident/features_uni_v1/all_label_roi64.csv', dtype={'pa_id': str, 'ct_id': str})
    assert len(c) == 721 and c.event.sum() == 251 and c.pa_id.is_unique and c.ct_id.is_unique
    meta = pd.read_csv(HERE.parent / 'cohort.csv', dtype={'pa_id': str, 'ct_id': str})
    c = c.merge(meta[['pa_id', 'ct_id', 'source_batch', 'n_patches', 'time', 'event', 'split']],
                on=['pa_id', 'ct_id'], how='left', validate='one_to_one', suffixes=('', '_metadata'))
    assert np.array_equal(c.time.to_numpy(np.float32), c.time_metadata.to_numpy(np.float32))
    assert (c.event == c.event_metadata).all() and (c.split == c.split_metadata).all()
    assert c.source_batch.notna().all()
    source = label_sources(c, private, out)
    distributions = []; metrics = []; strata = []; persistence = []
    for fold, g in c.groupby('split'):
        g = g.reset_index(drop=True); time = g.time.to_numpy(np.float32); event = g.event.to_numpy(bool)
        censor_km = KaplanMeierFitter().fit(time, event_observed=~event)
        distributions.append(dict(fold=fold, n=len(g), events=int(event.sum()), event_median=float(np.median(time[event])),
                                  censor_median=float(np.median(time[~event])), reverse_km_followup_median=float(censor_km.median_survival_time_),
                                  at_risk_36=int((time >= 36).sum()), at_risk_60=int((time >= 60).sum()),
                                  events_le12=int((event & (time <= 12)).sum()), censored_le24=int((~event & (time <= 24)).sum()),
                                  patch_median=float(g.n_patches.median())))
        credits = []
        for seed in [123, 7, 42, 2026]:
            base = HERE.parent / ('module_freeze' if seed == 123 else f'module_freeze_repeats/seed_{seed}')
            p = pd.read_csv(base / f'fold_{fold}/private/outer/baseline/results/val_predictions.csv', dtype={'case_id': str}).set_index('case_id')
            assert p.index.is_unique and set(p.index) == set(g.pa_id)
            p = p.loc[g.pa_id]
            assert np.array_equal(p['dfs.month'].to_numpy(np.float32), time)
            assert np.array_equal(p['dfs.status'].to_numpy(bool), event) and np.isfinite(p.risk_score).all()
            pairs = pairs_for(time, event, p.risk_score.to_numpy())
            score = concordance_index_censored(event, time, p.risk_score.to_numpy())[0]
            assert abs(score - pairs.credit.mean()) < 1e-12
            metrics.append(dict(fold=fold, seed=seed, cindex=score, n_pairs=len(pairs)))
            credits.append(pairs.credit.to_numpy())
            for name, s in pairs.groupby('stratum'):
                strata.append(dict(fold=fold, seed=seed, stratum=name, n_pairs=len(s), cindex=s.credit.mean()))
        arr = np.array(credits)
        pairs['always_wrong'] = (arr == 0).all(axis=0)
        pairs['always_correct'] = (arr == 1).all(axis=0)
        pairs['mean_credit'] = arr.mean(axis=0)
        pairs['wrong_seeds'] = (arr == 0).sum(axis=0)
        pairs['anchor_id'] = g.pa_id.to_numpy()[pairs.anchor]
        pairs['other_id'] = g.pa_id.to_numpy()[pairs.other]
        pairs.to_csv(private / f'fold_{fold}_pair_persistence.csv', index=False)
        for name, s in pairs.groupby('stratum'):
            persistence.append(dict(fold=fold, stratum=name, n_pairs=len(s), always_wrong_rate=s.always_wrong.mean(),
                                    always_correct_rate=s.always_correct.mean(), mean_credit=s.mean_credit.mean()))
        a = pairs.anchor.to_numpy(); b = pairs.other.to_numpy()
        degree = np.bincount(np.r_[a,b], minlength=len(g)); wrong = pairs.always_wrong.to_numpy(float)
        counts = np.bincount(np.r_[a,b], weights=np.r_[wrong,wrong], minlength=len(g))
        pd.DataFrame(dict(case_id=g.pa_id, comparable_degree=degree, persistent_wrong_pairs=counts,
                          persistent_wrong_rate=np.divide(counts, degree, out=np.full(len(g), np.nan), where=degree > 0))).to_csv(private / f'fold_{fold}_patient_review.csv', index=False)
    dist = pd.DataFrame(distributions); met = pd.DataFrame(metrics); st = pd.DataFrame(strata)
    # Label-based reference composition, counted once per fold (not once per training seed).
    reference = st[st.seed == 123].groupby('stratum').n_pairs.sum(); reference = reference / reference.sum()
    for i, row in met.iterrows():
        s = st[(st.fold == row.fold) & (st.seed == row.seed)].set_index('stratum')
        met.loc[i, 'reference_coverage'] = reference.reindex(s.index).sum()
        met.loc[i, 'composition_standardized_cindex'] = (s.cindex * reference).sum() if set(reference.index) <= set(s.index) else np.nan
    for frame, name in [(dist,'fold_distributions'),(met,'seed_fold_metrics'),(st,'pair_strata'),(pd.DataFrame(persistence),'pair_persistence')]:
        frame.to_csv(out / f'{name}.csv', index=False)
    c.groupby(['split','source_batch'], dropna=False).agg(n=('event','size'), events=('event','sum')).to_csv(out / 'batch_composition.csv')
    summary = met.groupby('fold').agg(cindex_mean=('cindex','mean'), cindex_seed_sd=('cindex','std'), standardized_mean=('composition_standardized_cindex','mean'))
    summary.to_csv(out / 'fold_summary.csv')
    persistent = pd.DataFrame(persistence)
    totals = persistent.assign(always_wrong_pairs=persistent.n_pairs*persistent.always_wrong_rate).groupby('fold')[['n_pairs','always_wrong_pairs']].sum()
    totals['always_wrong_rate'] = totals.always_wrong_pairs / totals.n_pairs
    totals.to_csv(out / 'persistent_error_summary.csv')
    types = persistent.assign(pair_type=persistent.stratum.str.split(':').str[0], credit=persistent.n_pairs*persistent.mean_credit).groupby(['fold','pair_type'])[['n_pairs','credit']].sum()
    types['cindex'] = types.credit / types.n_pairs
    types.to_csv(out / 'pair_type_summary.csv')
    report = f'''# DFS cohort and four-seed Baseline audit

No training, label changes, new splits, or patient exclusions. Models: matched Baseline only; seeds 123/7/42/2026. All 20 prediction sets align with the original 721 patients / 251 events. All pair credits reproduce scikit-survival C-index to <1e-12. Risks are never pooled across folds.

## Fold difficulty (mean over four training seeds)

```text
{summary.to_string()}
```

## DFS/follow-up composition

```text
{dist.to_string(index=False)}
```

Standardized C-index uses the pooled label-based eight-stratum reference (event-event / event-censored × event anchor time ≤12, 12–24, 24–36, >36 months). This is a descriptive composition adjustment, not a causal effect or replacement performance metric. No weights use model performance. Missing positive-weight strata yield NaN rather than silent renormalization.

## Raw label provenance

```text
{source.to_string(index=False)}
```

The primary HE预后.xlsx table matches {source.main_matched.sum()}/721 current records. The separate day-based table uniquely matches both CT and pathology IDs for {source.day_unique_matches.sum()} patients; another {source.day_ct_only_matches.sum()} CT matches have different pathology IDs and are excluded from label comparison; {source.day_ambiguous_matches.sum()} current patients have ambiguous duplicate CT IDs and are excluded only from this source comparison. Among unique matches, {source.day_event_disagreements.sum()} event codes differ and {source.day_time_disagreements.sum()} times differ by >0.1 months when comparing days/30. One month = 30 days is a diagnostic conversion assumption. Detailed records remain private; neither source is declared correct or used to overwrite training labels.

There are {source.death_with_dfs_censored.sum()} overlapping patients with OS death and current DFS censoring. Without endpoint definitions and follow-up dates, this does not establish an error: recurrence/death inclusion and censoring conventions cannot be clinically verified from these tables. Raw source versions and update dates are undocumented. Stage, subtype, grade, treatment, hospital and scanner fields are unavailable in inspected sources. Available source_batch is a file-collection batch, not hospital identity.

## Persistent pair errors

```text
{totals.to_string()}
```

pair_persistence.csv reports pairs wrong in all four Baseline runs, by fold and stratum. These are the same patients repeatedly evaluated, not four independent cohorts. Pair endpoints overlap; counts are dependent. Patient and pair review lists are private. Persistent errors alone do not imply incorrect labels or irrecoverable image information.

## Existing comparisons: do not repeat

Stage 5 already performed full-patch MeanPool + standardized linear Ridge-Cox (no MLP): nested C-index 0.7197. Stage 6 matched nested ABMIL: 0.7332; paired difference +0.0135 with conditional interval [-0.0010,+0.0282]. This does not establish a UNI information ceiling.

Experiment 10 already explored 128/256/384/full training patients, with historical best-validation means approximately 0.7238/0.7275/0.7350/0.7452. Those historical checkpoints used Outer Val selection, so this trend is exploratory and cannot substitute for strictly nested sample-size inference. No new learning-curve training is launched.

## Main findings

```text
{types.to_string()}
```

Across folds, event-event C-index is only 0.5535–0.6092. The overall advantage of Folds 1/2/3 mainly comes from event-censored pairs (approximately 0.808/0.839/0.808 versus 0.717/0.737 in Folds 0/4). Censored comparators are not confirmed permanent non-recurrence cases. This decomposition cannot establish time-specific AUC, treatment effects, or causal recurrence mechanisms.

Fold 0/4 remain harder across four seeds (mean C-index 0.6890/0.6921, training-seed SD 0.0095/0.0089). Descriptive pair-composition standardization leaves them at 0.6886/0.6956. Measured time and pair-type composition does not close the gap; this is not proof of equal distributions or of a specific cause. Fold 4 has later events (median 36.1 months; only five events by month 12). Across seeds, 23.9%/22.4% of comparable pairs in Fold 0/4 are always misranked, versus 15.2%–20.0% in the other folds. These persistent errors warrant review but are not a label-noise diagnosis.

## Interpretation limits

Fold composition and recurrent errors can identify diagnostic priorities but cannot establish whether limited events, omitted clinical variables, representation, or model fitting is the root cause. Previous Outer Val exploration and training seed reuse remain. Do not filter patients or patches using these validation errors and claim an independently validated improvement. This audit does not justify changing the endpoint or official splits.
'''
    (out / 'report.md').write_text('\n'.join(line.rstrip() for line in report.splitlines()) + '\n')
    (out / 'validation.json').write_text(json.dumps(dict(training_started=False, canonical_patients=len(c), events=int(c.event.sum()), matched_prediction_sets=len(met), seeds=[123,7,42,2026], pair_cindex_replay_tolerance=1e-12, main_source_matches=int(source.main_matched.sum()), all_strata_complete=bool((met.reference_coverage > 1-1e-12).all()), patient_outputs_private=True, original_labels_changed=False), indent=2))
    print(summary.to_string()); print(source.sum(numeric_only=True).to_string())


if __name__ == '__main__':
    run()
