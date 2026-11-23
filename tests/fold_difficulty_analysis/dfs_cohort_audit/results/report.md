# DFS cohort and four-seed Baseline audit

No training, label changes, new splits, or patient exclusions. Models: matched Baseline only; seeds 123/7/42/2026. All 20 prediction sets align with the original 721 patients / 251 events. All pair credits reproduce scikit-survival C-index to <1e-12. Risks are never pooled across folds.

## Fold difficulty (mean over four training seeds)

```text
      cindex_mean  cindex_seed_sd  standardized_mean
fold
0        0.689017        0.009516           0.688612
1        0.752501        0.004604           0.746583
2        0.771284        0.011673           0.767820
3        0.761706        0.007470           0.740529
4        0.692139        0.008909           0.695648
```

## DFS/follow-up composition

```text
 fold   n  events  event_median  censor_median  reverse_km_followup_median  at_risk_36  at_risk_60  events_le12  censored_le24  patch_median
    0 145      51     24.633333      65.550003                        68.0         100          77           16              9        3160.0
    1 144      50     17.100000      63.033333                        65.0          93          64           16              7        3065.0
    2 144      50     22.116667      63.849998                        66.0         102          69           16              9        2878.0
    3 144      50     21.799999      67.500000                        71.0         108          78           19              3        3004.5
    4 144      50     36.099998      61.000000                        63.0         104          62            5              7        3107.5
```

Standardized C-index uses the pooled label-based eight-stratum reference (event-event / event-censored × event anchor time ≤12, 12–24, 24–36, >36 months). This is a descriptive composition adjustment, not a causal effect or replacement performance metric. No weights use model performance. Missing positive-weight strata yield NaN rather than silent renormalization.

## Raw label provenance

```text
 fold  main_matched   n  day_unique_matches  day_ct_only_matches  day_ambiguous_matches  day_event_disagreements  day_time_disagreements  death_with_dfs_censored
    0           145 145                  54                    0                      0                        0                       1                        0
    1           144 144                  54                    0                      0                        0                       1                        1
    2           144 144                  56                    0                      0                        0                       1                        0
    3           144 144                  43                    0                      1                        1                       1                        0
    4           144 144                  49                    2                      0                        1                       1                        0
```

The primary HE预后.xlsx table matches 721/721 current records. The separate day-based table uniquely matches both CT and pathology IDs for 256 patients; another 2 CT matches have different pathology IDs and are excluded from label comparison; 1 current patients have ambiguous duplicate CT IDs and are excluded only from this source comparison. Among unique matches, 2 event codes differ and 5 times differ by >0.1 months when comparing days/30. One month = 30 days is a diagnostic conversion assumption. Detailed records remain private; neither source is declared correct or used to overwrite training labels.

There are 1 overlapping patients with OS death and current DFS censoring. Without endpoint definitions and follow-up dates, this does not establish an error: recurrence/death inclusion and censoring conventions cannot be clinically verified from these tables. Raw source versions and update dates are undocumented. Stage, subtype, grade, treatment, hospital and scanner fields are unavailable in inspected sources. Available source_batch is a file-collection batch, not hospital identity.

## Persistent pair errors

```text
      n_pairs  always_wrong_pairs  always_wrong_rate
fold
0        5149              1232.0           0.239270
1        5298              1059.0           0.199887
2        5133               779.0           0.151763
3        5275               978.0           0.185403
4        4796              1076.0           0.224354
```

pair_persistence.csv reports pairs wrong in all four Baseline runs, by fold and stratum. These are the same patients repeatedly evaluated, not four independent cohorts. Pair endpoints overlap; counts are dependent. Patient and pair review lists are private. Persistent errors alone do not imply incorrect labels or irrecoverable image information.

## Existing comparisons: do not repeat

Stage 5 already performed full-patch MeanPool + standardized linear Ridge-Cox (no MLP): nested C-index 0.7197. Stage 6 matched nested ABMIL: 0.7332; paired difference +0.0135 with conditional interval [-0.0010,+0.0282]. This does not establish a UNI information ceiling.

Experiment 10 already explored 128/256/384/full training patients, with historical best-validation means approximately 0.7238/0.7275/0.7350/0.7452. Those historical checkpoints used Outer Val selection, so this trend is exploratory and cannot substitute for strictly nested sample-size inference. No new learning-curve training is launched.

## Main findings

```text
                n_pairs   credit    cindex
fold pair_type
0    EC            3875  2776.50  0.716516
     EE            1274   771.25  0.605377
1    EC            4077  3294.75  0.808131
     EE            1221   692.00  0.566749
2    EC            3909  3281.50  0.839473
     EE            1224   677.50  0.553513
3    EC            4052  3273.00  0.807749
     EE            1223   745.00  0.609158
4    EC            3573  2632.00  0.736636
     EE            1223   687.50  0.562142
```

Across folds, event-event C-index is only 0.5535–0.6092. The overall advantage of Folds 1/2/3 mainly comes from event-censored pairs (approximately 0.808/0.839/0.808 versus 0.717/0.737 in Folds 0/4). Censored comparators are not confirmed permanent non-recurrence cases. This decomposition cannot establish time-specific AUC, treatment effects, or causal recurrence mechanisms.

Fold 0/4 remain harder across four seeds (mean C-index 0.6890/0.6921, training-seed SD 0.0095/0.0089). Descriptive pair-composition standardization leaves them at 0.6886/0.6956. Measured time and pair-type composition does not close the gap; this is not proof of equal distributions or of a specific cause. Fold 4 has later events (median 36.1 months; only five events by month 12). Across seeds, 23.9%/22.4% of comparable pairs in Fold 0/4 are always misranked, versus 15.2%–20.0% in the other folds. These persistent errors warrant review but are not a label-noise diagnosis.

## Interpretation limits

Fold composition and recurrent errors can identify diagnostic priorities but cannot establish whether limited events, omitted clinical variables, representation, or model fitting is the root cause. Previous Outer Val exploration and training seed reuse remain. Do not filter patients or patches using these validation errors and claim an independently validated improvement. This audit does not justify changing the endpoint or official splits.
