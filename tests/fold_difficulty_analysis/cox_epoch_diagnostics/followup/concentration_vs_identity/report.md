# Concentration versus patch-weight assignment

Frozen-inference diagnosis on the existing batch32 fivefold last checkpoints. No new neural training, no new loss, no L2, no patch removal. Compression/post-pooling projection/risk head remain fixed. Early reference is the previously specified Epoch5; mixture remains50/50. All controls were predefined before this inference run.

## Curves and missing data

concentration_epoch_curves.png shows all available Epoch C-index and mean effective patch count for Train/Val in eval mode. Dashed lines mark validation-selected best epochs. Top128 mass and mean(Neff_i/N_i) are shown only at saved Attention snapshots; missing Epochs remain NaN and are not interpolated. Mean(Neff_i/N_i) is computed per patient, not mean(Neff)/mean(N).

| Fold | Best Epoch | Best Val C | Last Val C | Mean Neff/N at best | Mean Neff/N at last |
|---|---:|---:|---:|---:|---:|
| 0 | 13 | 0.6902 | 0.6757 | 0.1041 | 0.0292 |
| 1 | 13 | 0.7559 | 0.7339 | 0.0936 | 0.0196 |
| 2 | 5 | 0.7908 | 0.7288 | 0.5871 | 0.0837 |
| 3 | 4 | 0.7719 | 0.7310 | 0.5930 | 0.0801 |
| 4 | 10 | 0.7202 | 0.6931 | 0.2198 | 0.0333 |

Best-checkpoint mean effective fractions vary from0.094 to0.593. These curves describe temporal association; selecting the maximum validation Epoch makes subsequent decline partly selection-dependent. They cannot establish that concentration causes overfitting or justify choosing a universal q_min on the same folds.

## Stronger frozen controls

The profile control sorts the mixture weights and assigns them according to native late Patch ranking. It preserves the full weight multiset, hence Neff, entropy and Top128 mass, while keeping late weak ranking. This is stronger than matching Neff alone. The temperature control uses softmax(log(alpha_late)/T); T matches mixture Neff separately per patient without using DFS outcomes. All721 temperatures are greater than1 in this dataset.

| Policy | Val C-index mean ± SD | Delta vs native late | Improved folds |
|---|---:|---:|---:|
| Native late weights | 0.7125 ± 0.0265 | +0.0000 | 0/5 |
| Late ranking, Neff-matched temperature | 0.7183 ± 0.0292 | +0.0058 | 4/5 |
| Mixture weight multiset, late ranking | 0.7177 ± 0.0312 | +0.0052 | 4/5 |
| Early5/late 50:50 mixture | 0.7220 ± 0.0318 | +0.0095 | 5/5 |
| Uniform weights | 0.6998 ± 0.0443 | -0.0127 | 2/5 |

## Paired conditional uncertainty

| Contrast | Mean delta C | Positive folds | Nominal95% interval |
|---|---:|---:|---:|
| Mixture weight multiset, late ranking minus Native late weights | +0.0052 | 4/5 | [-0.0034, +0.0137] |
| Late ranking, Neff-matched temperature minus Native late weights | +0.0058 | 4/5 | [-0.0013, +0.0130] |
| Early5/late 50:50 mixture minus Mixture weight multiset, late ranking | +0.0043 | 5/5 | [-0.0004, +0.0092] |
| Early5/late 50:50 mixture minus Late ranking, Neff-matched temperature | +0.0037 | 3/5 | [-0.0012, +0.0086] |
| Early5/late 50:50 mixture minus Native late weights | +0.0095 | 5/5 | [+0.0007, +0.0177] |

Bootstrap uses2000 paired patient resamples within each fold, keeping models fixed and averaging fold C-index differences. Intervals are unadjusted and exclude retraining, earlier repeated exploration, policy/epoch choices and external-cohort uncertainty. Do not treat them as confirmatory significance tests.

## Interpretation

Holding late ranking while adjusting its weight profile already recovers some performance. At the same complete concentration profile, assigning weights according to the early/late mixture improves all5 folds over assigning them by late ranking. Point estimates are consistent with contributions from both weight distribution and assignment under these fixed-network interventions. However, intervals for the isolated profile/temperature contrasts and mixture-versus-profile contrast cross0; stable independent increments have not been established.

Do not convert these path-dependent performance contrasts into causal contribution percentages. A single concentration statistic cannot describe all weighting behavior; the exact-profile control reduces this ambiguity, but interventions still change inputs to a head trained jointly with native late weights. The result does not identify correct histological regions or prove that temporal-consistency/concentration regularization will improve training.

The mixture remains below Epoch5/10 (~0.7325) and original validation-selected best (~0.7458). This is partial recovery of late deterioration, not a new superior model. No universal concentration threshold, training regularization strength or new architecture was selected.

## Verification

All721 native late risks reproduce with max error0.0; profile sorted weights match exactly; temperature Neff relative error max6.98e-08. All25 fold/policy C-indices were independently recomputed from saved risks. Prior50/50 mixture predictions reproduce.3 targeted tests pass; the plot was visually inspected for labels and data availability. All patient arrays and matching audits remain private.
