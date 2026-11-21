# Patient, background and fixed-Attention follow-up

Read-only follow-up to the completed Cox diagnostics; no neural network training. Attribution tables and all patient predictions remain local under private/. Thirty fold/phase/batch pair decompositions were reproduced exactly. Native late-model risks were reproduced with maximum error0.

## What changed for patients

Every changed comparable pair is attributed to both endpoints. Patient net harm is (late error credit minus early error credit) divided by the number of comparable partners. This measures involvement, not which patient caused the error. Four patients have no comparable partners and are excluded from correlation calculations.

| Batch32 phase | Top20% net-harm patients: all-pair coverage | Broken-pair coverage | Corrected-pair coverage |
|---|---:|---:|---:|
| 10_to_last | 0.354 | 0.784 | 0.189 |
| 5_to_10 | 0.379 | 0.802 | 0.140 |

This concentration is selected after observing errors, not an independently validated predictor. It cannot justify excluding patients or changing labels. The same patients are not uniformly harmed across training intervals or batch sizes; see cross_batch_consistency.csv.

## No strong monotone association with technical proxies

| Batch32 phase | Attention JS / net harm rho | Entropy change / net harm rho | Patch count / net harm rho | UNI norm / net harm rho |
|---|---:|---:|---:|---:|
| 5_to_10 | -0.0990 | 0.0311 | -0.0734 | -0.0147 |
| 10_to_last | 0.0102 | -0.0101 | 0.0032 | -0.0233 |

These are descriptive within-fold rank correlations, not causal effects or multiplicity-adjusted significance tests. Small correlations do not exclude nonlinear or subgroup effects. Patient-level UNI norm differences are distinct from global input-scale/optimizer effects.

## Coarse thumbnail background proxy

Original TRIDENT thumbnails and level0 coordinates were aligned to feature rows for721 slides. A white pixel is minRGB>=220 and saturation proxy<=0.1; a background-like patch has>=80% such pixels. This cannot diagnose tumor, stroma, blur, folds or stain artifacts, and air spaces/pale valid tissue may also appear white.

| Batch32 checkpoint | Attention-weighted white fraction | Attention mass on background-like patches | Top128 background-like fraction |
|---|---:|---:|---:|
| 5 | 0.0768 | 0.0310 | 0.0025 |
| 10 | 0.0515 | 0.0219 | 0.0065 |
| last | 0.0444 | 0.0165 | 0.0097 |

Total background-like attention mass decreases; a small increase in top128 background-like frequency remains, so individual outliers are possible.25 patients have>=10% late Attention mass on background-like patches; edges involving them account for about-0.0015 mean C change of the total-0.0200 decline from10 to last. This is descriptive accounting, not causal attribution. These measurements do not support a cohort-wide explanation that late training progressively focuses on blank background.

## Frozen late-model Attention intervention

Use the five batch32 last checkpoints. Compression, post-pooling projection and risk head remain fixed. Only patch weights change: native late, saved epoch5, a predefined50/50 mixture, or uniform. Identical patch ordering and native risk reproduction verified. No outcome-driven mixing-weight tuning.

| Weight policy | Val C-index mean ± SD | Delta versus native late | Positive folds |
|---|---:|---:|---:|
| early5_weights | 0.7185 ± 0.0407 | +0.0060 | 2/5 |
| half_early_half_late | 0.7220 ± 0.0318 | +0.0095 | 5/5 |
| late_native | 0.7125 ± 0.0265 | +0.0000 | 0/5 |
| uniform_weights | 0.6998 ± 0.0443 | -0.0127 | 2/5 |

Paired conditional bootstrap for mixture improvement: +0.0095, nominal95% interval [+0.0007, +0.0177]. This omits model retraining, epoch/architecture exploration and multiple-policy uncertainty.

Interpretation: late Attention reweighting contributes to deterioration in this frozen-model intervention, while completely uniform pooling loses useful information. Earlier weights alone do not consistently help. The50/50 mixture partly recovers last-model performance but remains below epoch5/10 (~0.7325) and selected best checkpoints (~0.7458). It is a mechanism diagnostic, not a demonstrated better training method.

Weight intervention creates inputs the fixed late prediction head was not trained on. It cannot establish that temporal Attention regularization will improve training, nor rule out projection/head drift, finite supervision or missing prognostic factors. No evidence here proves intrinsic UNI information exhaustion, label noise, or a specific histological cause.
