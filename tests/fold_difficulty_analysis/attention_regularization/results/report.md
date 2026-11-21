# Attention regularization: matched inner-selection survival comparison

Completed35 inner fits and15 freshly initialized outer refits. Raw UNI/full patches, original ABMIL/Cox32/Adam1e-4/WD5e-4/dropout.25/seed123. Each saved outer fold is unchanged. Fixed event-stratified80/20 FIT/STOP inside Outer Train selects candidate and Epoch; Outer Val is evaluated only at the final fixed-budget checkpoint. Single inner holdout, not threefold inner CV.

This implements the user-specified normalized entropy loss inspired by [AEM MICCAI2025](https://papers.miccai.org/miccai-2025/0056-Paper5183.html), not an exact classification/Cosine Weight Annealing reproduction. No EMA, architecture changes, input L2 or patch sampling.

| Method | Train C-index | Val C-index | AUC36m | AUC60m | Regularized folds |
|---|---:|---:|---:|---:|---:|
| baseline | 0.8576 ± 0.0563 | 0.7248 ± 0.0404 | 0.7636 ± 0.0150 | 0.7397 ± 0.0397 | 0/5 |
| entropy | 0.8562 ± 0.0545 | 0.7225 ± 0.0446 | 0.7605 ± 0.0202 | 0.7379 ± 0.0443 | 4/5 |
| neff | 0.8568 ± 0.0546 | 0.7229 ± 0.0432 | 0.7603 ± 0.0209 | 0.7373 ± 0.0442 | 1/5 |

## Outer-fold incremental performance

| Method vs matched baseline | Mean delta C | Positive folds | Conditional95% interval |
|---|---:|---:|---:|
| entropy | -0.0023 | 1/5 | [-0.0060, +0.0011] |
| neff | -0.0019 | 0/5 | [-0.0047, +0.0007] |

Intervals use2000 paired patient resamples within each validation fold, conditional on fitted models. They omit refitting/selection and previous repeated exploration and are unadjusted for two regularized-family comparisons. This remains exploratory internal validation, not an independent external test.

## Actual control strength

| Family | Mean Outer Val Neff/N | Mean Outer Val normalized entropy | Mean training active-penalty fraction |
|---|---:|---:|---:|
| baseline | 0.2504 | 0.8574 | 0.0000 |
| entropy | 0.2688 | 0.8712 | 0.8000 |
| neff | 0.2553 | 0.8740 | 0.0650 |

A family can select zero regularization. A nonzero coefficient with little active exposure is not proof the intended constraint materially changed optimization. Larger q/reduced Train-Val gap alone is not success; DFS ranking/AUC must improve relative to the freshly fitted matched baseline.

Selected inner curves are selected on STOP and therefore descriptive, not unbiased evidence of an effect. The independent outer final comparison evaluates the full selection procedure. Historical best scores selected on Outer Val are a different protocol and must not be used as the matched comparator.

All50 audits share the same initial parameter hash. Inner/outer patient roles, fixed outer Epoch, exposure counts and30 final Train/Val C-indices were independently checked from saved artifacts. No project-trained unimodal model initializes any fit. Private patient mappings/predictions and all checkpoints stay local. No guarantee of benefit to a future multimodal teacher or CT student is established here.
