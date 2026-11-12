# Experiment 15: full-query result review

Five-fold internal CV; mean ± sample standard deviation. Train metrics use the selected best checkpoint in eval mode.

| Model | Train C-index | Val C-index | AUC36 | AUC60 | Mean Train−Val gap |
|---|---:|---:|---:|---:|---:|
| Single pathology Top25 L2 | 0.7905 ± 0.0317 | 0.7528 ± 0.0456 | 0.7934 ± 0.0385 | 0.7726 ± 0.0571 | 0.0377 |
| Teacher Top25 queries | 0.7897 ± 0.0339 | 0.7544 ± 0.0400 | 0.7935 ± 0.0326 | 0.7728 ± 0.0518 | 0.0353 |
| Teacher all queries | 0.7854 ± 0.0344 | 0.7524 ± 0.0420 | 0.7905 ± 0.0308 | 0.7694 ± 0.0578 | 0.0330 |

All five folds completed normally, with best checkpoints, predictions, metrics and KM plots present. All-query training stopped at epochs 18/17/19/26/29; best epochs 8/7/9/16/19. Original Top25 best epochs: 9/16/10/9/19.

Only query coverage changes: all available same-grid Virchow2 patches (67–755, median 375), while direct ABMIL retains the original Top25. Cohort, saved splits, Cox32, patch L2, fusion-only loss, public initialization, dropout and optimization settings match. Changing bag size also changes dropout RNG consumption; this is one seed, not a paired identical-gradient experiment.

| Fold | Top25 Val C-index | All-query Val C-index | Difference |
|---|---:|---:|---:|
| 0 | 0.7275 | 0.7262 | -0.0014 |
| 1 | 0.7331 | 0.7305 | -0.0026 |
| 2 | 0.8139 | 0.8159 | +0.0019 |
| 3 | 0.7773 | 0.7740 | -0.0032 |
| 4 | 0.7204 | 0.7154 | -0.0050 |

## Fixed-checkpoint validation inference

All 721 validation predictions reproduce saved risks. Normalized CT attention entropy: 0.9999996038 (uniform = 1).

Normal five-fold mean C-index: 0.75238763; uniform-attention mean: 0.75242768; query-shuffle mean across 20 derangements/fold: 0.75239173. Query swaps keep the recipient direct pathology patient feature and CT unchanged.

Uniform attention retains learned V/output projections and replaces retrieval by the mean projected CT Value at each query. The near-identical metrics indicate that the present predictor is largely insensitive to patient-specific query-conditioned retrieval. This does not show that the direct pathology branch is unused or that CT contains no complementary information.

## Interpretation

More queries did not improve validation discrimination (4/5 folds slightly lower); the mean gap changes only from 0.0353 to 0.0330. All-query validation still improves from epoch 1 to its best epoch, then plateaus/declines; best-to-last drops are 0.0226/0.0110/0.0491/0.0282/0.0392. There is no consistent stability gain.

This run does not support insufficient Top25 query coverage as the primary explanation. It also does not prove a unique cause for the nearly uniform attention: projection scale, optimization, competing direct-branch signal and weak identifiable cross-modal relationships remain possible. Neither a significant performance difference nor the absence of complementary biological information can be inferred from one seed. No training or model weights were changed for this review.

Detailed aggregate inference values and checkpoint hashes: `inference_summary.json`.
