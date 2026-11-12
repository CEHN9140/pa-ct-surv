# Standard cross-attention + CT Z-score: completed review

All five folds completed with best checkpoints, predictions, metrics and KM plots present. Selected-checkpoint Train/Val metrics use eval mode. Mean ± sample standard deviation across five folds.

| Model | Train C-index | Val C-index | AUC36 | AUC60 | Mean gap |
|---|---:|---:|---:|---:|---:|
| Standard attention | 0.7897 ± 0.0339 | 0.7544 ± 0.0400 | 0.7935 ± 0.0326 | 0.7728 ± 0.0518 | 0.0353 |
| Standard attention + CT Z-score | 0.7762 ± 0.0278 | 0.7555 ± 0.0370 | 0.7894 ± 0.0247 | 0.7712 ± 0.0467 | 0.0207 |
| QKNorm | 0.8109 ± 0.0379 | 0.7524 ± 0.0442 | 0.7896 ± 0.0299 | 0.7726 ± 0.0549 | 0.0585 |
| QKNorm + CT Z-score | 0.7934 ± 0.0398 | 0.7594 ± 0.0400 | 0.7969 ± 0.0337 | 0.7825 ± 0.0640 | 0.0339 |

| Fold | Standard | Standard + Z-score | Difference |
|---|---:|---:|---:|
| 0 | 0.7275 | 0.7399 | +0.0124 |
| 1 | 0.7331 | 0.7422 | +0.0091 |
| 2 | 0.8139 | 0.8034 | -0.0105 |
| 3 | 0.7773 | 0.7818 | +0.0045 |
| 4 | 0.7204 | 0.7102 | -0.0102 |

Z-score changes mean Val C-index by only +0.00106; AUC36 falls by 0.00416 and AUC60 by 0.00158. The overall pattern is essentially comparable, without a clear discrimination benefit. The QKNorm-condition Z-score gain cannot be generalized to standard attention.

Mean gap decreases from 0.03530 to 0.02066, partly alongside earlier selected checkpoints. Fold4 still has Train 0.8253 versus Val 0.7102 (gap 0.1152), so a small average gap does not imply all folds avoid overfitting.

Best epochs are 6/7/11/9/17, stop epochs 16/17/21/19/27. Last-ten-epoch Val ranges are 0.0275/0.0127/0.0362/0.0220/0.0127, lower than original in 4/5 folds; the trajectories still peak and fluctuate/decline. This is a modest observed stability change, not a demonstrated generalization improvement.

## Read-only fixed-checkpoint validation checks

| Condition | Five-fold mean C-index |
|---|---:|
| Normal paired | 0.755504 |
| Uniform CT attention | 0.755542 |
| Shuffle query, 100 derangements/fold | 0.755392 |
| Shuffle CT, 100 derangements/fold | 0.690262 |
| Shared fixed training CT, 3 anchors/fold | 0.741532 |

Normalized CT-attention entropy is 0.9999943054 (uniform = 1). All 721 full-model and cached predictions reproduce saved risks, and all BN buffers stay unchanged during inference. Five checkpoint states contain no QKNorm scale and all 34 running-mean/variance buffers match public pretraining.

Despite CT identity affecting predictions, replacing pathology queries or forcing uniform attention barely changes discrimination. The current cross-attention retrieval is functionally close to averaging projected CT Values and concatenating the resulting CT summary with direct pathology. This supports CT use in the final predictor, not effective patient-specific pathology-guided spatial selection. CT-shuffle losses are interventions on mismatched inputs, not percentages of causal/clinical modality contribution.

The comparison does not establish an input-processing bug, intrinsic inferiority of standard attention, or a necessity to use QKNorm. It shows that CT Z-score alone has not resolved the weak query-conditioned behavior or produced a meaningful performance gain in this single-seed internal CV. No new training or architecture changes were made during the review.

Detailed aggregate inference methods and checkpoint hashes: `inference_summary.json`.
