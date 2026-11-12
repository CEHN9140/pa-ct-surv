# CT normalization / BN adaptation controls: completed review

All three runs use Top25 QKNorm, identical saved CV assignments and optimization settings. B changes CT input normalization only; C changes training BN-stat updates only. All five folds completed. Train metrics below use the selected best checkpoint in eval mode; mean ± sample standard deviation.

| Control | Train C-index | Val C-index | AUC36 | AUC60 | Mean gap |
|---|---:|---:|---:|---:|---:|
| A QKNorm baseline | 0.8109 ± 0.0379 | 0.7524 ± 0.0442 | 0.7896 ± 0.0299 | 0.7726 ± 0.0549 | 0.0585 |
| B CT Z-score + fixed BN | 0.7934 ± 0.0398 | 0.7594 ± 0.0400 | 0.7969 ± 0.0337 | 0.7825 ± 0.0640 | 0.0339 |
| C original CT + BN updates | 0.7594 ± 0.0222 | 0.7335 ± 0.0430 | 0.7679 ± 0.0345 | 0.7519 ± 0.0626 | 0.0259 |

| Fold | A baseline | B Z-score | C BN updates |
|---|---:|---:|---:|
| 0 | 0.7192 | 0.7514 | 0.7211 |
| 1 | 0.7393 | 0.7403 | 0.7167 |
| 2 | 0.8085 | 0.8019 | 0.7945 |
| 3 | 0.7882 | 0.7968 | 0.7545 |
| 4 | 0.7066 | 0.7068 | 0.6806 |

B improves mean C-index by 0.00706, AUC36 by 0.00734 and AUC60 by 0.00996 versus A. Most C-index improvement is fold0 (+0.03224), followed by fold3 (+0.00853); fold1/fold4 are nearly unchanged and fold2 falls by 0.00662. B mean gap decreases from 0.05849 to 0.03394, but checkpoints are selected at different epochs. C worsens mean C-index by 0.01891; its smaller gap does not establish a useful regularization benefit.

B is not consistently smoother within fold: last-ten-epoch Val ranges exceed A in 4/5 folds. Fold0 peaks at epoch2, and all folds decline after their selected peak. C has larger fluctuations; fold2 last-ten range is 0.1411 versus 0.0291 for A.

## Fixed-model inference checks

| Model | Paired C-index | Shuffle query | Shuffle CT | Shared training CT | Normalized attention entropy |
|---|---:|---:|---:|---:|---:|
| A baseline | 0.7524 | 0.5508 | 0.7375 | 0.7444 | 0.7563 |
| ctzscore | 0.7594 | 0.5927 | 0.7100 | 0.7439 | 0.6716 |
| ctbnadapt | 0.7335 | 0.5304 | 0.7207 | 0.7461 | 0.7686 |

Query/CT swaps use 100 derangements per fold. Shared-CT values average three predetermined training anchors, with exactly one CT shared by all validation recipients in each pass. All normal risks reproduce saved predictions, and inference leaves every BN buffer unchanged.

For B, the mean CT-shuffle penalty grows from 0.01486 to 0.04943; the shared-CT penalty grows from 0.00798 to 0.01553. Fold0 especially depends more on patient-specific CT (B paired 0.7514, shared 0.7104). This is consistent with altered CT utilization, but perturbation sensitivity does not establish anatomical alignment or independent clinical utility. Fold3 shared CT slightly exceeds paired CT, and fold4 remains weak. C shared CT (0.7461) exceeds its paired score (0.7335), providing no support for this particular BN-update scheme.

## Conclusions and limits

- Current results support CT input normalization as a useful candidate treatment in this configuration, not a proven preprocessing bug or a unique cause. Per-ROI Z-score also changes intensity invariances and optimization, so the gain cannot be attributed solely to matching pretraining BN statistics.
- The results do not support the claim that freezing BN was inherently wrong. Directly updating BN with patientwise forwards worsens validation performance and stability here. This does not evaluate other BN calibration methods or a combined Z-score/adaptive-BN condition.
- B is a more promising candidate than C: metrics improve modestly and mean gap decreases, but no repeat-seed superiority is established and within-fold oscillation is unresolved. Compared with pathology-only Val 0.7528, B at 0.7594 is a modest observed gain, not a definitive fusion advantage.
- Do not select C because of its smaller gap, or claim B has solved overfitting/generalization. No new training was launched for this review.

Files: `inference_summary.json` contains aggregate inference methods and checkpoint hashes; `training_and_bn_audit.json` contains per-fold trajectory and BN-update counts.
