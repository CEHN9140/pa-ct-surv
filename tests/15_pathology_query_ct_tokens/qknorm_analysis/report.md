# QKNorm experiment 15 result review

All five folds completed with best checkpoints, prediction tables, KM plots and metrics present. All 721 saved validation predictions were reproduced during fixed-checkpoint inference; no training or checkpoint changes were performed.

## Selected-checkpoint results

Train C-index is re-evaluated in eval mode using the best validation checkpoint, not the final epoch or online train-mode score. Values are five-fold mean ± sample standard deviation.

| Model | Train C-index | Val C-index | AUC36 | AUC60 | Mean gap |
|---|---:|---:|---:|---:|---:|
| Single pathology Top25 L2 | 0.7905 ± 0.0317 | 0.7528 ± 0.0456 | 0.7934 ± 0.0385 | 0.7726 ± 0.0571 | 0.0377 |
| Teacher Top25 | 0.7897 ± 0.0339 | 0.7544 ± 0.0400 | 0.7935 ± 0.0326 | 0.7728 ± 0.0518 | 0.0353 |
| Teacher all queries | 0.7854 ± 0.0344 | 0.7524 ± 0.0420 | 0.7905 ± 0.0308 | 0.7694 ± 0.0578 | 0.0330 |
| Teacher Top25 QKNorm | 0.8109 ± 0.0379 | 0.7524 ± 0.0442 | 0.7896 ± 0.0299 | 0.7726 ± 0.0549 | 0.0585 |

## Per-fold validation comparison

| Fold | Top25 | QKNorm | Difference |
|---|---:|---:|---:|
| 0 | 0.7275 | 0.7192 | -0.0084 |
| 1 | 0.7331 | 0.7393 | +0.0062 |
| 2 | 0.8139 | 0.8085 | -0.0055 |
| 3 | 0.7773 | 0.7882 | +0.0110 |
| 4 | 0.7204 | 0.7066 | -0.0138 |

QKNorm best epochs: 10/4/8/17/12; stopping epochs: 20/14/18/27/22. Final-epoch Val C-indices: 0.7003/0.7290/0.7660/0.7545/0.6781. The last-ten-epoch Val range is 0.0473 (fold 3) and 0.0655 (fold 4), versus 0.0251/0.0263 for original Top25. Optimization and stability have not consistently improved.

## Fixed-checkpoint functional checks

| Condition | Five-fold mean C-index | Drop from normal |
|---|---:|---:|
| Normal paired | 0.752375 | +0.000000 |
| Uniform CT attention | 0.647834 | +0.104541 |
| Shuffle pathology query (100/fold) | 0.550765 | +0.201610 |
| Shuffle CT (100/fold) | 0.737516 | +0.014859 |

Normalized CT-attention entropy is 0.756266, compared with approximately 0.999999 for original Top25 (uniform = 1). Learned scale is 17.979456 on average, from initialization 17.997179.

## Interpretation

QKNorm strongly changes functional query dependence: uniform attention and query swaps are no longer neutral. This supports activation of the query-conditioned retrieval pathway. The much smaller CT-shuffle penalty, including an almost zero penalty in fold 0, indicates uneven reliance on patient-specific CT content. It does not by itself prove correct pathology-to-CT anatomical correspondence or complementary DFS information.

Predictive performance does not improve over original Top25 or the pathology baseline at this seed, while the mean train-validation gap increases from 0.0353 to 0.0585. Thus near-uniform attention was a real limitation of the old mechanism, but resolving that behavior is insufficient to improve validation generalization. More expressive query-driven retrieval can still mainly re-express pathology information, learn unstable relationships, or overfit; these remain hypotheses, not established unique causes.

Do not describe this as pathology being unable to guide CT: this model is now clearly sensitive to the pathology query. The remaining issue is whether that conditioning adds reproducible predictive value beyond pathology alone. These are internal CV, single-seed, post-selection comparisons; no superiority or inferiority significance is established.

Detailed fold diagnostics, methods and checkpoint hashes: `inference_summary.json`.
