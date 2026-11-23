# Frozen80% subbag sampling stability

No neural training. Original Raw UNI full WSI, five saved bs32 best/last models, eval mode. Each patient uses30deterministic80%-without-replacement subsets; identical subsets across checkpoints/folds, original patch order retained. Ceil(0.8N), singleton retained.

In eval ABMIL, projection/attention logits are patch independent; cached subset-softmax/pooling is equivalent to direct subset forwards. Unit tests and direct real-GPU checks verify this. Full risks must reproduce original snapshot predictions. Risk SD uses30subbag predictions; normalized SD/RMSE divides by that checkpoint full-Train risk SD. Spearman, C-index, all-pair strict flips and comparable-pair credit/strict changes are distinct metrics.

| Phase | Role | Full C | Subbag mean C | Mean Spearman | Comparable flip fraction | Normalized subbag SD |
|---|---|---:|---:|---:|---:|---:|
| best | train | 0.8368 | 0.8366 | 0.9996 | 0.0050 | 0.0191 |
| best | val | 0.7458 | 0.7451 | 0.9993 | 0.0066 | 0.0204 |
| last | train | 0.9298 | 0.9279 | 0.9977 | 0.0102 | 0.0482 |
| last | val | 0.7125 | 0.7119 | 0.9970 | 0.0163 | 0.0491 |

Repeated subbags are correlated technical replicates, not30independent clinical validations. Train patients recur across folds. Best checkpoints used Outer Val selection. This diagnoses sensitivity, not its causal role in overfitting. No sampling proportion/lambda is chosen from Outer Val and no consistency training is automatically started. Historical strict baseline0.7248 applies only to its exact inner-holdout protocol; no new training comparison is made here.
