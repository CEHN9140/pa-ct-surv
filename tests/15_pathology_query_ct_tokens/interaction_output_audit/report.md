# Original standard cross-attention: query/output diversity audit

Scope: the original Top25 pathology-L2 standard-attention teacher, no QKNorm,
alignment, CT Z-score or BN adaptation. Five selected best checkpoints, all
721 validation patients. Read-only eval/inference; no optimizer or parameter
updates. Match the training seed and cuDNN deterministic settings. Predictions
reproduce saved risks (largest absolute error below 1.1e-7), manual internal
Q/K/V and output-projection reconstruction matches MHA, all BN buffers remain
unchanged, and checkpoint hashes are unchanged.

Statistics are computed per patient, averaged within fold, then averaged over
five folds. Pairwise cosine excludes diagonal/self-pairs and uses float64.
Attention similarity compares the 25 distributions over the same 512 CT
positions; retrieval similarity compares the 25 output-projected CT vectors
before query pooling. Similarity is within each patient, not between patients.

| Statistic | Five-fold mean |
|---|---:|
| Scaled QK logit SD along CT positions | 0.0023516908 |
| CT attention entropy / log(512) | 0.9999993776 |
| Different-query attention mean pairwise cosine | 0.9999996749 |
| Internal CT Value mean pairwise cosine | 0.5984065327 |
| Different-query retrieved output mean pairwise cosine | 0.9999999673 |
| Internal Query mean pairwise cosine | 0.9801801258 |
| Internal Key mean pairwise cosine | 0.6251541715 |

Cosine can be inflated by shared means/biases, so also examine relative
variation: within-query centered retrieval RMS / uncentered RMS is 0.00012663
(0.01266%). The RMS difference from uniform mean-Value retrieval, divided by
retrieval RMS, is 0.00080321 (0.08032%). Attention's within-query centered
relative RMS is 0.00037715 (0.03772%). These support functional near-uniform
retrieval; they are representation differences, not percentages of causal or
predictive modality contribution.

## Exact fusion path

```text
Pathology patch route:
25 x 1280 -> per-patch L2 -> Linear1280,512/ReLU/Dropout
                                      |
             +------------------------+---------------------------+
             |                                                    |
Linear512,256/ReLU/Dropout                              ABMIL attention pool
             |                                         512 -> 256/ReLU/Dropout
       internal Q Linear                                          |
             |                                               h_path [256]
       cross-attention                                            |
             ^                                                    |
             |                                                    |
CT layer4 512 positions x 512 channels                             |
 -> Linear512,256/ReLU/Dropout -> internal K/V                      |
 -> attention-weighted V -> output Linear                          |
 -> 25 x 256 retrieved CT outputs                                  |
 -> learned query attention pool -> h_guided_ct [256]               |
             |                                                    |
             +---------------- concat ----------------------------+
                                  |
                  512 -> 256 -> 128 -> 1 raw risk
```

The query-pooling scorer is Linear256,128/Tanh/Dropout/Linear128,1, with softmax
over 25 query outputs. Each hidden layer of the concat head has LayerNorm,
ReLU and dropout. There is no LayerNorm in the Query projection route and no
query residual in this cross-attention block. Original pathology information
is retained through its independent ABMIL patient route. The original global
CT risk head is diagnostic only (lambda_ct=0); its global feature is not an
additional bypass input to the fused prediction.

## Interpretation alongside existing experiments

The original norm argument needs a narrower conclusion: unit-norm raw input
alone cannot establish small QK logits, because several learned projections
follow. However, the actual post-projection internal Q/K and logits were
measured; the small scores and near-uniform retrieval are observed behaviors,
not deductions from input L2. Existing `score_diagnosis` shows raw patch cosine
0.6539, compressed patch cosine0.9122 and MHA-input Query cosine0.9816;
normalizing raw patches leaves their pairwise directions unchanged. This
does not rule out indirect effects of input scale on subsequent learning.

The suggested matched uniform-CT versus cross-attention training control is
already complete: Val C-index0.754291 versus0.754444. Pathology with the same
head gives0.752090. Uniform CT fusion still permits nonlinear feature
interactions in the final MLP; it removes query-dependent spatial selection,
not every kind of cross-modal interaction.

The full-query experiment already expanded queries to all available patches
while keeping the direct pathology Top25 route. It did not restore selective
attention or improve validation performance. Thus Top25 coverage alone is
not supported as the primary cause; this does not settle every patch-selection
or representation question.

CT Values are not all identical in this audit. Deep-layer receptive fields
and overlapping contexts remain representation-suitability hypotheses, not
evidence that layer4 is invalid. The observations support weak query-specific
retrieval, but do not establish that this is the sole reason for the absent
fusion gain, identify L2 as its cause, show that raw modalities lack biological
complementarity, or prove the pooling/head discards useful predictive signal.

Detailed values, per-fold checkpoint hashes and audit methods: `summary.json`.
