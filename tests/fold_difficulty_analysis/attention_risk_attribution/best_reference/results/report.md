# Best-attention reference: frozen Train/Val diagnostic

No retraining. Same original721 patients, saved five folds, Raw UNI complete patches and frozen last ABMIL projection/head. All Train/Val metrics use eval mode. Best epoch was selected on Outer Val: follow-up cannot remove this selection bias and conditional bootstrap does not cover it.

| Fold | Role | Actual best C | Actual late C | Best attention + late H/head C | Late attention + late H/head C | Attention delta C |
|---|---|---:|---:|---:|---:|---:|
| 0 | train | 0.8937 | 0.9540 | 0.9143 | 0.9540 | +0.0396 |
| 0 | val | 0.6902 | 0.6757 | 0.6844 | 0.6757 | -0.0087 |
| 1 | train | 0.8775 | 0.9526 | 0.9071 | 0.9526 | +0.0455 |
| 1 | val | 0.7559 | 0.7339 | 0.7371 | 0.7339 | -0.0032 |
| 2 | train | 0.7804 | 0.9100 | 0.8430 | 0.9100 | +0.0670 |
| 2 | val | 0.7908 | 0.7288 | 0.7602 | 0.7288 | -0.0314 |
| 3 | train | 0.7763 | 0.8944 | 0.8359 | 0.8944 | +0.0584 |
| 3 | val | 0.7719 | 0.7310 | 0.7488 | 0.7310 | -0.0178 |
| 4 | train | 0.8562 | 0.9383 | 0.8815 | 0.9383 | +0.0568 |
| 4 | val | 0.7202 | 0.6931 | 0.6987 | 0.6931 | -0.0056 |

## Contextual patch interpretation

Exact best-reference straight-line attention IG sums to frozen risk difference. For an event-patient/comparator pair, margin change=sum(event IG)-sum(comparator IG). The contextual scores measure negative margin support on credit-losing pairs, its opposing mass, positive support on credit-gaining pairs, and its opposing mass. Half-credit ties weight the corresponding pair context by0.5. One patch may support both correction and damage across different pairs. These scores are not unique allocations of C-index loss, tissue importance or causal effects. Raw mass includes pair exposure counts and risk units, so it must not be compared directly across Train/Val or models; it is used only for within-WSI selection here.

## Matching and native quality

Primary matching: up to8 positive-context-harm patches per WSI, disjoint bottom-half low-score controls, thumbnail white-fraction caliper0.1, then nearest normalized UNI cosine. UNI normalization is used only for matching, not for model input. No histology labels exist: this is not same-tissue matching. Actual native white-fraction differences are audited; primary QC contrasts use native difference<=0.1. Both matching failure rates and all screened patches are retained. White fraction is a balance variable here: small post-match white differences do not rule out background confounding.

Val QC covers eligible patients; Train QC covers only15 selected review examples, not the full Train distribution. Mean Train/Val C-indices are fold summaries; training patients recur across four folds and are not independent observations. No independent-patient bootstrap or hypothesis test is applied to the repeated Train sample.

Quality proxies are white/black fractions, tissue-masked Laplacian variance and OD/saturation summaries. They do not label blur, stain artifacts or biological tissue types. Native sharpness/OD can depend on local tissue structure; matching is incomplete control, not randomized evidence.

## Blinded review

Planned30 review cases; rendered30. Per fold: two most actual-best-to-late deteriorated Val patients plus a stable control; two most improved Train patients plus a stable control. Case IDs/group labels/tile order are shuffled. No-harm review cases may use abs IG fallback; these are marked and excluded from primary harm-QC summaries. Native tiles, WSI locations, matching keys and empty forms stay private; tissue labels and actual doctor review remain pending.

## Limits

The existing fixed-Epoch5 and best references are both retained in reference_sensitivity.csv; a best-conditioned sign pattern is not independent of how the peak was selected. Best-conditioned 5/5 negative Val changes are descriptive, not independent validation or proof attention is the sole cause. A Train improvement/Val decline contrast supports fitted-sample specificity under this intervention, not a proven disease mechanism. Counterfactual path components depend on the chosen endpoint network and representation. Technical QC associations cannot identify stable histological DFS relations; actual pathology review and an independent evaluation would still be needed. No new loss, encoder adaptation or formal method change follows automatically.

Verification: all3605 saved IG decompositions, separate Train/Val comparable pairs, patient roles, context counts/masses and matching endpoints are independently recomputed. Prior validation IG is checked against the original best_contributions before reuse. All checkpoint/source files are read-only.
