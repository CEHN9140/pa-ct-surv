# Completed nested scalar-risk increment diagnostic

All five outer folds and all 40 independent unimodal fits completed normally.
The background process exited, the completion marker is present, and all
three prediction variants have five-fold metrics, train/validation prediction
tables and KM plots. This is a diagnostic comparison, not a formal teacher or
student, and loads no project-trained unimodal checkpoint.

## Outer-validation performance

Mean ± sample SD across five folds. Train is the final outer-refit model in
eval mode and is in-sample. Outer-validation outcomes do not select training
epochs, checkpoints, scalers or alpha.

| Model | Train C-index | Val C-index | AUC36 | AUC60 | Mean gap |
|---|---:|---:|---:|---:|---:|
| Newly refitted pathology | 0.7637 ± 0.0441 | 0.7390 ± 0.0350 | 0.7744 ± 0.0279 | 0.7571 ± 0.0525 | 0.0247 |
| Newly refitted CT | 0.7436 ± 0.0135 | 0.7041 ± 0.0570 | 0.7250 ± 0.0522 | 0.7153 ± 0.0895 | 0.0395 |
| Scalar-risk fusion | 0.7647 ± 0.0453 | 0.7399 ± 0.0340 | 0.7757 ± 0.0290 | 0.7566 ± 0.0534 | 0.0248 |

Primary matched contrast: fusion−pathology Val C-index +0.00094655,
conditional paired-patient bootstrap 95% interval [−0.00180335,+0.00396234]
(2,000 replicates). AUC36 increases approximately0.00129; AUC60 decreases
approximately0.00050. There is no clear consistent increment over pathology.
Fusion−CT C-index +0.03585413, interval [+0.00575644,+0.06511928], conditional
on these trained models. All intervals exclude retraining/configuration-selection
uncertainty and remain exploratory internal CV.

## Calibration and per-fold effects

The prediction is `z_pathology + alpha*z_CT`, with the pathology coefficient
fixed at1. Alpha is unrestricted and learned only from standardized inner OOF
predictions. The scalers are fitted only on corresponding fitting patients.

| Fold | Alpha | Pathology Val | Fused Val | Difference | PA/CT fixed refit epochs |
|---|---:|---:|---:|---:|---|
| 0 | +0.058186 | 0.709458 | 0.715090 | +0.005632 | 1 / 25 |
| 1 | −0.027209 | 0.720461 | 0.718762 | −0.001699 | 14 / 25 |
| 2 | −0.021751 | 0.782194 | 0.781609 | −0.000584 | 1 / 23 |
| 3 | +0.006359 | 0.771374 | 0.772133 | +0.000758 | 14 / 27 |
| 4 | +0.122293 | 0.711426 | 0.712052 | +0.000626 | 23 / 27 |

The largest benefit is fold0; the other four effects are near zero or negative.
Alpha is small relative to the fixed unit pathology coefficient and its sign
varies. These coefficients are not probabilities or percentages of modality
contribution. Negative coefficients are conditional risk corrections, not a
change in event encoding. OOF Cox loss decreases in all folds, by
0.001607/0.000364/0.000249/0.000016/0.005740; all scalar optimizations converge
with absolute gradients below1e-7. Weak effects are not explained by a failed
alpha optimizer.

## Ranking changes and residual overfitting

Across25,651 within-fold comparable validation pairs, fusion corrects144
pathology errors and breaks120 formerly correct orderings (net+24). These
pairs are dependent and are not additional patient samples. Summed pair
counts weight folds differently from the reported mean fold C-index.
Mean within-fold Spearman correlation between fused and pathology risks is
0.998177, versus pathology/CT correlation0.639980. Prediction differences
remain small; correlation alone is not a biological information-overlap test.

Mean gap is unchanged (pathology0.024735, fusion0.024814). Fold4 still has
pathology Train0.819979/Val0.711426 and fusion0.823941/0.712052. Scalar
addition does not resolve that fold's overfitting. KM separation is present
for all three models but does not establish incremental fusion performance.

## Why these values differ from the historical baseline

The historical pathology Val0.7528 used each outer validation fold to select
its checkpoint. This diagnostic instead selects the training duration from
separate inner stopping sets, then refits fresh models for fixed epoch counts.
Pathology durations are1/14/1/14/23. Therefore this is a changed training and
selection procedure; the decrement cannot be attributed solely to selection
bias, feature quality or fusion. The primary incremental comparison is the
new pathology0.738983 versus new fusion0.739929 under the same protocol.
The small inner stopping sets and variable selected durations also limit the
power/stability of this particular diagnostic.

## Audit and conclusion

All721 paired patients (251 recorded events) appear in exactly one outer
validation fold. Within each inner fold, fit/stop/OOF IDs are disjoint and
exclude outer-validation patients. All40 checkpoint metadata, five completion
markers, OOF membership and epoch rules were verified. Alpha can be reproduced
from saved OOF predictions; outer histories contain no validation metrics.
Outer risk scalers reproduce from training predictions; raw, standardized and
fused risks reconstruct. All Train/Val C-index, AUC36/60 and training-median
KM group cutoffs reproduce the exports. See `review_audit.json`.

This adds evidence that the existing one-dimensional CT risk score does not
provide a stable increment over pathology in this nested linear-additive
calibration protocol. Complex attention or joint-encoder competition is not
required for a near-zero gain to occur. It does not prove absent biological
complementarity, optimal CT representation, all possible fusion ineffectiveness,
or that the exact performance cause is uniquely redundancy rather than
representation/calibration/generalization limitations. Higher-dimensional CT
features can contain conditional information absent from its single risk score.
