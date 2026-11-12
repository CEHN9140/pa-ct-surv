# Completed matched fusion controls

Both new runs completed all five folds without a logged traceback. Each has
five best checkpoints, train/validation predictions, survival metrics and KM
plots. All analyses below read existing outputs; no model training, checkpoint
updates or new architecture changes were performed during this review.

## Comparison

Mean ± sample standard deviation across five folds. Train metrics are the
selected best checkpoint re-evaluated in eval mode, not online training metrics
or the final epoch. P/CT are the original matched unimodal comparisons; A/B/C
are the matched-head pathology ablation, uniform-CT fusion and original
standard cross-attention respectively.

| Model | Train C-index | Val C-index | AUC36 | AUC60 | Mean Train−Val gap |
|---|---:|---:|---:|---:|---:|
| P: original pathology baseline | 0.7905 ± 0.0317 | 0.7528 ± 0.0456 | 0.7934 ± 0.0385 | 0.7726 ± 0.0571 | 0.0377 |
| Original CT baseline | 0.7340 ± 0.0141 | 0.7157 ± 0.0508 | 0.7445 ± 0.0444 | 0.7294 ± 0.0815 | 0.0183 |
| A: pathology only, same fusion head | 0.7993 ± 0.0642 | 0.7521 ± 0.0416 | 0.7935 ± 0.0310 | 0.7707 ± 0.0537 | 0.0472 |
| B: paired CT, uniform spatial attention | 0.7962 ± 0.0336 | 0.7543 ± 0.0407 | 0.7916 ± 0.0325 | 0.7743 ± 0.0549 | 0.0419 |
| C: original learned standard cross-attention | 0.7897 ± 0.0339 | 0.7544 ± 0.0400 | 0.7935 ± 0.0326 | 0.7728 ± 0.0518 | 0.0353 |

| Fold | A Val | B Val | C Val | B−A | C−B |
|---|---:|---:|---:|---:|---:|
| 0 | 0.7100 | 0.7250 | 0.7275 | +0.0150 | +0.0025 |
| 1 | 0.7516 | 0.7342 | 0.7331 | −0.0174 | −0.0011 |
| 2 | 0.8087 | 0.8104 | 0.8139 | +0.0018 | +0.0035 |
| 3 | 0.7754 | 0.7839 | 0.7773 | +0.0085 | −0.0066 |
| 4 | 0.7148 | 0.7179 | 0.7204 | +0.0031 | +0.0025 |

B improves four folds versus A, but the mean improvement is small and fold1
declines. AUC36 is slightly lower for B than A, while AUC60 is slightly higher.
Learned attention does not consistently improve any of these metrics over B.

## Paired uncertainty, conditional on selected predictions

Five thousand paired patient bootstrap samples, independently resampled within
each validation fold. Each draw averages the five fold C-index differences;
models always share sampled patients. Risk scores from different folds are
never compared. Weighted pair calculations were checked against scikit-survival
on explicit bootstrap patient samples.

| Comparison | Mean Δ Val C-index | Exploratory 95% percentile interval |
|---|---:|---:|
| B−A | +0.00220 | [−0.01089, +0.01460] |
| C−B | +0.00015 | [−0.00693, +0.00707] |
| C−A | +0.00235 | [−0.01177, +0.01544] |
| B−original pathology | +0.00153 | [−0.01059, +0.01349] |
| C−original pathology | +0.00169 | [−0.01185, +0.01531] |

All intervals cross zero. These are conditional descriptive intervals, not
independent confirmation or a proof of equivalence. They do not include seed,
retraining or validation-based checkpoint/model-selection uncertainty.

## Which rankings changed?

Among 25,651 within-fold comparable validation pairs:

| Comparison | Corrected errors of reference | Broke correct reference orderings | Net |
|---|---:|---:|---:|
| B versus A | 1,189 | 1,135 | +54 |
| C versus A | 1,293 | 1,237 | +56 |
| C versus B | 492 | 490 | +2 |

Pair counts are descriptive, not independent samples: each patient participates
in multiple pairs. Summed net counts also weight folds by their comparable-pair
counts and are not a replacement for the five-fold mean C-index difference.
Mean within-fold Spearman correlation of B/C validation predictions is 0.98670.
Thus both trained fusion paths produce closely related rankings, and their
corrections are largely offset by newly introduced errors.

## Stability and overfitting

| Model | Best epochs, folds 0–4 | Stopping epochs | Mean last-ten-epoch Val range | Mean absolute adjacent Val step |
|---|---|---|---:|---:|
| A | 3, 30, 7, 11, 11 | 13, 30, 17, 21, 21 | 0.03234 | 0.00538 |
| B | 7, 16, 10, 16, 20 | 17, 26, 20, 26, 30 | 0.02854 | 0.00656 |
| C | 9, 16, 10, 9, 19 | 19, 26, 20, 19, 29 | 0.02594 | 0.00639 |

Epoch positions are inferred from the rounded epoch logs; exact selected
validation scores were checked against saved prediction metrics. A fold1
selects epoch30 with Train0.9023/Val0.7516 (gap0.1507). Fold4 remains weak and
overfits in B (0.8400/0.7179, gap0.1221) and C (0.8344/0.7204, gap0.1140).
Fold2 has Val above Train in all A/B/C, so mean gaps hide substantial fold
heterogeneity. No control demonstrates a broad resolution of instability and
overfitting. Existing KM plots demonstrate within-model risk separation, not
an incremental advantage of fusion over pathology.

## Audit and interpretation

- All five models use the same 721 patients (251 recorded events) and saved
  folds. IDs were joined through the original PA/CT pairing table; no row-order
  or modality-ID equality assumption was used. Train/Val membership, labels
  and times match in every fold.
- Recomputed Train/Val C-index and 36/60-month IPCW AUC reproduce all saved
  metrics. AUC uses each model's corresponding training censoring distribution.
  Validation risk groups match the corresponding training-risk median cutoff.
- A checkpoints have no CT encoder or query/CT-projection/attention parameters.
  This is a pathology ablation, not a frozen-CT teacher.
- In every B checkpoint all 51 BN buffers (34 means/variances and 17 counters)
  match public MedicalNet initialization, while conv1 and layer4.1.conv2 weights
  changed. Together with the verified implementation/gradient tests, this
  supports active CT training with fixed BN and actual uniform attention.
- B/C share public CT initialization and jointly train CT/pathology; neither
  loads this project's trained unimodal checkpoints. A retained initialization
  matches C, but removal of a route changes subsequent RNG consumption; epoch
  sample orders/dropout masks are not claimed to match. This is one training
  seed, not a multi-seed controlled effect estimate.

The strongest conclusion is that learned spatial attention has not demonstrated
added predictive benefit over the uniform-CT representation in this setup.
The second conclusion is that even this simple CT fusion has not demonstrated
a stable, substantial increment over the matched pathology-only head.
Therefore ineffective learned selection alone does not account for the whole
performance ceiling. Current CT/pathology information may be redundant, or
the learned CT representation/joint optimization may not preserve or exploit
the additional information. These results cannot distinguish those causes or
establish that raw CT has no complementary DFS signal. No independent spatial
matching supervision was assessed.

Detailed aggregates and audit methods: `summary.json`. Fold metrics,
trajectories and pair changes are also saved as CSVs in this directory.
