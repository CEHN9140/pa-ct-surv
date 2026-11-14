# Experiment16: completed matched controls and inference diagnostics

All721 patients/251 events retain the saved five outer folds. All60 neural fits and15 final model reloads were verified. No new neural training was started for these diagnostics.

## Five-fold mean ± sample SD

| Model | Train C-index | Val C-index | AUC36 | AUC60 | Mean gap |
|---|---:|---:|---:|---:|---:|
| Matched PA-only | 0.7914 ± 0.0291 | 0.7288 ± 0.0555 | 0.7738 ± 0.0405 | 0.7433 ± 0.0754 | 0.0627 |
| PA + Radiomics Concat | 0.7888 ± 0.0235 | 0.7478 ± 0.0491 | 0.7930 ± 0.0301 | 0.7615 ± 0.0664 | 0.0410 |
| PA + Radiomics Cross-Attention | 0.8017 ± 0.0426 | 0.7476 ± 0.0456 | 0.7862 ± 0.0343 | 0.7677 ± 0.0478 | 0.0541 |

## Paired patient bootstrap

Patients were resampled within each fixed outer fold; folds were averaged equally,2000 replicates,seed123. Five direct sksurv checks per model/fold verified the multiplicity calculation (75 checks). The intervals condition on saved models, exclude retraining/prior exploration uncertainty, and are nominal without multiplicity adjustment.

| Contrast | Mean delta C-index | Nominal95% interval | Positive folds | Corrected / broken pairs |
|---|---:|---:|---:|---:|
| PA + Radiomics Concat minus Matched PA-only | +0.0190 | [+0.0009, +0.0377] | 4/5 | 2082/1594 |
| PA + Radiomics Cross-Attention minus Matched PA-only | +0.0188 | [-0.0023, +0.0410] | 4/5 | 2352/1871 |
| PA + Radiomics Cross-Attention minus PA + Radiomics Concat | -0.0002 | [-0.0123, +0.0120] | 1/5 | 1014/1021 |

Pair counts share patients and are not independent observations; their net change does not exactly equal the equally weighted fold-average delta.

## CT whole-vector permutation

For each fold,100 derangements swap complete CT Radiomics vectors among outer-validation patients, with no fixed points. The same permutations are used for both teachers. Pathology, labels, saved FIT preprocessing and eval-mode weights remain fixed. PA-only CT invariance was verified in all five folds.

| Model | Metric | Correct pairing | Mean after permutation | Mean drop |
|---|---|---:|---:|---:|
| PA + Radiomics Concat | val_cindex | 0.7478 | 0.6279 | 0.1199 |
| PA + Radiomics Concat | auc_36m | 0.7930 | 0.6540 | 0.1390 |
| PA + Radiomics Concat | auc_60m | 0.7615 | 0.6358 | 0.1257 |
| PA + Radiomics Cross-Attention | val_cindex | 0.7476 | 0.6211 | 0.1265 |
| PA + Radiomics Cross-Attention | auc_36m | 0.7862 | 0.6411 | 0.1452 |
| PA + Radiomics Cross-Attention | auc_60m | 0.7677 | 0.6330 | 0.1348 |

Permutation variability is across100 diagnostic perturbations, not the five-fold SD or a retraining confidence interval. These perturbations establish prediction dependence on correctly paired CT inputs, not a causal effect, unique patch-specific interaction or the size of the independent CT contribution. Incorrect pairings can be out of the joint training distribution.

## Interpretation

Concat has a positive contrast versus this matched PA-only (nominal lower bound close to zero), while Cross-Attention has no demonstrated benefit over Concat. Both teachers use patient-specific CT information. Fold4 remains weak: Concat Train/Val0.8148/0.6889, Cross-Attention0.8751/0.6874. Their independently selected outer budgets differ (14 versus23 epochs); these gaps cannot be attributed exclusively to attention capacity.

The matched PA-only score0.7288 is below the historical v7 ABMIL score0.7528. Their head/epoch-selection protocols differ; the new contrasts do not establish superiority over that original pathology workflow. Radiomics uses complete tumor/mask information, which also differs from a64-cubed CT student input.

These diagnostics do not fit risk-level Late Fusion, retrain a teacher, or initialize any model from a project-trained unimodal checkpoint.
