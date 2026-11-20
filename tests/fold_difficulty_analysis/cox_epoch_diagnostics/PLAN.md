# Three Cox/UNI diagnostics

Approved task: investigate risk stability, Cox batch sensitivity, and representation drift; verify complete results, then push this experiment only.

- [ ] Test risk scale/rank decomposition, censored comparable-pair accounting, exact fixed-model evaluation and RNG preservation.
- [ ] Reuse path_train.py unchanged; wrap Epoch logging to record eval Train/Val, full-risk vectors every epoch and representation/attention snapshots at0,1,3,5,7,10,last,best.
- [ ] Run batch32/64/128, same original five folds, Raw UNI/full patches, ABMIL/dropout.25/Adam1e-4/WD5e-4/seed123, max30 Epoch/patience10. Record optimizer updates. Same maximum patient exposure; no claim of equal optimizer budget. Report exact shared update-count comparisons separately.
- [ ] Verify added diagnostics preserve RNG and main-flow batch32 training trajectory. Evaluate saved risks independently, reload selected checkpoints, validate sources.
- [ ] Aggregate risk Spearman/SD/affine-fit and comparable-pair corrections; Attention entropy/effectiveN/JS/top128 overlap; pooled512/post256 CKA and drift; per-fold C/AUC/gaps and learning trajectories.
- [ ] Background three conditions; after all complete run report and validation, then narrow-scope Git commit/push. Never include private predictions, representations, source feature files, checkpoints or unrelated modifications. Abort publication safely on staged unrelated files or branch divergence.
