# Attention regularization implementation plan

Approved goal: independent baseline / normalized-entropy / Neff-hinge comparison on Raw UNI/full patches, inner parameter selection, complete validation and experiment-only GitHub push. Do not alter main source or old experiments.

- [x] Tests: exact penalty formulas, uniform/singleton/peaked distributions, finite gradients, per-patient Cox-batch mean, zero-weight main-flow equivalence, validation loss separation, outer selection isolation and fixed-epoch final checkpoint.
- [x] Reuse unchanged ABMIL/Pa_Model and path_train.train_path. Forward hook captures attention penalties, custom criterion returns Cox+weight*patient-mean penalty only in train mode; eval loss remains Cox.
- [x] Predefine seven inner candidates: baseline; entropy weights.01/.1; Neff q=.05/.1 crossed with weights1/10. Fixed event-stratified80/20 FIT/STOP inside each outer training pool, seed123, same roles/case order/init across candidates. Select strongest STOP C, tie prefer no/weaker regularization and earlier Epoch. Zero candidate shared with baseline.
- [x] For each of three methods, freshly initialize on full Outer Train, fit inner-selected Epoch, no outer early stopping. Use last weights for fixed budget, final Outer Val evaluated once. Fivefold total35 inner+15 outer fits. Three outer tasks concurrently.
- [x] Log standard Epoch train/validation metrics plus train loss components; fixed eval C/Loss/entropy/q/Neff/Top128 means preserve RNG and never inspect Outer Val while selecting. Record train-only outer monitor distinctly.
- [x] CPU and real-GPU formula/trajectory/reload checks; independent code review; freeze source hashes and scientific config.
- [ ] Generate aggregate comparison, selected parameters, inner stability curves, final q/control strength, AUC/KM and conditional paired intervals. Recompute selected risks and checkpoint metadata.
- [ ] Background all folds, validate all50 fits and sources, then narrow whitelist commit/push. Never upload private IDs/predictions/weights or unrelated changes. Abort publication on Git divergence or other staged files.

Runtime completion is recorded in status.json and results/validation.json. Background report/publication entries remain pending until all50 fits finish.
