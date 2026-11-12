# Nested risk-increment diagnostic: approved design and plan

Goal: test whether independently learned CT risk adds predictive information
to pathology risk, without initializing or replacing the formal teacher/student.
Approved by the user on 2026-11-12. All additions stay in experiment 15.

## Statistical design

Keep the original 721-patient paired cohort and five outer splits. For each
outer training set, use event-stratified three-fold cross-fitting (seed123).
For each inner held-out fold, split the remaining patients into fit/early-stop
sets (85/15, event-stratified). Only fit patients receive gradients, only the
stop set selects checkpoints, and the held-out inner patients supply OOF risks.
Each inner model's risk mean/population SD is fitted on its FIT predictions
only; apply that transform to its OOF predictions. Thus OOF patients enter
neither gradient updates, checkpoint selection nor scaler fitting.

Fit one unrestricted real coefficient alpha by minimizing the existing
event-averaged Breslow Cox loss on all standardized inner-OOF risks:
`risk = z_pathology + alpha * z_ct`. There is no alpha grid or outer-validation
tuning; negative alpha is allowed as conditional correction. Check convergence
and loss no worse than alpha0. Use the median inner-selected epoch count for
each modality, rounded down if needed. Initialize NEW outer models and train
all outer-training patients for those fixed counts, with no validation-based
stopping. Fit the outer risk scalers on outer-training predictions only, then
apply the frozen scalers/alpha to the outer validation set exactly once.

Settings match the selected unimodal comparisons: Top25 1280-D pathology,
patch L2, ABMIL, CT ROI64/ResNet18/public MedicalNet, CT pooled L2 off, CT BN
stats fixed but encoder trainable, dropout .25, Adam lr1e-4/backbone1e-5,
wd5e-4, 30 inner epochs/patience10. PA uses joint32-patient Cox forward
accumulation; CT uses true batch32. Both merge tails. Seed123 resets per fit.
Default run: 5 * (3 inner fits + 1 fresh outer refit) * 2 modalities = 40 fits.
No existing downstream checkpoint is consumed.

Report retrained pathology/CT and fused outer predictions, C-index, AUC36/60,
KM/HR and five-fold mean/sample-SD. Keep train scores explicitly in-sample.
Save split roles, OOF risks, fit-only scalers, alpha optimizer details, selected
epochs, histories, public initialization provenance and per-fold calibration.
Pairwise bootstrap comparisons are conditional on the final outer predictions;
do not pool raw risks across models/folds. Outer folds still form internal CV,
not external validation, and historical model/config selection remains a limit.

## Implementation plan

- [x] Write failing `test_risk_increment.py` tests for three-way split isolation,
  outer-label-independent splitting, training-only standardization, scalar Cox
  optimization including negative coefficients and degenerate inputs, and
  real tiny-model early-stopping/fixed-refit behavior and tail retention.
- [x] Add `risk_calibration.py`: `inner_splits`, `risk_scaler`, `standardize_risk`,
  `fit_alpha` and paired C-index bootstrap. Use the existing Cox implementation.
- [x] Add `risk_increment.py`: reuse main datasets/models/optimizers/evaluation,
  use a local joint-Cox trainer supporting early-stop OR fixed-epoch fits,
  build OOF calibration, fresh outer refits and final comparison exports.
- [x] Add simple `run_risk_increment.sh` nohup launcher. Distinct paths/log,
  default all five folds, refuse existing outputs. Exclude this diagnostic from
  the existing teacher-only inference/score-discovery defaults.
- [x] Run focused and all experiment-15 unit tests, shell syntax checks and
  diff review. Verify a real ROI64 nested one-epoch outer-fold smoke in /tmp,
  including split/scaler/alpha provenance, strict fresh-checkpoint reload and
  exported survival metrics. Do not reuse smoke checkpoints in the full run.
- [x] Launch the full independent five-fold diagnostic in the background,
  verify CUDA, saved config/cohort and first successful inner updates. Document
  commands/outputs in README. Do not commit or push unrelated work.

Verification: all 50 experiment-15 unittests pass, including a regression test
for preserving the original raw-risk C-index tie tolerance in paired contrasts.
Real ROI64 smoke: outer fold0, two inner folds, one epoch per model (six fresh
fits), 576 outer-training and145 outer-validation patients. All inner fit/stop/
OOF roles are disjoint; checkpoint reload reproduces stored OOF predictions;
recomputed fit-only and outer-train-only scalers match saved values; recomputed
alpha matches OOF calibration; all CT BN buffers stay fixed and convolution
weights change; outer refit histories contain no validation; scalar fused-risk
reconstruction and paired C-index differences match exports. Smoke artifacts
remain under /tmp and are not used in the full run.

Full run launched through `run_risk_increment.sh` with nohup. CUDA startup,
721-patient cohort hash, all five outer folds, three inner folds, exact training
settings and the first successful inner updates/checkpoint were verified.
Full-run fitting was in progress at implementation completion. It has since
completed all40 fits and five outer folds; the result review and audit are in
`results/nested_risk_increment/report.md` and `review_audit.json`.
