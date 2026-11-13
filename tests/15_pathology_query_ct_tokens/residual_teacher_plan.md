# Residual teacher implementation and evaluation plan

Goal: one experiment15 teacher ablation with jointly trained PA/CT branches,
pathology auxiliary Cox and a penalized CT risk correction. Keep main files,
previous experiments and original CT preprocessing unchanged.

Approved architecture from the user's specification:
Virchow2-CLS Top25 [25,1280] -> patch L2 -> existing ABMIL (1280->512,
attention512->128->1, patient512->256->1) produces rp. Public MedicalNet
ResNet18 layer4 global512 -> Linear512,128 -> ReLU -> Linear128,1 produces dc.
Final CT linear weight/bias start at zero; no CT-head dropout was specified.
ABMIL dropout remains0.25. Total risk rp+dc, no sigmoid. Both branches train;
BN statistics stay fixed but affine weights are trainable. No project-trained
unimodal weights may initialize any teacher. At the first backward pass, zero
CT output weights imply zero encoder gradient; after updating them, CT encoder
gradients can flow. This is not freezing or staged training.

Objective: Cox(rp+dc) +0.3*Cox(rp) +lambda_r*mean(dc**2). Defaults compare
lambda_r=0 and0.01; the mean makes the penalty independent of tail size.
Use existing event-normalized Breslow Cox, patient-forward accumulation32,
tail merged into the previous risk set, head/PA LR1e-4, CT backbone LR1e-5,
Adam weight_decay5e-4, seed123, max30/patience10 inside training only.

Evaluation: same721 patients and saved five outer splits as completed nested
risk experiment. Reuse identical three inner fit/stop/OOF role assignments.
For each lambda, train three fresh inner teachers with early stopping on stop
only; assess held OOF risk once. Select lambda by mean held-fold C-index before
any outer prediction; ties prefer stronger penalty. For each fixed lambda,
freshly refit all outer-training patients for floor(median(inner best epochs)),
then evaluate outer validation once. The inner-selected primary result picks
the preselected lambda; per-lambda outer results are secondary comparisons,
never used to change lambda or epochs. Serialize selection before outer fits.

The user explicitly chose ONLY the two residual penalty groups for this run.
Do not retrain concat/crossattention. Historical outer-stopped teachers are
descriptive references only, not matched controls.
The existing nested pathology baseline is a prediction reference only, never
an initialization. Record this distinction in all comparison metadata.

Files:
- residual_teacher.py: residual module, objective, comparable-pair accounting.
- residual_train.py: source audits, nested fitting, final evaluation/Bootstrap.
- test_residual_teacher.py: numerical, gradient, BN, tail and isolation checks.
- run_residual_teacher.sh: simple nohup command, separate log/results/checkpoints.

Checklist:
- [x] Write and run failing tests for initial equality, loss/regularizer scaling,
  zero-init gradient behavior and subsequent CT/PA learning, frozen BN buffers,
  event/censor/tie comparable-pair counting and train-only epoch selection.
- [x] Implement local architecture and training with all loss components logged;
  final metrics use eval predictions and shared IPCW AUC/KM/HR analysis.
- [x] Verify source cohort/config/public-weight hashes, role disjointness,
  prediction identity/time/status alignment and no overwritten outputs.
- [x] Run all experiment15 tests and a separate real-data one-epoch smoke test;
  inspect parameter updates, optimizer grouping and serialized head parity.
- [x] Document exact defaults/protocol and runnable commands. Launch full jobs
  only for the scope requested by the user; keep previous outputs intact.

Implementation and real-data verification completed. User selected two residual
penalty groups only; no Concat/Cross-Attention retraining. Full nohup startup
will be verified from its actual PID, CUDA use and first training epoch.

Full run startup verified: PID548763, PPID1, dedicated session via nohup+setsid,
CUDA memory allocated, first two inner epochs and checkpoint/history written.
All67 experiment15 tests pass. Both requested penalties and all five outer
folds are configured; no extra controls or source model weights are loaded.
