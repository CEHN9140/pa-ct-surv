# MedicalNet layer3/layer4 representation comparison

Input decision: user obtained the existing NIfTI files directly from clinicians
and cannot trace earlier transformations. Retain these files as FIXED inputs,
document unknown upstream intensity/acquisition provenance, and proceed with
the user-proposed CT-only comparison. The geometry/reconstruction audit passed
all721 patients/251 events; do not rewrite inputs, labels, folds or main code.

A: existing completed nested-risk CT-only baseline, Layer4 GAP512, reused by
reference. It already uses the identical721 cohort, saved five outer splits,
three fit/stop/held inner roles, public MedicalNet, fixed BN, no pooled L2,
head512->256->ReLU/dropout.25->1, Cox, true CT batch32+merged tail, seed123,
Adam head1e-4/backbone1e-5/wd5e-4, max30/patience10. Source predictions are
comparators only; never initialize B/C or any teacher with trained A weights.

B: Layer3 GAP256 -> Linear256,256 -> ReLU -> Dropout.25 -> Linear256,1.
C: concatenate Layer3 GAP256 and Layer4 GAP512 -> same hidden256/dropout/head1.
Initialize the common encoder from the same PUBLIC MedicalNet. Layer3 skips
and removes unused layer4 after public loading. Other variants train through
layer4. BN statistics remain fixed, affine parameters train. No CT feature L2,
new normalization, augmentation, attention, auxiliary loss or learning schedule.

Use a local ResNetCox subclass overriding only feature extraction/head input
width. For A, behavior/weights/RNG match CT_Model exactly. For B/C, replacing
heads occurs inside a CPU RNG fork so subsequent shuffle/dropout streams stay
aligned with A. Initialize new head weights normal0,.01 and bias0 as before.
Common-head hidden width256 remains fixed; parameter-count differences due to
input width and layer3 truncation are recorded rather than hidden.

Two independent background B/C jobs, separate logs/results/checkpoints,40 new
fits total. Each source: three inner fits per outer fold, early stop ONLY on
stop C-index; report held predictions but do not use outer validation for any
choice. Pre-save median inner best epoch, freshly refit all outer training,
then outer eval once. Store raw and standardized honest CT OOF risks (scaler
fit on inner FIT predictions), honest split provenance and final predictions.

Source audits require completed input/nested runs, unchanged cohort/public
weight/source config hashes, identical saved roles and A outer predictions.
Every final row uses eval-mode whole-train/whole-val C-index and Cox/AUC36/60/
KM/HR. Report B-A and C-A paired patient Bootstrap and per-fold changes. Do not
pick an apparent outer winner and call it an independent result. Any later
pathology-conditional increment test must reuse honest PA OOF predictions and
restrict representation/regularization choices to outer training. Do not start
increment tests or retrain a teacher until these results have been reviewed.

Files: ct_layer_model.py, ct_layer_train.py, test_ct_layer_model.py and
run_ct_layer_comparison.sh, all within experiment15. Main model/dataset and
existing experiment results remain untouched.

Checklist:
- [x] Failing tests for exact A parity/RNG, B/C dimensions/pooling order,
  layer4 omission in B, backbone gradients/BN policy and correct LR grouping.
- [x] Implement model/runner, reuse original nested trainer and eval helpers;
  verify identities/labels, source roles, transform fitting scope and sources.
- [x] Run complete tests and real public-weight ROI64 training/reload smoke;
  inspect actual GPU memory before launching B/C together.
- [x] Start nohup+setsid B/C jobs and verify real PIDs/GPU/initial epochs.
  Record fixed physician-provided input limitation and source A reference.

Implementation and independent review completed;78 tests pass. Real A
inference/API parity matches saved predictions to CSV precision; B/C real
ROI64 smoke verifies public-only initialization,32+34 risk sets, nonzero stem
updates, unchanged BN buffers and exact checkpoint reload predictions. Full
B/C nohup+setsid startup verified: PIDs1065936/1065937, both PPID1 and on CUDA,
inner histories/checkpoints progressing. A existing results are reused by
reference only. No conditional increment or teacher training started.
