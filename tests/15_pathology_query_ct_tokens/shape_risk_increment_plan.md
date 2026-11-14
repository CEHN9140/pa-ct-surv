# Fixed-Shape CT risk conditional increment (diagnostic, scheme A)

User request: implement the recommended low-cost Conditional Prognostic
Increment in test15. Do not train a teacher, use Cross-Attention, retrain
pathology, or change the formal teacher/student initialization constraint.

## Predeclared comparison

- 721 patients, 251 events; exact existing five outer / three inner FIT-STOP-HELD
  roles from `nested_risk_increment`. PA risks are unchanged source OOF/final risks.
- A: source Cox-only CT; C: ALL three Shape targets, **fixed lambda=0.2**.
  This is not the previous automatically selected C (Val 0.7136).
- Fifteen saved C inner checkpoints are used for inference only. Each patient's
  OOF risk comes from a model whose FIT and STOP exclude that patient and all
  outer-validation patients. Fixed weight means HELD scores cannot select lambda.
- Standardize every inner model's HELD risks using its FIT eval-risk mean/std,
  exactly like source A and PA. Outer CT risks use the outer model's training
  eval-risk mean/std. Then `IncrementCox` fits its own scalers only on OOF rows.
- C outer epochs = floor median of the three **fixed-weight** STOP epochs:
  [26,16,9,27,28]. Reuse existing C outer0/3 (lambda0.2 + same epochs); freshly
  train outer1/2/4 from PUBLIC MedicalNet, for exactly16/9/28 epochs. No trained
  model initializes another fit. No outer validation/HELD scores select epochs.
- M0: calibrated PA risk; M1: PA + A CT risk; M2: PA + C CT risk. Both slopes
  are free; same predeclared CT ridge=1.0, PA slope unpenalized (existing
  IncrementCox semantics). No Cox hyperparameter search; all three heads are
  fitted afresh on identical source PA OOF rows.
- Final Train metrics use outer-refit eval predictions (descriptive in-sample),
  while fusion fitting uses OOF covariates. Outer validation is evaluated once.

## Outputs

Independent `results/shape_risk_increment/` and
`checkpoints/shape_risk_increment/`: source/role/weight/scaler/checkpoint audits,
OOF CT covariates, per-fold M0/M1/M2 models, coefficients, predictions, KM,
C-index, event-averaged Cox NLL, IPCW AUC36/60, paired fold differences,
2,000-repeat within-fold paired-patient Bootstrap and a report. Include the
fixed-C CT-only C-index as context; historical automatic-C is not substituted.
Bootstrap conditions on fitted models and excludes prior exploration/retraining.

## Work / acceptance checks

- [x] Add tests first for fixed-weight epochs ignoring HELD selection, disjoint
  OOF roles/coverage, patient/label matching, FIT-only scaling, OOF-only Cox
  fitting, scaler/model reload, and preserving true event/censor handling.
- [x] Implement audit, cached-inner inference, three public-only outer refits,
  common Cox export/evaluation and paired comparisons without source mutation.
- [x] Verify all721 identities/labels, 15 inner checkpoint roles/scalers and
  2 reusable outer checkpoints; reject config/hash/epoch/target mismatches.
- [x] Run unit checks plus a real saved-checkpoint risk/scaler smoke and common
  Cox export smoke before launching full background work.
- [x] Provide one background launcher/log; retain completion/provenance checks.

Scheme A is exploratory because lambda0.2 was proposed after earlier work on
this same cohort. It is honest held-out risk generation at fixed settings, not
an evaluation of automatic lambda selection and not independent confirmation.

Verification: six tests pass. Full source audit passes; source PA/A raw and
standardized final covariates reproduce modality predictions, and hashes now
cover every directly read final prediction table. Real fold0 M0 head output
reproduces the previous M0; identical-CT null control gives M1=M2 exactly with
145 outer predictions and KM/AUC/NLL exports. A 34-patient GPU smoke verifies
the saved inner checkpoint; at the original held batch size32 its first32
risks reproduce exactly (maximum absolute difference0), BN buffers unchanged.
Two-patient batches gave ~7.7e-5 differences, so this diagnostic retains the
original source batch32/order. Independent read-only review found no blocking
leakage/comparator/initialization issue. See shape_risk_increment_validation.json.
