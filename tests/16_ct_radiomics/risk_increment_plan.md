# Radiomics risk conditional increment (M3)

Goal: extend test15's existing fixed-ridge M0/M1/M2 with full Original107
Radiomics risk, using identical PA OOF/outer predictions and outer patients.
This is a diagnostic, not a replacement for the jointly trained teacher/student.

Default source is the original v7-architecture ABMIL under strict nested epoch
selection (PA C-index0.7390, outer epochs1/14/1/14/23). Historical0.7528 used
outer stopping and is not substituted. User selected reuse of this existing
strict OOF source, without repeating neural training.

- Audit721/251 cohort SHA, test16 extraction signature/schema, source role
  partitions and calibrated PA/A/C predictions. Reject duplicate/mismatched
  identities, outcomes, roles, overlap, missing OOF cases or stale artifacts.
- Reuse saved FIT/STOP/HELD roles. For each of15 Radiomics OOF fits, train on
  FIT+STOP only; choose CT Cox ridge via event-stratified3-fold CV inside those
  patients, refitting constant/Spearman>.90 filtering and Z-score per candidate
  fold. Predict HELD only after selecting/refitting. Scale risks using that
  base model's fitting patients, never HELD/outer validation statistics.
- Five full outer CT refits choose ridge inside outer training only, then
  independently predict outer validation. Reuse no outer721 OOF table for
  stacking and no sample-in-training risks to fit the fusion head.
- Fit M0(PA), M1(PA+A), M2(PA+C), M3(PA+Radiomics) with IncrementCox,
  CT penalty1, PA unpenalized, both coefficients free; scalers fit on stacking
  OOF rows. No fusion hyperparameter selection. Verify M0/M1/M2 predictions
  reproduce prior fixed-Shape results before interpreting M3.
- Save base pipelines/scalers/penalties/roles/OOF privately; export fold metrics,
  KM, summary, paired2000-patient-bootstrap contrasts and corrected/broken
  comparable pairs. No Brier score, group search or neural retraining now.
- Use local .runtime, CPU-only setsid/nohup launcher, separate
  conditional_risk_increment directory/log. Refuse overwrite. Publish only
  small aggregate results; do not automatically commit/push.

Verification: tests first for identity/label checks, honest role coverage,
invariance to outer-validation labels, CT OOF fit/selection isolation, risk
scaler provenance, fixed IncrementCox penalty and checkpoint reload. Real
source audit, focused regressions, independent review, then run authorized
scope and inspect progress/results.

Completed:23 tests pass; real source audit and independent review pass. All
20 Radiomics base models,360 candidate CV fits and20 fusion heads completed.
Post-run reload verified base membership/filter/scaler/CV choices, all heads,
fold metrics and147 source hashes. M0/M1/M2 reproduce original exports. M3
Val C-index0.7513±0.0420; M3-M0+0.0123 (nominal conditional95% interval
[+0.0015,+0.0234],4/5 positive), M3-M2+0.0012 (interval crosses zero).
See risk_increment_validation.json and conditional_risk_increment/report.md.
