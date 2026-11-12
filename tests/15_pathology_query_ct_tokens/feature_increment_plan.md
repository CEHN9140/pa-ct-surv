# High-dimensional CT increment and conditional Shape diagnostic

User-approved on 2026-11-12. Scope: experiment15 only; no change to formal
jointly trained teacher/CT-student models and no frozen-unimodal replacement.

## Models and evaluation

Reuse the completed nested-risk diagnostic's721 patients, five outer splits,
honest inner OOF pathology risks, final outer pathology predictions and trained
fold-restricted CT checkpoints. Verify source config/cohort/completion/provenance.
All models freely calibrate pathology slope beta_p:
- M0: calibrated pathology risk only, unpenalized one-variable Breslow Cox.
- M1: calibrated pathology and CT risks, unpenalized two-variable Breslow Cox.
  Refit both slopes; do not compare M2 with the previous fixed-pathology-slope M1.
- M2_public / M2_finetuned: pathology risk plus512-D layer4 global-average CT
  features. Fit CT standardization, PCA16/32 and PC standardization within each
  calibration-training fold. Ridge grid .01/.1/1/10/100/1000 penalizes CT terms
  only; pathology slope is unpenalized consistently across M0/M1/M2. Select by
  mean inner-fold C-index; exact ties prefer stronger ridge, then fewer PCs.

Public CT features use only official public weights, no DFS training. For the
fine-tuned comparison, each inner model supplies BOTH calibration-train and
calibration-validation512-D features, and one final outer CT model supplies
all final train/validation features. Never pool channel coordinates from
different fine-tuned encoders into a single PCA fit.

For M2 hyperparameter selection, the pathology risks on each calibration-train
subset need their own honest OOF predictions, generated without that subset's
held-out calibration-validation patients. Make two subfolds with separate fit/
stopping/OOF roles using the existing local trainer. Reuse this cache for both
CT sources and Shape. This adds30 small pathology fits, not another40 CT/PA
fits. Keep max30 epochs/patience10/dropout.25/Cox32/seed123. For calibration
validation use the previous strict inner OOF pathology predictions. Final head
fits use the previous full outer-training OOF risks; all outer baselines and
evaluation risks are shared, preserving the observed1-epoch refits as a reported
stability limitation instead of changing them only for a candidate model.

Outer patients never select PCA/ridge/coefficients. Report C-index/AUC36/60,
KM/HR, per-fold gains, conditional paired bootstrap and selected parameters.
M0/M1 have no tuned hyperparameters; their calibration uses only outer-training
OOF predictions. Record raw source predictions, IDs, checkpoint/cohort hashes,
all transformation fitting sets, inner scores, heads and completion markers.

## Conditional second stage

Predeclare stable gain as positive mean delta, lower conditional bootstrap
bound>0, and improvement in>=4/5 folds versus BOTH M0 and M1. If neither
M2 source meets this criterion, run the exploratory Shape stage automatically.
This adaptive research sequence is not an independent confirmatory test.

Extract only three predeclared Shape measures from original binary NIfTI masks
with their native physical spacing: voxel volume(mm3), maximum3D diameter(mm),
and sphericity using PyRadiomics. Validate all721 mask paths; keep every tumor
component, reject nonbinary/empty masks rather than silently dropping cases.
Crop only the mask bounding box for computation, retaining its geometry; the
official shape implementation pads surfaces. No CT intensity normalization,
resampling, masks or preprocessing outputs are rewritten.

Evaluate M3 (pathology + log-volume, two free slopes) and M4 (pathology + all
three Shape features, ridge selected on the same clean inner contexts). Also
probe log-volume from all512 public/fine-tuned CT features using training-only
standardization/Ridge selection (no PCA to discard volume signal); report outer log-volume R2/MAE and raw
volume R2/MAE. No arbitrary R2 threshold implies information absence. Do not
add intensity/texture searches or new teacher modules at this stage.

## Execution checklist

- [x] Write failing tests for training-only PCA/scaling, calibrated M0/M1/M2
  dimensions and penalties, fold-isolated tuning, stability gate, generic paired
  references, true512-D extraction bypassing risk head, physical Shape scaling,
  invalid masks, and independent linear-volume prediction.
- [x] Implement `increment_cox.py` (small Cox/PCA models and tuning),
  `feature_increment.py` (source audits, feature caches,30 PA cross-fits,
  M0/M1/M2 and automatic staging), `shape_increment.py` (Shape and probe).
  Reuse existing local trainers and shared evaluation; preserve old experiments.
- [x] Add simple `run_feature_increment.sh`, separate log/results/checkpoints,
  provenance and patient-cache ignore rules. Default all outer folds/sources.
- [x] Verify all experiment15 unittests and real-data partial integration in
  /tmp, with strict source-checkpoint reload/prediction parity, CT basis/scaler
  provenance, outer exclusion, head serialization, Shape geometry and metrics.
- [x] Start the full diagnostic as nohup, verify actual CUDA PID, source hashes,
  data and first feature extraction/PA fitting progress. Preserve all source runs.

Verification: all58 experiment15 unit tests pass. Real ROI64 outer-fold0
integration with one-epoch sub-OOF fits completes six new PA fits and exports
M0/M1/M2_public/M2_finetuned. Audits verify honest sub-OOF roles, source512-D
feature caches, strict head reload reproducing all145 validation predictions,
training-only PCA/scaler sample counts and unchanged source artifacts. Three
original NIfTI masks pass physical Shape extraction. A separate60-case
synthetic mask/feature integration exercises M3/M4, conditional tuning, KM/HR
and full-dimensional linear-volume probes; all outputs are finite where the
synthetic follow-up supports them. All verification artifacts stay in /tmp.

Official APIs checked: scikit-survival CoxPHSurvivalAnalysis supports per-feature
ridge penalties and Breslow ties; PyRadiomics shape features are mask-derived,
use physical spacing and pad the mask for surface calculations.

Full run launched with `run_feature_increment.sh`. CUDA startup,721 paired
patients, both CT sources, PCA16/32, six ridge values, all five outer folds,
reused source pathology epochs and first new PA checkpoint/epoch updates were
verified. The full diagnostic is running; conditional Shape has not yet been
decided. Existing source runs and main-project files remain unchanged.
