# Experiment 16: CT-only Radiomics baseline

User-approved scope: simple independent CT-only baseline; no pathology, fusion,
neural training, checkpoint initialization, or changes to the main workflow.

## Design

- Preserve all 721 patients, 251 events and the five `split` assignments in
  `data/eagle/top25/all_label_roi64.csv`. The CSV supplies identities/outcomes
  only; no pathology features are opened.
- Read doctor-provided `ct/image/<ct_id>.nii.gz` and corresponding binary mask.
  Extract the whole 3D tumor, not the normalized neural ROI. PyRadiomics 3.0.1:
  Original only, 107 nondeprecated features, 1 mm spacing, B-spline image /
  nearest-neighbor mask interpolation, binWidth=25 in current input units,
  normalize=False, voxelArrayShift=0, no intensity clipping/resegmentation.
  The unknown upstream intensity processing remains a limitation.
- Fail on missing/mismatched identities, invalid labels/masks, failed or
  nonfinite extraction. No patients silently dropped. Geometry correction uses
  physical coordinates and is recorded.
- Fit a deterministic training-only filter: remove exact constant features,
  greedily retain sorted feature names whose absolute Spearman correlation
  with already retained features is <=0.90; then training-only Z-score.
- Ridge-Cox (Breslow), candidate penalties 0.01, 0.1, 1, 10, 100, 1000. Select
  by mean C-index in three event-stratified inner folds (seed=123); repeat the
  complete filter/scaler/model in every inner fold. Exact ties prefer stronger
  regularization. Refit on all outer training patients; outer fold is evaluated
  once. No LASSO/mRMR/univariable screening/ICC/ComBat/PCA/filter-image search.
- Save features/QC/config, per-fold fitted pipeline, selected feature audit,
  tuning records, train/validation predictions and KM plots, consolidated
  metrics (five folds plus mean/sample SD), AUC36/60 and HR via the existing
  shared survival evaluation. Report full-risk-set event-normalized Cox NLL.
- Separate extraction and fitting entry points let the cached feature table be
  reused without touching other experiments. An extraction manifest binds the
  table to cohort/settings/source-file size+mtime; it does not claim full-file
  content hashes. Reject stale or incomplete caches.

## Implementation / verification

- [x] Add meaningful tests first: training-only feature choices/scaling,
  constant/duplicate/nonfinite handling, nested patient separation, model reload,
  exact 107 extraction, mask geometry and patient/label/cache integrity.
- [x] Implement `radiomics_ct.py`, `run.sh`, and concise README.
- [x] Run focused tests and syntax checks in a NumPy 1.26.4 isolated runtime.
- [x] Verify extraction on real NIfTI cases and fit/evaluate a small real-data
  smoke run without claiming completion of the full 721-patient experiment.

No automatic commit, push, full experiment launch, or modification of test15.

Verification: six focused tests passed; only third-party SimpleITK/SWIG
deprecation warnings. A 31-patient real-data smoke completed 107-feature
extraction, all five outer folds and 90 inner candidate fits without extraction
or candidate failures. All five reloaded models reproduced validation risks;
training-only scaler parameters and retained-feature correlations were checked.
One physical-geometry correction and the anisotropic input were included.
The full 721-patient run has not been launched. See `validation.json`.
