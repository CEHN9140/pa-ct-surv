# Experiment1 only: predefined CT phenotype increment over pathology risk

User scope: only experiment1 now. Do not run MedicalNet phenotype probes,
cross-fitted predicted-phenotype replacements, neural training or new teachers.
Keep proposed method WSI+CT teacher -> CT-only student unchanged. Everything
below is a diagnostic within test15, using721 patients/251 events/original folds.

Predeclare features BEFORE extraction/model evaluation:
- Shape3: log native-mask voxel volume, maximum3D diameter, sphericity. Reuse
  feature_increment/cache/shape_features.csv and explicitly mark mask geometry.
- Tumor First-order6: Mean, sqrt(Variance),10Percentile, Median,90Percentile,
  Entropy. Use sqrt(Variance) rather than deprecated StandardDeviation.
- Tumor Texture6: GLCM Contrast, Idm (=Homogeneity2), Correlation; GLRLM ShortRun-
  Emphasis, LongRunEmphasis, GrayLevelNonUniformityNormalized.
- Peritumoral12: same6+6 measured in the observed5mm outside-mask shell.
- Combined27: exactly all above, no additional features or outcome screening.

PyRadiomics installedv3.0.1, Original image only, binWidth25 in recovered clipped
intensity units, distances[1], symmetricalGLCM=True, weightingNorm=None,3D,
no extra normalization/resampling/filters. Current saved ROI float32[0,1]
is the sole intensity input. Recover clipped integer values via round(1400*x-
1000), checking round-trip error before extraction: this removes float32
inverse-roundoff around histogram edges, not recovers lost original intensities.
Current NIfTI upstream315 bound/original HU calibration remain unknown.

Use saved ROI mask to locate tumor and Euclidean outside shell0<distance<=5mm.
Exclude artificial padding using only raw NIfTI HEADER geometry and the saved
crop center. Calculate theoretical full5mm shell from the padded tumor mask
and record observed valid-ROI coverage; verify coverage against input audit.
No CT pixels outside64^3 are accessed. All721 cases remain; coverage is QC,
not a predictor. Fail on missing/nonfinite features, never silently drop cases.
All region groups use mask localization; positive value is mask-assisted and
does not prove an unmasked CT encoder should recover identical information.

Models M0 PA risk; M1 PA+Shape; M2 PA+FirstOrder; M3 PA+Texture; M4 PA+Peri5;
M5 PA+Combined. Also CT-only Cox for all5 feature groups. Reuse completed
nested-risk PA OOF/final predictions and feature_increment's15 nested PA
calibration-training OOF caches; raise if any cache absent, NEVER retrain PA.
Audit saved role disjointness and source hashes. Conditional heads use existing
IncrementCox: unpenalized free PA slope, standardized phenotype terms ridge.
CT-only heads standardize features and ridge-penalize all coefficients. No PCA,
imputation, outcome selection, univariate filtering or adaptive feature subsets.
Ridge grid.01/.1/1/10/100/1000 fixed; mean inner-fold C-index selects, ties favor
stronger ridge. All transforms fit inner training only. Outer final head fits
outer-training PA OOF + phenotypes; outer evaluation uses corresponding saved
outer PA predictor, same as source M0/Shape. Verify recreated Shape PA+CT
predictions match original M4_shape, preserving the reference protocol.

Report CT-only and conditional C-index (mean/sampleSD), paired conditional Δ
versus M0 and versus Shape, positive folds, conditional patient Bootstrap2000,
validation event-normalized Cox partial NLL, AUC36/60 and KM/HR. Combined must
also be compared with Shape so a positive geometry effect is not misattributed
to new intensity/texture information. Predeclared exploratory signal criterion:
meanΔ>0, paired lower>0 and>=4/5 positive folds vsM0; report Shape comparison
separately. Multiple groups and earlier exploration prevent confirmatory claims;
bootstrap excludes retraining/configuration-selection uncertainty.

Files: ct_phenotype_features.py; ct_phenotype_increment.py;
test_ct_phenotype_increment.py; run_ct_phenotype_increment.sh. Separate results
root results/ct_phenotype_increment and log, small joblib heads there, no neural
checkpoint root. Case features/coverage/split tables stay locally ignored.

Checklist:
- [x] Failing tests for integer intensity recovery, PyRadiomics feature values/
  dimensionality, shell coverage and padding exclusion, no exterior pixels,
  CT-only penalty/scaling vs unpenalized PA, and cached-OOF isolation.
- [x] Implement extraction and nested evaluation, exact source M0/Shape checks,
  component metrics, paired deltas and predeclared aggregate result table.
- [x] Verify tests, real native/anisotropic ROI extraction and five-fold Shape parity;
  independent review; no source writes/neural fits/MedicalNet probe calls.
- [x] Run experiment1 in background with a single dedicated log, verify actual
  CPU progress, completion marker and all721/no-dropped-case extraction.

Official references verified: PyRadiomics features and customization docs,
including Variance, Idm, per-angle averaging and explicit binWidth settings.

Verification:85 experiment15 tests passed in74.976s; all15 cached PA contexts
passed isolation checks. Five-fold Shape predictions reproduced with maximum
absolute error2e-15. Three real-case extractions included anisotropic input and
the worst5mm shell coverage case. Independent review found no important issue.
Original UNI has mixed NumPy1/2 native extensions: use ignored isolated
.phenotype_runtime with NumPy1.26.4 and existing dependencies; UNI is unchanged.

Completed:721/721 extracted in46.8s, no failures or dropped patients. All55
model-fold outputs passed independent prediction/label/C-index/Cox-NLL checks;
all50 saved heads reproduce validation predictions and ridge selections. All55
AUC36/60 estimates are finite and KM plots present. Source hashes unchanged.
See results/ct_phenotype_increment/report.md and completion_review.json.
Combined-Shape ΔC=-0.0029, conditional paired95% interval[-0.0107,+0.0054].
No experiment2 or teacher training was launched.
