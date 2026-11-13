# Stage1 Shape ablation: true and cross-fitted predicted phenotypes

Goal: execute only the requested low-cost ablation, using existing caches.
Do not train MedicalNet, add Shape losses, start CT/teacher/student fitting,
or select a final auxiliary target based on outer-fold results.

Preserve721 patients,251 events,original five folds,strict PA OOF contexts,
source public-pretrained predicted Shape and train-only six-point Cox ridge
selection. V=log(volume/mm3),D=maximum3D diameter(mm),S=sphericity.

Predeclare7 unique nonempty groups V,D,S,VD,VS,DS,VDS. VD is exactly the
leave-S-out group, not a separate duplicate experiment. Compare both true
Shape and predicted Shape to distinguish phenotype value from decoding error.
True references are mask-assisted; no causal feature-importance claim follows.

For each of14 pipelines, fit/tune only the existing IncrementCox. Reuse the
20 saved predicted-Shape contexts (15 inner,5 outer): crossfit_shapes.csv are
the only fitting Shape covariates; predicted_shapes.csv provide held values.
Slice columns only, NEVER refit the Probe or choose columns from outcomes.
Validate IDs,finite values,columns,roles and source hashes. For descriptive
training metrics/KM thresholds only, evaluate the already fitted final Probe
on cached public512 train vectors; these are NOT Cox fitting covariates.
VDS true and predicted risks must reproduce their completed references for
BOTH train and validation, including selected ridge.

Output every group's mean/sampleSD Train/Val C-index,Cox NLL,AUC36/60,KM/HR,
paired2000-patient intervals versus PA and full VDS,positive-fold counts and
per-fold metrics. For predicted groups also compare matching true groups.
Predeclared conditional contrasts: VDS-DS (add V),VDS-VS (add D),VDS-VD(add S),
VD-V(add D given V),VD-D(add V given D). Record phenotype correlations only as
descriptive diagnostics, never as feature selection. Each contrast refits and
retunes its own ridge, so it measures whole predictive-pipeline utility, not a
unique causal coefficient. Intervals are nominal/exploratory and conditional
on existing models; they exclude retraining and earlier exploration uncertainty.

Independent results/shape_ablation and logs/shape_ablation.log. Main flow,
all previous results and cohort CSVs stay unchanged. Save only Cox heads;
patient-level prediction tables stay locally ignored as before.

- [x] RED tests: exact7 groups and deduped drop-one mapping; cached row identity,
      nonfinite values/overlap/incomplete roles must fail; selected covariate
      slices preserve correct train/held membership and values.
- [x] Implement cache audits,14 nested Cox pipelines,paired/conditional
      comparisons,reference parity,correlation and readable report.
- [x] Focused/full test15 suite,real-cache audits and independent code review.
- [x] Launch CPU-only background experiment; no neural fits or Probe fits.
- [x] Verify all70 model-fold outputs and10 reference-parity checks,reload Cox
      heads,recompute metrics/paired comparisons/check hashes; report evidence.

Numerical parity guard: require identical selected ridge,absolute risk error
<=1e-8 and identical every-patient-pair ordering/ties (tie threshold1e-8).
Preflight true-VDS fold1 refit differs by2.9e-9 at the existing Cox tol1e-9,
while the old head applied to current CSV inputs differs by4e-16. All other
source/folds differ <=9e-15. No model hyperparameters/tolerances were changed.

Pre-run evidence:97 test15 unit tests passed in73.659s,including four new
ablation/cache/parity tests. All20 cached contexts and15 reference folds passed
audit. All10 VDS source/fold refits passed numeric AND full-pair ordering checks.
Independent review found no important/blocking issue.

Completed: all70 Cox heads reconstructed from the correct true/OOF covariates;
training scalers,train/val risk,C-index,Cox NLL and140 AUC values recomputed.
All70 KM plots exist; all14 subset and10 conditional paired comparisons
reproduced;10 full-model reference checks passed;source hashes unchanged.
See results/shape_ablation/completion_review.json and report.md.

Exploratory result: adding true S to VD gives ΔC+0.0056,nominal95% interval
[+0.0004,+0.0110],4/5 positive folds; predicted S gives-0.0018
[-0.0067,+0.0036],2/5 positive. True/predicted V-D Pearson correlations
0.9367/0.9718. These repeated,unadjusted comparisons are a candidate signal,
not independent confirmation or automatic selection of an auxiliary target.
No stage2 CT-only,teacher or student training was started.
