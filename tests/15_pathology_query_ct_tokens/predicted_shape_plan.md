# Cross-fitted predicted Shape conditional DFS increment

Goal: stage3 diagnostic using PUBLIC-pretrained MedicalNet layer4 GAP512.
Preserve721 cases/251 events/original outer five folds and strict cached PA
OOF risks. Never change or initialize a formal teacher/student from baselines.
DFS-finetuned stage3 needs additional inner-inner CT training. Proceed with
the stated public-only default; no extra DFS encoder fits are started. A count
of30 describes one submodel level, not a guarantee for fully nested tuning.

Comparators: PA only; PA+true Shape; PA+predicted Shape; PA+public512 features
(existing train-only PCA16/32+ridge). No new feature filtering or outcome-based
group choices. Shape remains log-volume, maximum3D diameter and sphericity.

Each Cox INNER context:
- Training PA risk = existing calibration-training sub-OOF cache. Shape
  predictions for these same patients use the corresponding saved two-way
  PA sub-OOF partitions: Probe fit on FIT union STOP, predict OOF. Public CT
  vectors are fixed and have seen no cohort DFS labels.
- For EACH such Probe, choose3 target-specific ridge penalties using three
  KFold splits STRICTLY within that Probe's fitting patients, by mean target
  MSE. Fit feature scaler and Ridge on fitting patients only. Predict held
  patients once; their Shape labels are not read by fitting/tuning functions.
- Inner validation Shape: another Probe fitted/tuned only on the entire
  inner-training set. Its prediction uses CT features only.
- Select Cox CT-term ridge with existing six-point grid and inner C-index.

Outer final: use original three inner-OOF partitions to cross-fit Shape for all
outer training patients, with Probe tuning nested within each training part.
Fit Cox on PA OOF + Shape OOF. Fit/tune one final Probe on all outer training
patients; predict outer validation CT vectors, then use the Cox head. Outer
validation does not select Probe/Cox hyperparameters or epoch counts. Saved
full-Probe train predictions are evaluation/cutoff diagnostics only and NEVER
Cox fitting covariates; save explicit cross-fitted Cox fitting table separately.

Target-wise Probe grid and Cox grid .01/.1/1/10/100/1000; seed123 inherited from
source config, no new bash seed. Probe tuning KFold3/shuffle, deterministic.
No Mask/Shape/DFS labels at final Probe inference; training masks supervise
the decoder. Thus comparison with direct512 is also a comparison of auxiliary
phenotype supervision/regularization, not proof of a defective DFS risk head.
True Shape is a mask-assisted reference, not a theoretical performance bound.

Output independent results/predicted_shape_increment and dedicated log;
crossfit roles/covariates/patient predictions stay ignored. Save all20 contexts'
Probe roles/choices/full models, training OOF Shape and held Shape; five Cox
heads, metrics/NLL/AUC/KM/HR; paired2000 patient Bootstrap vs all three references,
per-fold changes and mean/sampleSD. Record source hashes before/after, fail on
missing/mismatched caches, no neural fallback/extraction. Report limits of
conditioning on fitted models and extensive prior exploration.

- [x] Tests RED: perturb held/outer Shape labels without changing fit outputs;
      all Cox-training rows predicted once by disjoint Probe; reject malformed
      splits/overlap; reconstruct saved fitted Probe predictions.
- [x] Implement nested Probe cross-fitting and stage3 Cox evaluation.
- [x] Focused + complete test15 tests, real cached smoke and independent review.
- [x] Launch CPU-only public diagnostic with nohup/setsid; verify progress.
- [x] Verify all folds, source hashes, saved-model predictions, roles/choices,
      Cox fitting tables, paired intervals and report.

Pre-run verification: all93 tests passed in74.047s; the real384-train/192-held
nested Probe context passed in1.80s. All15 reference model/fold patient-label-
C-index checks passed. Independent review found no important/blocking issue.

Completed public-only stage3: PA+predictedShape Val0.7439+/-0.0349,
Δ versus PA+0.0049[-0.0049,+0.0161],4/5 positive; versus trueShape
-0.0081[-0.0167,+0.0007],1/5 positive; versus direct512+0.0102
[-0.0006,+0.0223],4/5 positive. These conditional intervals span zero.
Completion audit verified20 Probe contexts,65 fit roles,195 tuning partitions,
refit all45 OOF Probes to reconstruct predictions, refit all5 Cox heads from
their actual OOF fitting covariates, reproduced train/val metrics and paired
Bootstrap, and checked source hashes unchanged. No DFS encoder training.
