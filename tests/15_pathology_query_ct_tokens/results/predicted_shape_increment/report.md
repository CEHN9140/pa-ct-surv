# Cross-fitted public MedicalNet Shape increment

721 patients,251 DFS events,original five outer folds. Public-pretrained CT features only; no neural fits/extraction.

| Model | Train C-index | Val C-index | AUC36 | AUC60 | Val Cox NLL |
|---|---|---|---|---|---|
| PA | 0.7637 ± 0.0441 | 0.7390 ± 0.0350 | 0.7744 ± 0.0279 | 0.7571 ± 0.0525 | 4.2528 ± 0.0776 |
| true_shape | 0.7737 ± 0.0328 | 0.7520 ± 0.0327 | 0.7933 ± 0.0176 | 0.7671 ± 0.0466 | 4.2325 ± 0.0629 |
| direct512 | 0.7768 ± 0.0362 | 0.7336 ± 0.0246 | 0.7642 ± 0.0240 | 0.7495 ± 0.0372 | 4.2658 ± 0.0505 |
| predicted_shape | 0.7685 ± 0.0384 | 0.7439 ± 0.0349 | 0.7833 ± 0.0232 | 0.7611 ± 0.0579 | 4.2420 ± 0.0704 |

Predicted Shape minus PA: ΔC=+0.0049, 95% paired interval[-0.0049,+0.0161], 4/5 positive folds.

Predicted Shape minus true_shape: ΔC=-0.0081, 95% paired interval[-0.0167,+0.0007], 1/5 positive folds.

Predicted Shape minus direct512: ΔC=+0.0102, 95% paired interval[-0.0006,+0.0223], 4/5 positive folds.

## Actual outer Shape decoder performance

| Target | R² | MAE |
|---|---|---|
| log_volume_mm3 | 0.7783 ± 0.0253 | 0.3857 ± 0.0292 |
| diameter_mm | 0.7101 ± 0.0166 | 5.9380 ± 0.2303 |
| sphericity | 0.3094 ± 0.0635 | 0.0625 ± 0.0062 |

## Validation boundaries

- Cox fitting uses honest PA OOF risks and nested OOF predicted Shape. Each OOF patient is excluded from Probe fitting AND alpha tuning. Inner Cox validation is excluded from its entire upstream Probe fitting/tuning. Outer validation never tunes a decoder or Cox head.
- Public CT vectors are fixed across all patients and have never used cohort DFS labels. Probe tuning is target-wise three-fold MSE, nested within each fitting subset. Saved PA sub-OOF partitions produce inner-training Shape OOF; source three inner partitions produce outer-training Shape OOF.
- Cox ridge is selected using the unchanged three inner PA calibration contexts. Scalars/features are standardized only inside each fitting partition. Final validation inference uses CT vectors plus pathology risk; no validation Mask/Shape is read by Probe/Cox prediction.
- The existing CT vectors were extracted from tumor-mask-localized ROI64. No extra Shape vector is needed at Probe inference, but this diagnostic does not remove the upstream localization/mask requirement.
- Train predictions use the final fitted Probe and saved final PA predictor only for descriptive training metrics/KM cutoffs; they are NOT the Cox fitting inputs. Explicit OOF fitting covariates and all nested roles are saved locally for audit.
- True Shape is a mask-assisted reference, not a theoretical upper bound. Direct512 uses the existing nested PCA16/32+ridge pipeline, not every possible512-D readout. Predicted Shape additionally benefits from training-mask phenotype supervision; a difference is not proof of a defective original head.
- Shape volume is log(volume/mm3), diameter is in mm, sphericity dimensionless. Probe MAE follows those units. Phenotype recovery and DFS gain remain distinct questions. Actual stage3 Probe R² is reported because its strict nested tuning can differ from the earlier standalone Probe.
- Mean/sample SD are over five folds. Patient-paired Bootstrap2000 conditions on fitted models and excludes retraining/tuning and prior exploration uncertainty. Multiple repeated analyses are exploratory.
- DFS-finetuned stage3 is not run: it requires further inner CT encoders to avoid inherited DFS-label leakage. No trained baseline initializes a teacher/student; the formal joint teacher -> CT student method is unchanged.
