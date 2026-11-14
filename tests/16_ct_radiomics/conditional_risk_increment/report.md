# Full Radiomics risk conditional increment

Same721 patients/251 events, outer5 and inner3 roles as test15. PA/A/C risks remain fixed. M0=PA; M1=PA+original CT; M2=PA+ALL-Shape CT(lambda0.2); M3=PA+Original107 Radiomics Risk. All heads use fixed CT ridge1, no PA penalty and free slopes.

| Model | Train C-index | Val C-index | AUC36 | AUC60 | Val Cox NLL |
|---|---|---|---|---|---|
| M0 | 0.7637 ± 0.0441 | 0.7390 ± 0.0350 | 0.7744 ± 0.0279 | 0.7571 ± 0.0525 | 4.2528 ± 0.0776 |
| M1 | 0.7780 ± 0.0377 | 0.7433 ± 0.0337 | 0.7776 ± 0.0222 | 0.7633 ± 0.0588 | 4.2447 ± 0.0808 |
| M2 | 0.7769 ± 0.0378 | 0.7501 ± 0.0360 | 0.7864 ± 0.0242 | 0.7710 ± 0.0578 | 4.2233 ± 0.0789 |
| M3 | 0.7720 ± 0.0331 | 0.7513 ± 0.0420 | 0.7918 ± 0.0290 | 0.7661 ± 0.0570 | 4.2346 ± 0.0848 |

| Contrast | Delta C-index | Conditional95% interval | Positive folds | Corrected/broken |
|---|---|---|---|---|
| M1-M0 | +0.0043 | [-0.0046, +0.0139] | 3/5 | 804/692 |
| M2-M0 | +0.0111 | [+0.0014, +0.0209] | 4/5 | 942/654 |
| M3-M0 | +0.0123 | [+0.0015, +0.0234] | 4/5 | 1139/815 |
| M3-M1 | +0.0080 | [-0.0002, +0.0161] | 4/5 | 791/579 |
| M3-M2 | +0.0012 | [-0.0067, +0.0089] | 2/5 | 662/626 |

Radiomics OOF base fitting and all CV tuning exclude HELD and outer validation. Inner base models fit FIT+STOP; each filter/scaler is refitted within each tuning fold. Risk normalization uses base fitting predictions; head scalers use only OOF covariates. Outer fitting/penalty selection uses outer training only. M0/M1/M2 reproduce prior exports.

Bootstrap2000, fixed folds/patient resampling, conditional on saved models; excludes retraining, earlier selection and multiple-comparison uncertainty. Patient pairs are not independent.

The strict PA baseline has historical outer epochs1/14/1/14/23; two short refits remain a limitation. Historical PA0.7528 used outer stopping and is not a matched benchmark. A small scalar-risk increment does not exclude complementary high-dimensional information. Radiomics uses complete tumor masks/current NIfTI intensities, not only ROI64. This is diagnostic reuse of predictions, not warm-start/frozen unimodal teacher training. No neural training, feature-group search or Brier-score analysis was performed.
