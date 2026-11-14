# Fixed-weight Shape CT risk conditional increment (scheme A)

721 patients, 251 events; original outer5/inner3 roles. Fixed ALL-Shape lambda=0.2; fixed CT ridge=1, PA slope unpenalized. Both risk slopes are free.

| Model | Train C-index | Val C-index | AUC36 | AUC60 | Val Cox NLL |
|---|---|---|---|---|---|
| M0 | 0.7637 ± 0.0441 | 0.7390 ± 0.0350 | 0.7744 ± 0.0279 | 0.7571 ± 0.0525 | 4.2528 ± 0.0776 |
| M1 | 0.7780 ± 0.0377 | 0.7433 ± 0.0337 | 0.7776 ± 0.0222 | 0.7633 ± 0.0588 | 4.2447 ± 0.0808 |
| M2 | 0.7769 ± 0.0378 | 0.7501 ± 0.0360 | 0.7864 ± 0.0242 | 0.7710 ± 0.0578 | 4.2233 ± 0.0789 |

| Contrast | Mean ΔC-index | Conditional 95% interval | Positive folds |
|---|---|---|---|
| M1_minus_M0 | +0.0043 | [-0.0046, +0.0139] | 3/5 |
| M2_minus_M0 | +0.0111 | [+0.0014, +0.0209] | 4/5 |
| M2_minus_M1 | +0.0068 | [+0.0013, +0.0120] | 5/5 |

Fixed-lambda C CT-only Val C-index: 0.7112 ± 0.0451. This differs from the historical automatic-lambda strategy (0.7136).

## Scope and safeguards

- M0/M1/M2 refit the same Cox procedure on unchanged PA OOF risks. M1 uses source A CT OOF; M2 uses fixed-weight C CT OOF. No head hyperparameter is selected.
- Each C HELD prediction comes from its matching FIT/STOP checkpoint. Risk scalers use FIT eval predictions only; outer scalers use outer-training eval predictions only. Cox scalers use OOF covariates only.
- Reused checkpoints are inference/reference only. New outer CT fits start from PUBLIC MedicalNet/new heads, never a project-trained unimodal checkpoint. No PA, teacher or student is trained.
- Outer epochs are the median of fixed-candidate STOP epochs, not the previous selected-lambda epochs. HELD scores and outer outcomes cannot select lambda or outer epochs.
- Final Train metrics are in-sample outer-refit eval predictions; they are not independent validation. Fusion training covariates and predictions are saved separately.
- This is exploratory fixed-setting diagnosis after previous same-cohort exploration, not evaluation of automatic lambda selection, independent confirmation, or a causal Shape effect.
- A null scalar-risk increment does not prove the 512-dimensional representation has no complementary information. A positive result does not guarantee joint-teacher/student improvement.
- Cohort source pathology refits still use historical epochs [1,14,1,14,23]; two short fits remain a baseline stability limitation. Time/event and IPCW/KM conventions are unchanged.
- Shape targets require masks during auxiliary training. Risk inference uses CT only on the existing mask-localized ROI64; upstream CT acquisition/intensity processing remains unknown.
