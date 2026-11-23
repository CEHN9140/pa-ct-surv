# Freeze-C prespecified training-seed replication

Fixed original Outer fivefolds and fixed inner event-stratified80/20 split seed123. New training seeds7/42/2026; same Raw UNI, ABMIL, Cox32, Dropout.25, lr1e-4, WD5e-4,3Epoch common fork and0..20continuation selection. Only baseline and freeze_c. All warmups start newly; no project baseline checkpoint initialization. Per-seed audits reuse the original tensor/RNG/order/prediction checks, with exactly two arms.

| Training seed | Baseline Val C | Freeze-C Val C | Paired mean delta | Positive folds |
|---|---:|---:|---:|---:|
| 7 | 0.7383 | 0.7353 | -0.0029 | 1/5 |
| 42 | 0.7319 | 0.7387 | +0.0067 | 5/5 |
| 2026 | 0.7322 | 0.7361 | +0.0039 | 3/5 |

Seed123 is retained as the previously explored reference; new seeds are reported separately. Repeated seeds/folds are not new independent patients, so no15-fold independent-sample significance or pooled raw-risk C-index is claimed. Conditional per-seed bootstrap does not cover retraining, selection or prior exploration. No freeze-epoch/lambda/architecture search is performed. A positive mean alone does not establish reproducible benefit or identify head C as a unique overfitting cause.
