# Fresh Full-WSI MeanPool and original ABMIL controls

Two fresh fivefold runs directly use unchanged `path_train.py`. Both train/validate on ALL Raw UNI1024 patches, no L2, sampling or clustering. Original `data/trident/features_uni_v1/all_label_roi64.csv` folds.

Same settings: Cox32 merged tail, Adam1e-4/WD5e-4, dropout.25,30 epochs,patience10,seed123 (main default),8 workers. Fresh initialization per fold. Val selects best epoch, same protocol as preceding Epoch experiments, not Nested CV. The model is the only difference between these two runs.

MeanPool: raw mean1024→512→256→1. Original ABMIL: patch1024→512, attention512→128→1, pool512→256→1. Existing projection/attention/head dropout settings remain unchanged.

Launch separately with run_meanpool.sh and run_abmil.sh; parallel background runs. Each writes its own training.log, results/config/metrics/predictions/KM, and checkpoints. No edits to main code or prior experiments; no automatic push. Logs end with fivefold mean±SD summary.

Validate shell syntax, source/CSV/config parity against preceding controls, and both models on real GPU patients with Cox backward and checkpoint reload; then verify actual Epoch output. Validation artifacts stay under validation/. Patient-level predictions, weights and plotting caches remain local.

## Completed runs

Both fivefold runs completed. MeanPool Val C-index:0.7279±0.0457; original ABMIL:0.7458±0.0405. See each results/metrics.csv for complete Train/Val, AUC and per-fold metrics. Uploaded artifacts include launchers, configuration, training logs, KM plots, aggregate metrics and checkpoint epoch metadata. Model weights and patient-level tables remain local.
