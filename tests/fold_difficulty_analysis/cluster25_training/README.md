# UNI cluster-balanced25

Only change versus completed Cluster128: dynamically resample25 instead of128 training patches per patient per epoch. Reuse eight per-WSI cluster labels and exact source/config checks read-only. Raw UNI remains unnormalized as model input. Same ABMIL/Cox, five folds, max30 epochs, patience10, Adam1e-4, WD5e-4, dropout.25, Cox32, seed123. Validation and final Train evaluation use full WSI. Outer Val selects best epoch, consistent with the prior epoch experiment; this is not Nested CV.

`train25.py` reuses the existing dataset cache/loader and main-flow training; no main-flow or Cluster128 files are modified. Read-only cache validation refuses to create missing labels. Each epoch uses independent patient/epoch seeded sampling; no DFS-based ranking or adaptive reclustering.

`report25.py` exports five-fold comparisons to Cluster128, Full UNI and historical EAGLE Top25. EAGLE uses Virchow2 rather than UNI, so that comparison is not a single-variable selection test.

All new outputs stay here. `checkpoints/`, `training.log`, patient prediction tables, cache and WSI visualizations are excluded from Git. After five folds and validation finish, publish code and aggregate results only; never publish intermediate progress.

Run from repository root with `bash tests/fold_difficulty_analysis/cluster25_training/run.sh`. Existing results/checkpoints are never overwritten.
