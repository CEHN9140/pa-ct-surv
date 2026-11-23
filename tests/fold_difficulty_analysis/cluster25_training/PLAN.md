# Cluster-balanced25 comparison

Use the completed cluster128 experiment as the matched protocol: same Raw UNI features, per-WSI eight-cluster cached labels, fixed five folds, seed123, ABMIL/Cox, Adam1e-4 WD5e-4, dropout.25, Cox32, max30 epochs/patience10. Change only training patch budget128→25; sample anew each epoch. Evaluate full WSI, including final Train evaluation. No DFS-dependent cluster update or learned patch selection. Reuse validated dataset/loader/training helpers through an experiment-local adapter; never edit main or old experiment.

1. Test deterministic epoch-changing balanced25 sampling, small bags, complete inference and main-flow training/checkpoint smoke.
2. Implement adapter, read-only cache guard and exact control config/source checks.
3. Train five folds; summarize versus Cluster128, Full UNI and EAGLE Top25 (EAGLE comparisons change encoder/selection and are descriptive).
4. Check original cohort alignment, aggregate only folds0–4, checkpoint/evaluation completeness, source stability. Keep all outputs inside new test. Push only after completion and validation, excluding predictions/cache/weights/logs.
