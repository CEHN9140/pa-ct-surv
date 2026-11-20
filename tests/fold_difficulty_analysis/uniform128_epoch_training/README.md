# Uniform-random128 Raw UNI epoch control

Approved design: change only cluster-balanced selection to uniform no-replacement128 from the complete bag, independently per patient/epoch. N≤128 retains all. Private SHA-derived seed keeps sampler RNG separate from shuffle/dropout. Raw UNI1024, original fivefold CSV, ABMIL, Cox32 with merged tail, dropout.25, Adam1e-4/WD5e-4, max30 epochs, patience10, seed123; Full WSI validation and final evaluation. Reuse unmodified path_train.main/train_path/evaluate_survival, fresh initialization each fold. No cluster input or model-input L2.

Implementation plan: tests first; experiment-local dataset adapter; verify shared source hashes/configs against completed cluster128 run; CPU tests and real GPU two-epoch smoke; background fivefold training. All outputs here. No changes to existing experiments or main code, no automatic push.

Completed all five folds. Results are in `results/metrics.csv`, KM plots in `results/fold_*/km_high_vs_low.png`, and epoch metrics in `training.log`. Train C-index at the selected checkpoint:0.8347±0.0411; Val C-index:0.7485±0.0399. Checkpoint metadata includes best/last epochs; weights and patient-level predictions remain local. Only this experiment is published.
