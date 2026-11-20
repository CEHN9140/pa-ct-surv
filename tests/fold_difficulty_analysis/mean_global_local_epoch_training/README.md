# Mean global + low-capacity DFS local: independent Epoch control

User-approved design: both branches see all Raw UNI1024 patches. Global branch directly averages raw vectors (1024). Local branch uses Linear1024→128/ReLU/Dropout.25; attention128→32/Tanh/Dropout.25→1, softmax over patches; weighted pool128→64/ReLU/Dropout.25. Concatenate1024+64=1088, then Linear1088→64/ReLU/Dropout.25→1 scalar risk. Total213378 trainable parameters (~70.5% fewer than full ABMIL722178).

One joint Cox loss trains both the local attention representation and the fusion risk head. No auxiliary losses, residual risk sum, Patch Top-K, feature normalization or pre-trained downstream model weights. Attention is DFS-supervised through joint Cox gradients, not a verified label of DFS-relevant tissue.

Reuse unmodified `path_train.py`: original UNI CSV five folds; Adam1e-4/WD5e-4, dropout.25, Cox batch32 with tail merging, 30 Epoch, early-stop10, seed123. Outer validation selects the best epoch, matching the recent cluster/uniform controls. Full WSI training and evaluation. Per-epoch metrics use main-flow online Train and eval Val; final metrics use best checkpoint with full eval Train/Val.

Only the pathology model changes relative to a Full-WSI main-flow control. Comparison to random128/cluster128 also changes how many patches the local trainable branch sees and cannot isolate capacity alone. No claims of solved overfitting or complementary local information before results and ablations.

Plan: model shape/gradient/invariance tests; real GPU main-flow training/checkpoint/full-evaluation smoke; parameter/source/config verification; background fivefold training. No modifications to old experiments or main source. No automatic push.

Run: `bash tests/fold_difficulty_analysis/mean_global_local_epoch_training/run.sh`.
Outputs: `training.log`, `status.json`, `results/`, `checkpoints/`, `validation/`, all within this experiment. `results/run_config.yaml` records actual custom architecture even though the main entry's parser option remains abmil. Results include fold metrics, predictions and KM plots using the existing main-flow evaluator.

## Completed run

All five folds completed. Selected-checkpoint Full WSI Train C-index:0.7840±0.0659; Val C-index:0.7361±0.0465. See `results/metrics.csv` for per-fold metrics and AUC, `training.log` for complete epoch records, and `checkpoints/fold_*/checkpoint_metadata.yaml` for selected epochs. Published files exclude weights, patient-level predictions and smoke-test artifacts.
