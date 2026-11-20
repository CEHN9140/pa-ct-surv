# Cox batch sensitivity, risk trajectories and late representation changes

Independent diagnostic; no modifications to main training or old experiments. Original721 patients/251 events, saved UNI fivefold CSV; all Raw UNI1024 patches, original ABMIL, no L2/sampling. Batch32/64/128, Adam1e-4/WD5e-4, dropout.25, max30 Epoch, early-stop10, seed123. Fifteen fresh fits, no project checkpoint warm-start.

This uses unchanged path_train.train_path including patient shuffle, dropout, Cox tail merging, online Train metrics, Val-based checkpoint selection and final survival analysis. Added diagnostics evaluate the same fixed model on full Train/Val every epoch. RNG state is preserved and verified against original training. At0,1,3,5,7,10,last,best save risks, per-patch attention, pooled512 and post256 representations locally. Only cohort-level summaries are published.

Three analyses:
1. Risk evolution: full Train/Val eval C/Loss/SD, checkpoint Spearman, affine-fit R²/slope, risk-scale ratio, pair flips, corrected/broken comparable pairs (including event/censor ties and prediction ties).
2. Batch sensitivity: same maximum patient-pass budget and selection rule; actual updates/patient exposures are reported. Batch128 may merge its tail into a ~192-patient last risk set. Exact shared update-count comparisons are separate. Equal Epoch is not equal optimizer-update budget; neither comparison isolates every exposure/optimization effect.
3. Late representations: linear CKA and centered relative drift of pooled512/post256, attention JS divergence, top128 overlap, entropy/effective patch count. Attention and feature drift are not direct evidence of biological correspondence or absence of usable information.

Reports: results/report.md, batch_summary.csv, fold_metrics.csv, learning_curves.csv, training_budgets.csv, matched_update_comparisons.csv, risk_pair_changes.csv, risk_phase_changes.csv, feature_attention_changes.csv, diagnostic_curves.png. Every reported saved-risk C-index is independently recomputed; batch32 selected-model results must reproduce the recent full-patch ABMIL baseline. Plot means±SD use epochs available for all5 folds only.

Logs: bs32/training.log, bs64/training.log, bs128/training.log. The `Epoch ...` line is original online Train/eval Val; `Eval Epoch ...` uses fixed eval Train/Val. No validation set is used for training, but ordinary Val-based early stopping remains; this is exploratory internal CV, not an untouched test or Nested CV.

Run `bash tests/fold_difficulty_analysis/cox_epoch_diagnostics/run.sh` after validation. The background pipeline runs three conditions concurrently, then verifies results, generates reports and performs the user-authorized Git commit/push of this experiment only. It aborts publication on source changes, unrelated staged files, branch divergence, incomplete fits or verification failure. All private patient vectors, source data and model weights stay local. Pipeline progress/publication status: validation/pipeline_status.json. No forced push, no unrelated commits.

Pre-launch verification:7 targeted tests passed, including unchanged original-training weights/RNG, full report fixtures and simulated whitelist-only Git publication. Real GPU16-patient/2-Epoch comparison reproduced weights, CPU/CUDA RNG and final metrics exactly. A193-patient real Full-WSI joint Cox backward passed with peak allocated GPU memory≈5878MiB. Three conditions can run concurrently on the48GiB GPU. These smoke results are not scientific CV performance.
