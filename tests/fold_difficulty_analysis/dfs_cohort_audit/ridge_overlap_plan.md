# Matched Stage5 Ridge / Stage6 ABMIL error overlap

Goal: frozen predictions only; no training/feature extraction/model loading or label/split changes. Compare strict nested models, never historical Outer-Val-best models.

Files: ridge_overlap.py, test_ridge_overlap.py; aggregate ridge_overlap_results/; sensitive case/pair lists private/ridge_overlap/.

- [x] Validate 20 Train/Val prediction files (5 folds × 2 splits × 2 models), membership, labels, float32 times, finite risks. Replay 20 saved C-indices and five Stage6 risk correlations/deltas.
- [x] Enumerate identical Harrell pairs and all nine correct/tie/wrong transitions. Prove credit-difference reconstruction. Describe EE/EC and EE time gaps (0,6], (6,12], (12,24], >24 months without pooling raw risks.
- [x] Produce fold summaries and pair-weighted descriptive totals. Use Outer Train risk SD per model for margin units. Predefine large-gap EE as >24 months, high-confidence wrong as margin≤-1 training SD; retain private lists only, no case exclusion.
- [x] Add useful conditional overlap rates (shared/any wrong, shared/Ridge wrong, shared/ABMIL wrong). No patient-pair independence claims, significance tests or outcome-driven threshold selection.
- [x] Run meaningful tie/direction/C-index tests, all input replays and read-only review. Report common error is not proof of absent UNI information, label error or irreducible clinical risk. No automatic new encoder/loss/model experiments.
