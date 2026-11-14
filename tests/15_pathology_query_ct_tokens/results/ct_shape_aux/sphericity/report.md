# CT-only auxiliary Shape supervision

Target=sphericity;721 patients/251 events,original outer5/inner3 roles. Public-only fresh initialization for every new fit.

| Model | Train C-index | Val C-index | AUC36 | AUC60 |
|---|---|---|---|---|
| A Cox baseline | 0.7436 ± 0.0135 | 0.7041 ± 0.0570 | 0.7250 ± 0.0522 | 0.7153 ± 0.0895 |
| Auxiliary selected by inner HELD | 0.7375 ± 0.0147 | 0.7098 ± 0.0489 | 0.7341 ± 0.0423 | 0.7236 ± 0.0843 |

ΔC vs A=+0.0057,95% paired interval[-0.0039,+0.0152], 3/5 positive folds.

Selected lambda / epochs by outer fold: 0:0.2/22, 1:0.2/29, 2:0.05/21, 3:0.2/27, 4:0.05/27.

## Linear recovery on outer validation

| Source | Target | R² | MAE |
|---|---|---|---|
| A | log_volume_mm3 | 0.7684 ± 0.0197 | 0.3894 ± 0.0160 |
| A | diameter_mm | 0.6866 ± 0.0353 | 6.1080 ± 0.2080 |
| A | sphericity | 0.2894 ± 0.0481 | 0.0626 ± 0.0056 |
| A | sphere_residual_given_true_VD | 0.0016 ± 0.0249 | 0.0456 ± 0.0047 |
| aux | log_volume_mm3 | 0.7579 ± 0.0741 | 0.3922 ± 0.0570 |
| aux | diameter_mm | 0.6934 ± 0.0301 | 6.0924 ± 0.5405 |
| aux | sphericity | 0.2795 ± 0.0458 | 0.0625 ± 0.0053 |
| aux | sphere_residual_given_true_VD | -0.0003 ± 0.0455 | 0.0451 ± 0.0037 |

## Boundaries

- STOP C-index selects each epoch; separate inner HELD C-index selects lambda0/.05/.2. Exact ties prefer smaller lambda. Outer epochs are floor median chosen-candidate STOP epochs; outer patients never tune anything.
- A/lambda0 reuses matched completed control outputs and representation caches; this is not auxiliary initialization or warm-start. Nonzero candidates and outer refits start only from public MedicalNet/new heads. All backbone, risk and auxiliary parameters train; BN running statistics remain fixed.
- Original risk head512->256->ReLU->Dropout.25->1 is preserved. Extra head512->K has no dropout and its construction preserves baseline RNG. True batch32+tail,optimizer/LRs,seed and inputs match A.
- Shape targets use only neural FIT population mean/std; outer refits use only outertrain statistics. These buffers are checkpointed. MSE averages patients and targets: at the same lambda each C target receives one third of the B per-target weight. B/C is a supervision-configuration comparison,not a causal isolation of two extra targets.
- A and selected lambda0 have no trained auxiliary head: direct auxiliary-head MSE/R²/MAE are N/A. Nonzero direct-head errors are inverse-transformed using checkpoint FIT scalers. Probes are separately fitted readouts for all groups.
- Residual target is S-g(logV,D),where unpenalized LINEAR g is fitted on the corresponding Probe training patients only. It is not complete conditional independence from VD; nonlinear size relationships can remain and CT features themselves were not residualized. Residual R² and reconstructed-S R² have different denominators and are reported separately in fold metrics.
- Oracle VD baseline/reconstruction uses held trueVD only for diagnostic labels/comparators,not deployed risk prediction. Risk inference uses CT alone on the existing mask-localized ROI64; training Shape targets come from masks.
- Lambda and Probe alpha reuse outer-training inner partitions for tuning; inner Probe errors are not independent validation. Each inner readout uses one appropriate encoder basis; the final readout uses its outer encoder. Outer Probe results measure recovery,not DFS success.
- Target choice follows earlier exploration on this same cohort; nested CV here does not erase historical selection bias. Paired Bootstrap2000 conditions on fitted models and excludes retraining/prior exploration uncertainty. Independent confirmation is still needed.
- No teacher/student or formal method change was launched. A better Shape R² alone is not success: the primary outcome is paired CT-only DFS improvement.
