# Shape-adjusted First-order and MedicalNet phenotype decoding

721 patients / 251 events / original five outer folds; cache-only diagnostics. No neural training or stage3 DFS fits.

| Prognosis model | Train C-index | Val C-index | AUC36 | AUC60 | Val Cox NLL |
|---|---|---|---|---|---|
| PA | 0.7637 ± 0.0441 | 0.7390 ± 0.0350 | 0.7744 ± 0.0279 | 0.7571 ± 0.0525 | 4.2528 ± 0.0776 |
| PA_Shape | 0.7737 ± 0.0328 | 0.7520 ± 0.0327 | 0.7933 ± 0.0176 | 0.7671 ± 0.0466 | 4.2325 ± 0.0629 |
| PA_Shape_FirstOrder | 0.7743 ± 0.0308 | 0.7527 ± 0.0387 | 0.7941 ± 0.0283 | 0.7657 ± 0.0552 | 4.2256 ± 0.0820 |

PA+Shape+FirstOrder minus PA+Shape: ΔC=+0.0007, 95% paired interval[-0.0063,+0.0079]; 2/5 positive folds.

PA+Shape+FirstOrder minus PA only: ΔC=+0.0137, 95% paired interval[+0.0019,+0.0268]; 4/5 positive folds.

## Linear probes

| Target | Units | Public R² | Finetuned R² | Public MAE | Finetuned MAE |
|---|---|---|---|---|---|
| log_volume_mm3 | ln(volume/mm3) | 0.7735 ± 0.0319 | 0.7684 ± 0.0197 | 0.3925 ± 0.0337 | 0.3894 ± 0.0160 |
| diameter_mm | mm | 0.7101 ± 0.0166 | 0.6866 ± 0.0353 | 5.9380 ± 0.2303 | 6.1080 ± 0.2080 |
| sphericity | dimensionless | 0.2955 ± 0.0971 | 0.2894 ± 0.0481 | 0.0627 ± 0.0053 | 0.0626 ± 0.0056 |
| tumor_mean | clipped-intensity units | 0.7918 ± 0.0278 | 0.8035 ± 0.0299 | 68.8813 ± 5.3576 | 65.6061 ± 4.2620 |
| tumor_std | clipped-intensity units | 0.5053 ± 0.0929 | 0.4886 ± 0.0416 | 33.8934 ± 2.1675 | 33.4023 ± 1.0825 |
| tumor_p10 | clipped-intensity units | 0.6758 ± 0.0614 | 0.6704 ± 0.0473 | 69.9527 ± 3.1316 | 68.3020 ± 3.3408 |
| tumor_median | clipped-intensity units | 0.7894 ± 0.0391 | 0.7961 ± 0.0317 | 92.0540 ± 7.7161 | 86.2376 ± 4.1050 |
| tumor_p90 | clipped-intensity units | 0.6028 ± 0.0349 | 0.6274 ± 0.0786 | 74.6642 ± 6.5481 | 68.3076 ± 9.2970 |
| tumor_entropy | bits | 0.6218 ± 0.0669 | 0.6134 ± 0.0581 | 0.2492 ± 0.0084 | 0.2466 ± 0.0133 |

## Protocol and interpretation

- Probe: train-only feature scaling, float64 linear Ridge with intercept, target-wise inner MSE tuning. Each inner fold uses its own DFS encoder; outer validation is never used for selection. No concatenation of features from different trained encoders into one regression basis.
- Probe scores use outer-validation patients only. Tables report five-fold means/sample SD; pooled outer R² is separately labeled. Baselines predict each outer training target mean. No survival outcome enters probe regression or its tuning; the finetuned encoders previously used DFS labels inside their appropriate training folds.
- Low linear R² does not prove absence of nonlinear encoding. High R² does not establish DFS benefit or prove a risk head failed. No threshold for good decoding is selected from these results.
- Primary prognosis contrast is PA+Shape+FirstOrder versus PA+Shape. Re-selected ridge applies to all CT terms; this evaluates a feature bundle, not a causal or independently identified coefficient effect.
- All phenotype labels require masks. Probe inference uses only cached CT features. Density units are recovered clipped-input proxies, not verified raw HU. Shape volume is log-transformed, matching the earlier Cox reference.
- Multiple analyses remain exploratory. Patient-paired Bootstrap conditions on fitted models and excludes retraining, tuning and earlier exploration uncertainty.
- Stage3 (cross-fitted predicted phenotypes -> DFS Cox) is intentionally not run. The formal method remains a jointly trained pathology–CT teacher -> CT-only student; these cached baseline representations are diagnostics only.
