# Raw UNI cluster-balanced128: completed epoch experiment

Five folds completed. Original `data/trident/features_uni_v1/all_label_roi64.csv` assignments retained. Validation patient IDs, times and events were checked against the historical Full-UNI reference and matched exactly.

Protocol: ordinary five-fold CV; outer validation C-index selects the best epoch. No Nested CV in this completed run. ABMIL, Raw UNI1024, dropout0.25, Cox batch32 with tail merging, Adam lr1e-4/WD5e-4, max30 epochs, patience10, seed123.

Training samples128 patches per patient per epoch with balanced allocation across local unsupervised clusters. Validation and final evaluation use Full WSI. L2 is used only on a copy for clustering, never on ABMIL inputs.

| Method | Train C-index | Val C-index | AUC36m | AUC60m | Train-Val gap |
|---|---:|---:|---:|---:|---:|
| Raw UNI, Full patches | 0.8368 ± 0.0550 | 0.7458 ± 0.0405 | 0.7846 ± 0.0162 | 0.7613 ± 0.0357 | 0.0910 ± 0.0914 |
| Raw UNI, cluster-balanced128 | 0.8280 ± 0.0413 | 0.7461 ± 0.0443 | 0.7866 ± 0.0162 | 0.7618 ± 0.0464 | 0.0818 ± 0.0817 |

Train metrics above are from the selected checkpoint in eval mode on Full WSI. Epoch log training metrics are the main-flow online metrics.

| Fold | Full UNI | Cluster128 | Delta |
|---|---:|---:|---:|
| 0 | 0.6902 | 0.6945 | +0.0043 |
| 1 | 0.7559 | 0.7567 | +0.0008 |
| 2 | 0.7908 | 0.7898 | -0.0010 |
| 3 | 0.7719 | 0.7846 | +0.0127 |
| 4 | 0.7202 | 0.7050 | -0.0152 |

Mean delta C-index: +0.0003; 3/5 folds improved. The training-validation gap declined slightly, but validation discrimination did not materially improve. No claim of statistical significance is made. This historical comparison is not a freshly retrained paired Full-WSI control.

Best epochs for cluster128:15,13,5,7,10. Later training still showed overfitting in several folds. This experiment supports reducing training patch count while roughly retaining prediction performance; it does not establish DFS-specific patch selection or resolution of the generalization problem.

Published: experiment-local epoch runner/tests, clustering/sampling/visualization code, log, configuration, aggregate metrics and five KM figures. Patient-level tables, raw features, cluster caches, WSI panels, checkpoints, stopped Nested outputs and unrelated project changes are not included.
