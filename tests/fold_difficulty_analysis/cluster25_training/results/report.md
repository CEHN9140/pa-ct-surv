# Raw UNI cluster-balanced25

Completed five folds. Same protocol as Cluster128; only training patch budget changed128→25. Eight fixed per-WSI clusters, newly sampled each epoch; Raw UNI inputs. Full WSI validation and final Train evaluation. Adam1e-4/WD5e-4, dropout.25, Cox32, max30 epochs, patience10, seed123. Outer Val selects best epoch; exploratory internal CV, not Nested CV.

|Method|Train C|Val C|AUC36|AUC60|
|---|---:|---:|---:|---:|
|Cluster25|0.8281 ± 0.0524|0.7465 ± 0.0397|0.7877 ± 0.0093|0.7660 ± 0.0448|
|Cluster128|0.8280 ± 0.0413|0.7461 ± 0.0443|0.7866 ± 0.0162|0.7618 ± 0.0464|
|Full UNI|0.8368 ± 0.0550|0.7458 ± 0.0405|0.7846 ± 0.0162|0.7613 ± 0.0357|
|EAGLE Top25 lr=1e-4|0.8497 ± 0.0669|0.7576 ± 0.0383|0.8015 ± 0.0199|0.7827 ± 0.0428|
|EAGLE Top25 lr=1e-3|0.8402 ± 0.0575|0.7585 ± 0.0407|0.8045 ± 0.0214|0.7906 ± 0.0438|

EAGLE uses CHIEF selection and Virchow2 CLS features; its comparison changes encoder and selection and cannot isolate patch sampling. Cluster25 vs Cluster128 is the intended sampling-budget contrast. These historical controls were not freshly retrained. No claim that sampled patches are DFS-specific. All predictions, checkpoints and logs remain local.
