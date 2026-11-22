# Fold-local UNI classes and Best-to-Late attribution stability

Completed5 Train-only dictionaries, 3605fold-patient observations, 5768outcome-independent native QC patches and 721OOF WSI maps. No ABMIL training, no feature suppression or formal model change.

## Predefined geometry and fit roles

Raw UNI1024 Euclidean MiniBatchKMeans,20classes per fold, up to128fixed patches per Train patient with equal total patient weight, seed123, batch2048, max_iter100, n_init3. No PCA, L2, Zscore, DFS labels or Val data enter dictionary fit. Fold-local labels are nominal, not tissue types. Sampled fit points are not all WSI patches; all Train/Val patches are assigned afterward without centroid updates.

## What the signed margin metric means

For a changed comparable pair, patch IG from the event patient contributes+c to risk margin and comparator IG contributes-c. Credit-loss/credit-gain context keeps support and opposition. The normalized net metric sums signed class margin changes on all credit-changing pairs, weighted by absolute credit change (ties0.5), divides by the split comparable-pair count and the same full-Train reference risk SD. It is not delta C-index, a unique C-index allocation, or proof of a causal/biological relation. A positive patient risk shift alone is never treated as helpful. One class can support both harm and correction in different pair contexts.

| Fold | Local classes with Train positive / Val negative margin | Classes with conditional simultaneous Val upper below0 |
|---|---:|---:|
| 0 | 10/20 | 0/20 |
| 1 | 12/20 | 0/20 |
| 2 | 13/20 | 0/20 |
| 3 | 16/20 | 0/20 |
| 4 | 9/20 | 0/20 |

Val bands use1000paired patient resamples per fold, recomputing comparable pairs/credit transitions with fixed risks/IG/dictionary. The max-standardized band approximately covers20class contrasts within a fold; it does not cover fivefold exploration, dictionary refits, model retraining, best-epoch selection or prior repeated analysis. Best epochs were selected on Outer Val; a negative best-conditioned pattern is not independent validation. Train patients recur across four models; no significance from pooled Train rows is claimed.

## Prevalence, attention and geometry checks

fold_cluster_summary.csv separates class prevalence, best/late attention mass, delta mass, signed IG, normalized contextual helpful/harmful margins, UNI norm, assignment distance and coarse all-patch thumbnail white fraction. Attention masses are compositional (sum to1), so class changes are dependent. class_margin_concentration.csv indicates whether contextual allocations are diffuse or concentrated; an allocation share is not causal importance. Changes can reflect class prevalence/domain differences and the projection/head-dependent attribution path, not just unstable morphology. Per-class means of norm/white/distance condition on presence; absent classes have zero mass and undefined technical mean.

## Crossfold labels

descriptive_centroid_correspondence.csv provides post-hoc Raw Euclidean Hungarian matches to fold0 plus mutual-nearest and distance/radius diagnostics. It never changes class assignment or model predictions and does not establish identical biology. It is not used to pool class outcomes or declare that the same numeric class repeats across folds. Forced one-to-one matches can be weak and dictionaries can split/merge phenotypes.

## Technical proxies and visualizations

Eight deterministic random patches per patient are selected without consulting outcomes, IG or class, read once at the native TRIDENT FOV, then assigned to each fold dictionary. random_qc_by_class.csv reports sparse patient-balanced conditional technical summaries; sample counts and missingness matter. Laplacian/OD/white proxies depend on local tissue structure and cannot label blur, stain artifact or tissue. OOF maps show local class colors on original thumbnails. Prototype panels show up to four nearest sampled Train patches from distinct patients per class; prototype_coverage.csv records rare-class shortfalls, not DFS-selected regions or histology diagnoses. All maps/native images/IDs/centers stay private.

## Decision boundary

Train-positive/Val-negative classes are exploratory candidates for inspection, not a license to delete or downweight those patches and report improvement on the same Val set. No model performance gain is established. If most classes share the same sign contrast or allocations are diffuse, the result does not isolate one unstable phenotype. If a candidate appears, validate a predefined intervention with independent selection/evaluation; if no repeatable pattern is supported, stop this limited attention diagnostic rather than search K or tune cluster geometry on Val. Without reliable tissue labels this does not identify necrosis/stroma/tumor morphology, and no doctor is required to execute these computations.

Checks: Train fit roles, deterministic sampled indices/weights, immutable saved centroids, fresh Train/Val assignment samples, all class patch/mass/IG/context sums, both endpoint margin identity and independent QC sample assignment were verified.
