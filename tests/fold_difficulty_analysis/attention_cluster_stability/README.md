# Train-only UNI classes and Best-to-Late attribution stability

One limited, no-ABMIL-training diagnostic. Original721 patients/251 events/saved five folds; Raw UNI1024 full WSI and existing Best-to-Late exact IG. Main flow and old experiments remain unchanged. No automated tissue labels, doctor dependency, patch removal, model initialization, new loss or automatic push.

## Dictionary geometry and patient roles

Each fold fits20MiniBatchKMeans classes on Outer Train only. Raw Euclidean UNI, no L2/PCA/Zscore. Per Train patient select up to128deterministic feature rows, with equal total sample weight per patient; this is dictionary fitting only, not MIL patch truncation. Seed123, batch2048, max_iter100, n_init3, max_no_improvement20. K/geometry are fixed before Val analysis. DFS labels/time are not used to fit dictionaries. Assign all Train/Val patches without updating centers; audit/source hashes and sampled fit indices are saved privately.

## Margin interpretation

Per patient-class: patch count/fraction, best/late Attention mass, delta mass, signed/absolute IG and both harm/repair support/opposition from comparable-pair context. The normalized net measure equals signed class margin changes on credit-changing pairs weighted by absolute credit change, divided by total comparable-pair count and the same full-Train reference risk SD for both roles. Event endpoint has+IG; comparator endpoint has-IG. A positive risk shift is not automatically helpful. Same patch/class can support both corrected and broken pairs. This is not delta C-index, unique credit attribution or a causal biological effect.

Val bootstrap recomputes comparable pairs/credit transitions from paired resampled patients and fixed component risks; 1000repeats yield pointwise intervals and approximate20class within-fold max-standardized bands. They do not cover dictionary/model refits, best-epoch selection, earlier exploratory decisions or all fivefold comparisons. Best epochs used Outer Val C-index. Train patients recur in4folds; no pooled Train significance is claimed. High/low sign patterns are exploratory, not proof of feature exhaustion or a license to tune/suppress on the same Val set.

Classes are fold-local nominal IDs, not histological phenotypes. Post-hoc Raw Euclidean Hungarian matches to fold0 include mutual-nearest and distance/radius diagnostics. They do not alter fitting/assignments or pool class outcomes; forced matches do not establish identical biology and dictionaries can split/merge feature populations. Norm, prevalence, assignment distance and attention allocation are reported separately. Raw UNI classes may reflect scale, stain/scanner or density differences rather than biological morphology.

## Native QC and private visualizations

Eight random native patches per patient, chosen independently of outcome/IG/classes, are read once from original SVS at metadata-matched TRIDENT FOV. Same pixels are assigned to every fold dictionary. QC summaries are patient-balanced conditional means by fold/role/class, with native-patch/valid-patient counts and explicit zero coverage. Laplacian/OD/white/saturation are proxies, not blur or tissue diagnoses. All-patch thumbnails provide only coarse white fraction.

private/wsi_maps contains721OOF color overlays, one per patient using its held-out dictionary; class colors are not matched across folds. private/prototype_images has up to4nearest sampled Train patches from distinct patients per local class, no DFS selection. prototype_coverage.csv records rare-class shortfalls; empty panel slots are not images. All centers, IDs, patch coordinates, arrays and native images stay private.

## Execution and outputs

run.sh starts a CPU background pipeline: analyze.py→visualize_qc.py→report.py; no GPU/network training is required. status.json and validation/pipeline.log track progress. Real-data smoke and unit tests gate launch; existing outputs are not overwritten and source hashes are checked per stage.

results/: fold_cluster_summary.csv; cluster_contrasts.csv; val_conditional_bands.csv; fold_pattern_counts.csv; class_margin_concentration.csv; descriptive_centroid_correspondence.csv; random_qc_by_class.csv; prototype_coverage.csv; train_val_cluster_margins.png; report.md; validation.json. Patient-class rows/predictions remain private.

If contributions are diffuse or the pattern lacks crossfold support, stop this limited attention diagnostic rather than tune K on Val. Candidate classes still require independently selected/evaluated interventions before claiming improved DFS C-index. No tissue labels or predictive gains are established by the computations alone.
