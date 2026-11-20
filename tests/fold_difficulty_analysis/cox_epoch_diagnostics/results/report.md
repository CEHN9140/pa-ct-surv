# Cox batch, risk trajectory and representation diagnostics

Completed: Raw UNI/full patches, original ABMIL, five saved folds, batch32/64/128. Max30 Epoch, patience10, Adam1e-4/WD5e-4, dropout.25, seed123. Added evaluation preserves RNG; batch32 selected-model Train/Val C-index reproduces the original main-flow control.

The primary budget is a maximum of30 patient passes, not equal optimizer updates. Early stopping changes actual exposure. Matched-update comparisons are reported separately at exact common completed-epoch update counts; those comparisons have different exposure. No isolated causal effect of batch size is claimed.

| Cox batch | Train C-index | Val C-index | AUC36m | AUC60m |
|---|---:|---:|---:|---:|
| 32 | 0.8368 ± 0.0550 | 0.7458 ± 0.0405 | 0.7846 ± 0.0162 | 0.7613 ± 0.0357 |
| 64 | 0.8269 ± 0.0503 | 0.7450 ± 0.0384 | 0.7861 ± 0.0132 | 0.7669 ± 0.0333 |
| 128 | 0.7970 ± 0.0603 | 0.7444 ± 0.0375 | 0.7909 ± 0.0174 | 0.7662 ± 0.0337 |

## Early/late risk changes

| Batch | Phase | Split | Spearman | Risk SD ratio | Affine R² | Delta C |
|---|---|---|---:|---:|---:|---:|
| 32 | 10_to_last | train | 0.9090 | 1.822 | 0.8167 | +0.0787 |
| 32 | 10_to_last | val | 0.9362 | 1.786 | 0.8805 | -0.0200 |
| 32 | 1_to_5 | train | 0.9215 | 3.331 | 0.8160 | +0.0565 |
| 32 | 1_to_5 | val | 0.9238 | 3.316 | 0.8271 | +0.0172 |
| 32 | 1_to_last | train | 0.6893 | 7.913 | 0.4476 | +0.1950 |
| 32 | 1_to_last | val | 0.7590 | 7.734 | 0.5588 | -0.0029 |
| 32 | 5_to_10 | train | 0.9432 | 1.294 | 0.8835 | +0.0598 |
| 32 | 5_to_10 | val | 0.9519 | 1.283 | 0.9099 | -0.0001 |
| 64 | 10_to_last | train | 0.8776 | 1.792 | 0.7565 | +0.0963 |
| 64 | 10_to_last | val | 0.9086 | 1.730 | 0.8236 | -0.0237 |
| 64 | 1_to_5 | train | 0.9433 | 4.737 | 0.8659 | +0.0363 |
| 64 | 1_to_5 | val | 0.9417 | 4.724 | 0.8727 | +0.0172 |
| 64 | 1_to_last | train | 0.7005 | 10.874 | 0.4569 | +0.1813 |
| 64 | 1_to_last | val | 0.7462 | 10.388 | 0.5372 | -0.0023 |
| 64 | 5_to_10 | train | 0.9549 | 1.282 | 0.9095 | +0.0487 |
| 64 | 5_to_10 | val | 0.9570 | 1.280 | 0.9184 | +0.0042 |
| 128 | 10_to_last | train | 0.9058 | 1.536 | 0.8165 | +0.0731 |
| 128 | 10_to_last | val | 0.9202 | 1.534 | 0.8496 | +0.0016 |
| 128 | 1_to_5 | train | 0.9308 | 5.694 | 0.8784 | +0.0214 |
| 128 | 1_to_5 | val | 0.9291 | 5.702 | 0.8778 | +0.0190 |
| 128 | 1_to_last | train | 0.7594 | 13.243 | 0.5530 | +0.1283 |
| 128 | 1_to_last | val | 0.7723 | 13.220 | 0.5829 | +0.0303 |
| 128 | 5_to_10 | train | 0.9669 | 1.511 | 0.9125 | +0.0337 |
| 128 | 5_to_10 | val | 0.9677 | 1.512 | 0.9173 | +0.0096 |

Risk SD growth with near-perfect affine fit/Spearman supports mainly rescaling for that interval. Material rank changes or corrected/broken comparable pairs preclude saying the change is only amplitude. A positive affine rescaling cannot itself improve ranking, apart from numerical tie-tolerance effects.

## Representation/attention drift

| Batch | Phase | Split | Pool512 CKA | Post256 CKA | Attention JS | Top128 Jaccard |
|---|---|---|---:|---:|---:|---:|
| 32 | 10_to_last | train | 0.8758 | 0.8232 | 0.17258 | 0.4207 |
| 32 | 10_to_last | val | 0.9125 | 0.8871 | 0.16563 | 0.4253 |
| 32 | 1_to_last | train | 0.6490 | 0.4701 | 0.35794 | 0.1020 |
| 32 | 1_to_last | val | 0.7114 | 0.5789 | 0.35198 | 0.1010 |
| 32 | 5_to_10 | train | 0.9318 | 0.8971 | 0.07367 | 0.4068 |
| 32 | 5_to_10 | val | 0.9418 | 0.9206 | 0.07327 | 0.4070 |
| 64 | 10_to_last | train | 0.8347 | 0.7673 | 0.19409 | 0.3072 |
| 64 | 10_to_last | val | 0.8692 | 0.8324 | 0.18929 | 0.3131 |
| 64 | 1_to_last | train | 0.6493 | 0.4575 | 0.30933 | 0.0798 |
| 64 | 1_to_last | val | 0.6967 | 0.5330 | 0.30540 | 0.0789 |
| 64 | 5_to_10 | train | 0.9499 | 0.9214 | 0.03241 | 0.5361 |
| 64 | 5_to_10 | val | 0.9543 | 0.9287 | 0.03218 | 0.5355 |
| 128 | 10_to_last | train | 0.8872 | 0.8330 | 0.10212 | 0.4376 |
| 128 | 10_to_last | val | 0.9013 | 0.8626 | 0.10107 | 0.4335 |
| 128 | 1_to_last | train | 0.6898 | 0.4866 | 0.18513 | 0.0789 |
| 128 | 1_to_last | val | 0.7159 | 0.5174 | 0.18317 | 0.0799 |
| 128 | 5_to_10 | train | 0.9464 | 0.9218 | 0.02191 | 0.4063 |
| 128 | 5_to_10 | val | 0.9506 | 0.9261 | 0.02176 | 0.4042 |

CKA/attention drift describes frozen-checkpoint behavior, not biological correspondence or evidence that later features contain no DFS information. Attention entropy near1 also does not imply numerically identical weights.

All Train/Val diagnostic curves use the same fixed model in eval mode. Original online Train metrics remain in each training log. Each plotted mean±SD includes all5 folds only; later epochs with early-stopped folds are excluded from the plot, and counts remain available in CSV.

Absolute Train/Val Cox losses use different risk-set sizes and are not directly comparable as a gap; compare each split over time.

These are exploratory internal-validation results on a repeatedly inspected cohort. They cannot establish an intrinsic UNI information ceiling or prove that event count is the sole bottleneck. No new architecture, L2, label changes or teacher/student warm-starting were introduced. Private patient risks/attention/embeddings and all weights stay local.
