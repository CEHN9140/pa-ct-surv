# Frozen attention-risk attribution and native-patch quality diagnostic

No new neural training. Original Raw UNI full bags and bs32 five-fold last checkpoints are frozen. Epoch5 attention is the reference. All outputs are exploratory, conditional on models originally early-stopped using outer validation.

## What is decomposed

For each patient, freeze late projected patch vectors and risk head; interpolate normalized Epoch5 to late attention. The eval head is Linear/ReLU/Dropout(identity)/Linear. Exact integration of each ReLU active fraction yields patch IG. All patch contributions sum to the counterfactual attention-only risk change. This is not patch deletion, a unique causal attribution, or attribution of the entire actual early-to-late prediction change.

Actual Epoch5/Epoch10/best/late risks and best-attention-to-late IG are recorded separately. patients.best_to_late_delta denotes frozen best-attention-to-late risk, while pairs.best_to_late_credit_change denotes actual best-model-to-late credit; these are different comparisons. All comparisons use comparable survival pairs including event/censor at the same time and half-credit for risk ties. Both endpoint contributions are needed: pair-margin delta equals event-patient summed IG minus comparator summed IG. A positive risk contribution can help or harm depending on the paired patient and event ordering.

| Fold | Actual E5 C | Actual best C | Actual late C | Early attention / late head C | Late attention / late head C | Attention delta C | Corrected pairs | Broken pairs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.6722 | 0.6902 | 0.6757 | 0.6702 | 0.6757 | +0.0054 | 382 | 354 |
| 1 | 0.7488 | 0.7559 | 0.7339 | 0.7293 | 0.7339 | +0.0045 | 337 | 313 |
| 2 | 0.7908 | 0.7908 | 0.7288 | 0.7602 | 0.7288 | -0.0314 | 166 | 327 |
| 3 | 0.7655 | 0.7719 | 0.7310 | 0.7513 | 0.7310 | -0.0203 | 212 | 319 |
| 4 | 0.6856 | 0.7202 | 0.6931 | 0.6816 | 0.6931 | +0.0115 | 258 | 203 |

## Matched native-patch quality

Per WSI: eight largest absolute-IG patches versus eight smallest absolute-IG controls, disjoint; fewer if necessary. Both directions of risk contribution are retained. Patches are read at native metadata-matched TRIDENT FOV, not approximated from thumbnails. Within-WSI matching controls slide-wide scanner/stain differences but not local tissue type or other patch confounders. This extreme-group contrast does not estimate WSI-wide artifact prevalence.

White/black fractions are technical proxies. Laplacian variance is measured inside an eroded non-background mask and also depends on tissue texture; it is not a validated blur label. Optical density/channel summaries are not H&E stain concentrations or tissue diagnoses. Blank patches may have undefined sharpness/OD: available-patient counts are explicit.

| Proxy | Patients with both values | High-minus-low mean | Conditional95% patient interval |
|---|---:|---:|---:|
| white_fraction | 721 | -0.47818 | [-0.49146, -0.46380] |
| black_fraction | 721 | -0.00302 | [-0.00504, -0.00135] |
| tissue_fraction | 721 | +0.48120 | [+0.46750, +0.49523] |
| laplacian_variance | 720 | +138.45350 | [+100.84189, +175.07400] |
| od_mean | 720 | +0.01648 | [+0.00961, +0.02279] |
| saturation_mean | 720 | +0.03584 | [+0.03253, +0.03922] |
| od_r | 720 | +0.01428 | [+0.00720, +0.02114] |
| od_g | 720 | +0.05194 | [+0.04308, +0.06063] |
| od_b | 720 | -0.01677 | [-0.02204, -0.01149] |
| valid_bounds_fraction | 721 | +0.00000 | [+0.00000, +0.00000] |

Quality differences and pair-harm correlations are descriptive associations, not proof technical quality caused overfitting. Multiple proxies and outcome-dependent follow-up are unadjusted; intervals omit retraining and prior exploration. Pair endpoints are dependent. No pathology diagnoses or stable DFS-related histological associations can be established by this script.

## Blinded review

private/reviewer contains15 cases: two worst actual Epoch5-to-late mean pair-credit changes and one stable control with fewest changed comparable pairs per fold. A/B are shuffled high-impact/control groups; IDs and group decoding stay in private/review_key.csv. Blank tissue/artifact/quality fields await an actual reviewer. Selection uses these same outcomes and therefore cannot serve as independent validation. Panels show native patches and WSI locations. No automatic histology label was generated.

## Interpretation boundary

A complete IG decomposition establishes computational accounting for the chosen frozen intervention, not biological causation. High absolute IG includes positive and negative risk changes, not just harmful patches. Absolute-IG controls differ in model influence by construction; individual patch allocations also depend on the chosen representation parameterization even when their sum is invariant. Correlated nearby patches are not independent tissue observations. If high-impact quality differences replicate across folds and relate to pair deterioration, that motivates a future predefined QC intervention; otherwise this diagnostic cannot distinguish unstable biological associations from limited DFS supervision or missing prognostic information. No new loss or network is warranted from a correlation alone.
