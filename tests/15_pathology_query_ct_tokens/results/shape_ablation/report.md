# Stage1 Shape ablation

721 patients / 251 events / original five outer folds. Stage1 only; no neural/Probe fitting or auxiliary supervision.

PA reference Val C-index: 0.7390 ± 0.0350.

V=log(volume/mm3), D=maximum3D diameter(mm), S=sphericity. VD also is leave-S-out; no duplicate model.

## true Shape

| Group | Val C-index | AUC36 | AUC60 | ΔC vs PA [nominal95% interval] | Positive folds |
|---|---|---|---|---|---|
| V | 0.7448 ± 0.0422 | 0.7825 ± 0.0243 | 0.7614 ± 0.0629 | +0.0058 [-0.0051,+0.0186] | 4/5 |
| D | 0.7488 ± 0.0395 | 0.7897 ± 0.0193 | 0.7658 ± 0.0563 | +0.0098 [-0.0021,+0.0229] | 4/5 |
| S | 0.7507 ± 0.0262 | 0.7917 ± 0.0144 | 0.7667 ± 0.0350 | +0.0117 [+0.0001,+0.0237] | 4/5 |
| VD | 0.7464 ± 0.0392 | 0.7868 ± 0.0221 | 0.7633 ± 0.0551 | +0.0074 [-0.0034,+0.0196] | 4/5 |
| VS | 0.7521 ± 0.0286 | 0.7926 ± 0.0172 | 0.7676 ± 0.0445 | +0.0131 [+0.0006,+0.0260] | 4/5 |
| DS | 0.7513 ± 0.0351 | 0.7930 ± 0.0177 | 0.7671 ± 0.0499 | +0.0123 [-0.0001,+0.0256] | 4/5 |
| VDS | 0.7520 ± 0.0327 | 0.7933 ± 0.0176 | 0.7671 ± 0.0466 | +0.0130 [+0.0014,+0.0254] | 3/5 |

## predicted Shape

| Group | Val C-index | AUC36 | AUC60 | ΔC vs PA [nominal95% interval] | Positive folds |
|---|---|---|---|---|---|
| V | 0.7445 ± 0.0370 | 0.7830 ± 0.0232 | 0.7609 ± 0.0583 | +0.0055 [-0.0041,+0.0160] | 4/5 |
| D | 0.7446 ± 0.0365 | 0.7842 ± 0.0238 | 0.7619 ± 0.0571 | +0.0056 [-0.0027,+0.0149] | 4/5 |
| S | 0.7447 ± 0.0368 | 0.7834 ± 0.0267 | 0.7610 ± 0.0574 | +0.0057 [-0.0037,+0.0161] | 3/5 |
| VD | 0.7457 ± 0.0382 | 0.7844 ± 0.0247 | 0.7627 ± 0.0594 | +0.0067 [-0.0027,+0.0169] | 4/5 |
| VS | 0.7432 ± 0.0358 | 0.7813 ± 0.0225 | 0.7586 ± 0.0570 | +0.0042 [-0.0057,+0.0147] | 4/5 |
| DS | 0.7435 ± 0.0336 | 0.7811 ± 0.0244 | 0.7595 ± 0.0553 | +0.0045 [-0.0051,+0.0147] | 3/5 |
| VDS | 0.7439 ± 0.0349 | 0.7833 ± 0.0232 | 0.7611 ± 0.0579 | +0.0049 [-0.0049,+0.0161] | 4/5 |

## Conditional additions (current minus reduced model)

| Source | Added | Existing | ΔC [nominal95% interval] | Positive folds |
|---|---|---|---|---|
| true | V | DS | +0.0007 [-0.0021,+0.0035] | 3/5 |
| true | D | VS | -0.0002 [-0.0038,+0.0033] | 3/5 |
| true | S | VD | +0.0056 [+0.0004,+0.0110] | 4/5 |
| true | D | V | +0.0016 [-0.0041,+0.0073] | 2/5 |
| true | V | D | -0.0024 [-0.0056,+0.0009] | 1/5 |
| predicted | V | DS | +0.0004 [-0.0037,+0.0047] | 2/5 |
| predicted | D | VS | +0.0007 [-0.0026,+0.0043] | 3/5 |
| predicted | S | VD | -0.0018 [-0.0067,+0.0036] | 2/5 |
| predicted | D | V | +0.0012 [-0.0008,+0.0033] | 4/5 |
| predicted | V | D | +0.0010 [-0.0027,+0.0046] | 2/5 |

## Descriptive correlations

| Source | Pair | Pearson | Spearman |
|---|---|---|---|
| true | V-D | 0.9367 | 0.9663 |
| true | V-S | -0.6025 | -0.6435 |
| true | D-S | -0.7273 | -0.7494 |
| predicted | V-D | 0.9718 | 0.9760 |
| predicted | V-S | -0.8288 | -0.8505 |
| predicted | D-S | -0.8900 | -0.9039 |

## Interpretation boundaries

- True and predicted feature subsets answer different questions: phenotype value versus value remaining after decoding. A null predicted-only result does not establish absence of true-phenotype value.
- Conditional increment is relative to the current scalar pathology risk predictor, not all information in the WSI. A positive result does not establish an independent biological mechanism.
- All subsets refit/reselect their own Cox ridge within the same saved inner contexts. Leave-one-out differences measure entire predictive pipelines; correlated features and retuning prevent interpreting them as unique causal contributions.
- No features or auxiliary targets are selected automatically from outer results. No CT-only, teacher or student auxiliary-supervision experiment is started. This is repeated exploratory analysis, not independent confirmation.
- Intervals bootstrap paired patients within each fixed outer fold2000 times; they are nominal, not adjusted for multiple comparisons, and exclude network/Probe retraining, tuning and prior exploration uncertainty.
- True Shape uses tumor masks. Predicted Shape uses public512 vectors with training-mask phenotype supervision. Existing CT input is mask-localized ROI64; this does not make the entire pipeline mask-free.
- Cox fits PA OOF plus saved Shape OOF. Full-Probe own-training predictions are used only for descriptive train metrics/KM cutoffs, never Cox fitting. Source PA/true-VDS/predicted-VDS patient predictions remain unchanged.
- Full VDS parity requires the same selected ridge, risk error at most1e-8 and identical pair ordering/ties. A preflight true-VDS fold1 refit differed by about3e-9 from solver roundoff at Cox tolerance1e-9; applying the saved old head to current covariates differed by only4e-16.
- Means/sample SD are across five folds. Full train/val metrics,NLL,HR,KM,paired subset/full/matching-true comparisons and per-fold changes accompany this report.
