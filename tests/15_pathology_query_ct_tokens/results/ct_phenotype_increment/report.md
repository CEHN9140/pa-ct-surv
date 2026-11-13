# Predefined CT phenotype increment: exploratory diagnostic

721 patients, 251 events, unchanged five outer folds. No neural models were retrained; no MedicalNet probes were run.

Pathology reference C-index: 0.7390 ± 0.0350.

| Phenotype | CT-only C-index | PA+CT C-index | Δ vs PA (95% paired interval) | Positive folds | AUC36 | AUC60 |
|---|---|---|---|---|---|---|
| Shape | 0.7044 ± 0.0471 | 0.7520 ± 0.0327 | +0.0130 [+0.0014, +0.0254] | 3/5 | 0.7933 ± 0.0176 | 0.7671 ± 0.0466 |
| FirstOrder | 0.7245 ± 0.0536 | 0.7488 ± 0.0402 | +0.0099 [-0.0009, +0.0213] | 4/5 | 0.7906 ± 0.0271 | 0.7633 ± 0.0560 |
| Texture | 0.7031 ± 0.0507 | 0.7379 ± 0.0339 | -0.0011 [-0.0083, +0.0068] | 3/5 | 0.7760 ± 0.0285 | 0.7505 ± 0.0483 |
| Peritumoral | 0.6755 ± 0.0676 | 0.7407 ± 0.0390 | +0.0017 [-0.0064, +0.0108] | 3/5 | 0.7776 ± 0.0279 | 0.7548 ± 0.0511 |
| Combined | 0.7318 ± 0.0603 | 0.7491 ± 0.0414 | +0.0101 [-0.0024, +0.0231] | 4/5 | 0.7871 ± 0.0287 | 0.7614 ± 0.0554 |

Combined minus Shape: -0.0029 [-0.0107, +0.0054].

Peritumoral 5 mm coverage: 666/721 complete; minimum 92.3013%. All patients retained; coverage is QC only.

## Interpretation limits

- Means and sample SD are across five folds. Paired intervals bootstrap patients within each fixed validation fold, conditional on fitted models; they exclude retraining, tuning and prior exploration uncertainty.
- All groups depend on tumor masks. Shape reuses native-mask geometry; intensity and texture use only the saved ROI64 and valid observed shell, excluding padding. This is mask-assisted information, not a demonstrated capability of an unmasked encoder.
- Recovered integer intensities are clipped-input proxies. Upstream intensity calibration is unknown; inverse normalization cannot recover the original raw HU.
- Combined versus Shape tests the added feature bundle after retaining Shape. Other groups versus Shape are replacement comparisons, not independent effects adjusted for Shape.
- Conditional value is relative to the current pathology risk predictor, not all WSI information. Multiple predefined comparisons remain exploratory; a null result does not prove absence of other CT information.
- Validation Cox partial NLL is event-normalized Breslow loss, not calibrated survival probability error. Full NLL, AUC, per-fold metrics, ridge selection and pairwise comparisons are saved in the accompanying tables/JSON.
- The formal method remains a jointly trained pathology–CT teacher and CT-only student. No downstream unimodal initialization is introduced.
