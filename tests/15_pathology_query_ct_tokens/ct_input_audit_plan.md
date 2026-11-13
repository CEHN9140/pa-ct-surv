# CT input audit before representation experiments

User-requested order: audit inputs first, then compare CT layer3/layer4/combined
representations. This task implements a read-only audit of all721 existing
ROI64 patients (251 events), retaining their exact saved five outer folds.
No training or dataset/preprocessing rewrite during the audit.

Never import process_ct.py: its top-level code clears generated directories.
Read its source statically, hash it and independently reproduce its current
LPS orientation, mask alignment,1mm target grid, all-component bounding-box
centering, linear CT/nearest-mask sampling, HU[-1000,400] clipping and minmax.
Use small output grids for independent reconstruction rather than resampling
whole scans. Honor original output pixel type and crop padding outside the
full target grid. Compare all saved mask voxels and CT values; report exact
equality and one-HU interpolation-rounding tolerance separately.

For each raw NIfTI image/mask pair, record geometry, native file spacing,
raw-intensity range/regular-grid sampled quantiles and available acquisition
metadata. Current NIfTI spacing is not evidence of original DICOM resolution.
Whitelist only scanner/protocol fields; do not expose patient metadata.

Independently reconstruct the complete resampled tumor support from its
physical native-mask bounding box (with nearest-neighbor support halo), verify
the crop center/voxel counts and100% retention against both log and saved NPY.
Compute tumor fraction, face contact, minimum margin and5/10mm Euclidean
peritumoral-shell coverage inside the ROI AND acquired image field. Distinguish
full tumor retention from complete peritumoral coverage. Do not silently drop
or fix cases. HU statistics in ROI/tumor/shell refer to values before clipping;
global raw quantiles use deterministic regular sampling and are labeled so.

Also compare ROI64/96/128 patient sets, label agreement and common-case fold
assignments without changing any CSV. If original scanner/reconstruction
metadata are absent, report unavailability, not inferred protocol homogeneity.

Files: ct_input_audit.py, test_ct_input_audit.py. Outputs and log stay under
tests/15_pathology_query_ct_tokens/results/ct_input_audit and logs. Aggregates
and report are shareable; patient-level audit rows/failure details stay local.

Checklist:
- [x] Failing tests for output-grid physical origin/orientation, exact native
  sampling, anisotropic interpolation, acquired-domain padding, all-component
  tumor support, shell coverage and wrong ROI detection.
- [x] Implement audit and source/cohort/log hashes; per-case failure isolation
  with nonzero completion state on reconstruction/cohort errors.
- [x] Validate synthetic tests and real native1mm/anisotropic cases against
  full-volume preprocessing reconstruction in /tmp only.
- [x] Audit all721 cases, summarize input quality and uncertainty, retain all
  source files unchanged. Gate A/B/C training on the audit findings.

Audit complete:721 cases/251 events, zero case errors or hard reconstruction
issues, masks exact and all tumors retained. Two real native/anisotropic
whole-volume spotchecks match saved ROIs exactly; small-grid checks on all
cases are within1HU integer-interpolation rounding tolerance.75 tests pass;
independent review found no important issues. Source hashes cover only cohort,
script and preprocessing log. Representation training is not started because
719 input NIfTI maxima equal315 and upstream preprocessing/metadata provenance
is unresolved; user has been asked for source DICOM/NIfTI/conversion path.
See results/ct_input_audit/report.md for numeric findings and limitations.
