# Concentration versus patch assignment

No new training. Existing bs32 last-models and Epoch5 attention are used. intervene.py compares native late, predefined50/50 early-late mixture, the mixture weight multiset reassigned by Late ranking, Neff-matched rank-preserving temperature and uniform weights. The exact-profile control preserves the entire weight distribution (Neff/entropy/Top128 mass), unlike a scalar concentration match alone. Actual inference preserves Late weak ordering; native risk and prior mixture reproduce.

curves.py plots all-epoch eval C-index/effective patch counts. Top128 mass and mean of per-patient Neff/N are snapshot-only scatter points; missing epochs are not interpolated. Per-fold best checkpoints are validation-selected; subsequent decline is descriptive and not causal evidence. The figure has been visually checked.

Reproduction with .radiomics_env/bin/python: test_matching.py through pytest; curves.py; intervene.py (CUDA); report.py. Set PYTHONDONTWRITEBYTECODE=1 and MPLCONFIGDIR to private/matplotlib. Private source snapshots/checkpoints and original UNI files must be present. Report.py recomputes25 fold C-indices and paired2000-resample conditional intervals.

The isolated component contrasts have nominal intervals crossing0; retain uncertainty and do not claim independent causal percentages. None of these frozen policies beats the original selected best checkpoint. No concentration threshold or training regularizer was selected.

Public: code, report, aggregate tables, curve figure and validations. private/ contains patient predictions, temperatures, matching audits and caches and stays local. No images/weights/source data or unrelated experiment changes are published.
