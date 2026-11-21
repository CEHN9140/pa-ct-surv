# Read-only patient/QC/Attention follow-up

No neural training. Existing bs32/64/128 diagnostics supply patient risks/attention; only bs32 five last checkpoints undergo the frozen Attention intervention. Compression, post-pooling MLP and risk head remain fixed; compare native late, Epoch5 weights, predefined50/50 early-late mixture and uniform weights. Native late risks reproduce exactly. See report.md for results and limits.

Reproduction order, using the project .radiomics_env/bin/python: analyze_patients.py; qc_thumbnails.py; attention_intervention.py (CUDA); summarize.py. Original feature/coordinate files, TRIDENT thumbnails, private diagnostic snapshots and local patient_error_profiles.csv must be present. Patient incidence is normalized by comparable partners; four patients without partners have undefined harm and are excluded from outcome associations. All30 pair decompositions match original results.

Thumbnail whiteness is a coarse background proxy, not tumor/normal/artifact annotation. Weight swaps are frozen-model diagnostics and can be outside the prediction head's training distribution. Bootstrap intervals condition on fixed models and omit retraining, prior exploration and multiplicity. Do not interpret the mixture as an already validated improved training method.

Published: code, aggregate tables, validation and report only. private/ retains patient IDs, outcomes, individual risks, QC and Attention predictions locally. No weights, patch images, arrays or source data are published. Existing main training and completed diagnostic results are unchanged.
