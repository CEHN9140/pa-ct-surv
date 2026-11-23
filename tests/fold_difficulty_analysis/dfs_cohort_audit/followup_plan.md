# EE/EC and collection-batch follow-up plan

Goal: reuse all four matched Baseline seeds and five outer folds; no training, labels, official split changes, checkpoint selection or patch filtering.

Files: pair_followup.py (read-only analysis), test_followup.py (pair classification / IPCW reference tests), followup_results/ (aggregate CSV, JSON and report). Reuse existing audit.pairs_for and private metadata; publish no patient IDs.

- [x] Check train/validation prediction membership, float32 endpoint agreement and finite risks; reconstruct all 40 C-indices.
- [x] Compare Train/Val EE vs EC, EE event-time gaps (0,6], (6,12], (12,24], >24 months; include counts, margin and risk-tie fraction.
- [x] Describe both collection batches' event/censor times, reverse-KM follow-up, KM survival36/60; retain the single supplementary case as its own group.
- [x] Describe risk by batch in training-risk SD units, and pair performance by within/cross batch with event-anchor direction. No pooling raw scores across models/folds.
- [x] Compute Uno at fixed tau36/60 months using Outer Train censoring KM. Compare with Harrell using the identical event-anchor t<tau pairs and retain all later comparators. Verify against sksurv.concordance_index_ipcw; report unsupported cases rather than changing tau.
- [x] Test meaningful tie/boundary/IPCW behavior, run audit, compare prior Val replay and review interpretation limits. Summarize means over seeds/folds; pair counts are not independent observations. No new significance claims or model changes.

This is exploratory follow-up on repeatedly inspected data. Collection batch is not a verified technical site or scanner. Overall Uno is not an EE-specific correction and marginal censoring KM does not establish censoring independence. No new loss or architecture is justified solely by this diagnostic.
