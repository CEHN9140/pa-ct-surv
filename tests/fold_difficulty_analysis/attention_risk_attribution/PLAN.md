# Frozen attention-risk attribution diagnostic

User-approved design: retain original bs32 five-fold models, no training or architecture/loss changes. Compare Epoch5 and last actual predictions, and separately freeze last projection/head and intervene early vs late attention.

- [ ] Test exact Integrated Gradients on the piecewise-linear eval risk head, including nonlinear sign-crossings, zero pooled displacement, singleton bags and completeness. Test survival comparable pairs/ties against sksurv.
- [ ] Load original fold roles/checkpoints/snapshots, verify all patch coordinates and native late risks/weights. Compute exact attention-path IG for all outer validation patients. Persist every per-patch attribution locally.
- [ ] Connect counterfactual attention-only and temporal changes to comparable patient pairs. Report both corrected/broken pairs and pair-margin contribution accounting, without attributing all temporal changes to attention.
- [ ] For each WSI, preselect eight largest absolute-IG patches and eight least-absolute-IG controls (disjoint, equal count when possible). Read original patches at the stored field of view. Measure background, black pixels, tissue-masked sharpness and optical-density/color summaries. These are technical proxies, not tissue labels.
- [ ] Within-WSI high-minus-low comparisons, fold summaries and paired patient bootstrap. Relate within-WSI contrasts to patient pair harm; report uncertainty and no causal/independent-prognostic claims.
- [ ] Prepare 15 blinded representative review cases (per fold two worst actual temporal pair changes and one stable control), with patch and location panels plus blank review table. Never auto-label histology.
- [ ] Run unit checks, real-patient GPU replay/completeness and WSI-QC smoke; freeze source hashes and run complete diagnostic in background. Summarize all721 patients and5 folds, no new neural training.

Artifacts stay in this directory. Patient IDs, coordinates, patch images and arrays are private and excluded from Git. No automatic push is added for this task unless requested. Existing checkpoint early stopping used outer validation; all follow-up interpretations remain exploratory and conditional on these fitted models.

Latest scope: actual Epoch5/Epoch10/best/late risks are all retained. Exact IG uses both Epoch5 and best-attention references with the same late feature/head. Primary QC/review selection remains Epoch5-reference absolute IG; each selected patch also records its best-reference contribution, so this is not a separate best-reference Top-K quality search. Worst/stable review case selection uses actual Epoch5-to-late pair changes; attention-only harm is reported separately. No train/validation tissue-association study is claimed at this stage.
