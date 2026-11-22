# Best-reference frozen attribution follow-up

User design: no retraining. Reanalyze best_attention→late_attention with fixed late H/head; add complete eval Train/Val pair comparisons and best-based patch QC. Preserve previous Epoch5 analysis.

1. Unit-test contextual pair-margin scores (both endpoints, loss/gain/ties), within-WSI matched selection, and technical matching failure handling.
2. All five models infer all721 patients each (3605 fold-patient observations). Verify native last risk/attention; exact IG best-reference completeness; reuse prior validation IG only after risk/array checks. Separate actual best→late and frozen intervention.
3. Reconstruct all comparable pairs separately within each fold Train/Val. Export descriptive C-index/margin/credit changes and patient roles. Do not pool repeat Train appearances as independent patients. Best checkpoint was selected on Outer Val: this selection bias cannot be removed retroactively.
4. Compute both negative margin mass on worsened pairs and positive mass on corrected pairs per patch. Also preserve opposing mass; same patch may support both correction and damage in different contexts. These are contextual descriptive scores, not unique allocations of C-index loss.
5. For Val patients and15 predefined Train review examples, select up to8 high-context-harm patches (fallback abs IG for no-harm review controls), match disjoint low-harm patches within the same WSI using a white-fraction proxy caliper0.1 and nearest UNI cosine. No tissue labels exist: do not call this histology matching. Keep match coverage/distances, no forced poor match.
6. Native SVS QC and blinded panels:15 Val cases (two most deteriorated actual best→late plus one stable per fold);15 Train cases (two most improved plus one stable). No automatic tissue diagnosis. Val QC uses harmed-context cases; fallback controls clearly marked/excluded from harm aggregate.
7. Independently recompute pairs/IG sums, conditional paired Val bootstrap, fold-level Train/Val contrasts, matching/QC summary and report. No significance from overlapping Train folds. Source pinning, CPU and realGPU smoke before background launch. No automatic publication.

Private: all patient rows, arrays, images, coordinates and review key. Public aggregates only. Same-WSI matching does not establish same tissue type or remove all confounding; best-conditioned 5/5 negative signs are exploratory and not independent causal evidence.
