# CONCHv1.5 MeanPool vs frozen TITAN

Primary contrast: exact same CONCHv1.5 features/coordinates, 20x512 patches; full721/251, fixed original5outer and3inner(seed123), same Ridge grid0.1..1e8, FIT-only StandardScaler. No extra L2 or patch sampling, no DFS fine-tuning of CONCH/TITAN. Existing UNI .7201 is contextual (different field size), not a pure aggregation contrast. Do not automatically run ABMIL or sample-size curves.

- [x] Verify upstream CONCH model recipe and all721 source/coordinate/geometry/finite features; save mean and frozen TITAN matrices privately with hashes. Preflight one case and repeated inference before full extraction; reject grid collisions/invalid input rather than dropping cases.
- [x] Pin TITAN official revision and local weights; use offline local tokenizer/cache after authorized download. Read remote code before execution. Store weights only in user-designated directory, runtime dependencies/cache private. Never publish model code/weights copied from gated repository.
- [x] Test geometry/mean, normalization absence, official API input shape and frozen deterministic inference with a fake model. Run real checkpoint GPU smoke separately, never report fake results as scientific data.
- [x] Fit300inner attempts +10outer refits. Strict failure rules reuse Stage5 implementation; verify inner roles, model reload/predictions. All transforms/alpha selection use FIT, OuterVal final only.
- [x] Report C-index/AUC36/60, EE/EC and EEgap>24, paired nominal bootstrap difference TITAN−CONCH mean. No causal attribution to spatial modeling or pretraining separately, no WSI information-ceiling claims. No case filtering by OuterVal.
- [x] Verify source/weights/code, all saved scores, small public aggregate file allowlist. Push completed code/aggregate results only; if TITAN unavailable, explicitly mark blocked, do not invent results.
