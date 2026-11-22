# Best-reference frozen ABMIL follow-up

Independent extension of3af4670, leaving Epoch5 analysis and main training untouched. Original721 patients/251 events, Raw UNI all patches, existing bs32 five-fold late checkpoints. No training, new loss, architecture change, teacher/student initialization or automatic push.

Every fold evaluates all721 patients in eval mode: separate Train and Val comparable risk sets,3605 fold-patient observations in total. Actual best/late full-model risks and frozen late projection/head with best/late attention are different comparisons. All Val best-reference IG must reproduce the previous best_contributions. Exact attention-path IG completeness and both pair endpoints are checked. Best checkpoints used Outer Val selection; this exploratory follow-up cannot remove that bias. Bootstraps are conditional on fixed selected models.

## Context-aware scores

For each changed comparable pair, event-patient patch margin contribution is+c and comparator contribution is-c. Negative contributions support margin deterioration on pairs losing C-index credit; positive contributions support improvement on pairs gaining credit. Opposing contributions are retained too. Half-credit ties weight contexts by0.5. A patch can support both broken and corrected pairs in different contexts; high absolute IG does not mean harmful. Scores are not unique C-index allocations or causal explanations.

The raw mass sums pair exposures and risk-margin units: it is not directly comparable between Train and Val or across models. It is used to rank patches within the same WSI/model. Train/Val reporting compares split-specific C-indices and per-patient mean pair-credit changes. Train patients recur across4folds, so no independent-patient bootstrap/test is applied to the repeated Train sample.

## Quality matching

For Val patients and15 representative Train review examples: up to8 positive contextual-harm patches versus disjoint bottom-half low-score candidates; thumbnail white-fraction caliper0.1 and nearest UNI cosine without replacement. UNI L2 is only a matching metric; model inputs remain Raw UNI. Match distances/failures are retained. Native SVS tiles are read at the stored exact FOV; actual white differences are checked. Primary QC uses native white difference<=0.1 and excludes absolute-IG fallback review cases. White is a balance variable, not an outcome test for absence of background effects after matching.

No tissue-type annotations exist. Technical/embedding matching does not guarantee identical histology, sufficient stain matching, or causal confounding control. Blur/color/white proxies are not validated tissue or artifact labels. Same-tissue comparison remains pending actual pathology annotation.

## Review panels

Thirty planned cases: per fold2Val patients with greatest actual best→late deterioration plus1stable;2Train patients with greatest improvement plus1stable. A no-broken-pair control may use abs IG for display, explicitly marked and excluded from harm-QC aggregate. Case IDs and A/B group labels are shuffled; review.csv has blank tissue/artifact/quality fields. No doctor review has been performed automatically.

Private arrays, predictions/pair tables, matched tiles, coordinates and review keys stay under private/. Aggregate fold metrics, conditional Val intervals, matching coverage, native-QC contrasts and report go under results/. run.sh starts a no-fit analyze.py→inspect_patches.py→summarize.py pipeline with source hashes and verification gates. Status: status.json; log: validation/pipeline.log.
