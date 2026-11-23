# Strict nested patient-number learning curve

Same721/251 and5outer fixed, same CONCH raw bags / cached frozen TITAN features. Target128/256/384/fullOuterTrain. Three prespecified sampling seeds123/7/42; event-stratified prefixes nested within repeat, canonical order for learners. Full point once, not replicated. Neural initialization fixed123 to isolate sampling variation.

- [ ] TDD: private subset plan nested/disjoint/count/event balance; same per-patient epochs mapped to complete-epoch step budgets and dropout schedule; inner selection completeness/ties; repeat-first aggregation rejects incomplete runs.
- [ ] Freeze config/cohort/feature cache identity and50subset jobs. Shared3inner event-stratified roles per subset for both methods. Public composition summaries only; patient plan private.
- [ ] ABMIL50conditions ×4fits =200new fits; reuse Stage6 training core with epoch-to-step mapping, max30epochs all inner FITs. OuterTrain subset refit fixed chosen epochs, OuterVal final once. Record actual updates/exposures and boundary selections, no labels used for sampling outside OuterTrain.
- [ ] TITAN45new small-sample Ridge fits (1350inner attempts) +5exactly reused full reference models. Use same subset/roles, FIT-only scaler and fixed alpha grid, all3fold failure rules. No new TITAN inference or fine-tuning.
- [ ] Aggregate50ABMIL+50Ridge final conditions. Within each outer fold average3repeats before five-fold mean/SD; full is singleton. Same final-model Train/Val evaluation and EE/EC/EEgap, AUC failures preserved. Plot comparable curves with foldSD (not CI), record event/followup composition.
- [ ] Verify200neural budgets/exposures/initializations, selection/roles,1500Ridge CV records including150reused vs1350fresh, input/code stability and private-output exclusion. Report epoch-vs-update/risk-set composition confounding and no causal proof of limited sample size or WSI ceiling. Publish only completed code/aggregate artifacts.
