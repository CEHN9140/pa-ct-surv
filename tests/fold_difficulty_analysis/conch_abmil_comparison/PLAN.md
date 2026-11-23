# Matched CONCH MeanPool / scratch ABMIL / frozen TITAN

Scope: new scratch ABMIL only; reuse both completed Ridge references from TITAN comparison. Same721/251, original five outer folds and three event-stratified inner roles, source CONCH values/coords verified against original extraction hashes. No feature re-extraction or pretrained/project-baseline model loading. All artifacts isolated here.

- [x] TDD: source bag identity/numerical checks; 768-D fresh model, inner roles and step selection reuse; patient-bootstrap subgroup masks/ties/repeated-patient weighting.
- [x] Cache all FP32 CONCH bags, same upstream numerical values; compare all original hashes. Train15inner to180 steps at predefined candidates, select one step by3HELD mean, then5fresh outer refits. Use Stage6 run_outer unchanged with in_dim768. OuterVal final only.
- [x] Audit20 initializations, budgets/roles, no outer intermediate evaluations, model reload; save source/code hashes and all patient outputs private.
- [x] Compare all3 pipelines at matching final fixed checkpoints. Record Train/Val, AUC36/60, EE/EC/>24EE. Pairwise patient bootstrap within fold for all/group C deltas, same resampling across contrasts. These subgroup intervals are exploratory, no multiplicity or retraining correction.
- [x] Verify20fit records/30saved prediction metrics/paired deltas/privacy; report complete procedure differences, no pure Attention/pretraining causal claim. Publish only source and aggregate small files, no weights/features/IDs.
