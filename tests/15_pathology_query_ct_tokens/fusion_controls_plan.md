# Matched fusion controls: approved design and implementation plan

Goal: distinguish CT incremental prediction from learned spatial interaction,
using the original Top25 standard-attention experiment as reference C.

All additions stay in experiment 15. Main training/data/model files are unchanged.
The user approved implementing and launching both controls on 2026-11-12.

## Design

- A, `pathology_only`: retain the original ABMIL patient route and exactly the
  same concat risk head. Its CT input vector is identically zero. Remove the
  CT encoder and unused query/CT-attention modules after constructing the shared
  initialization, so retained PA/head weights start exactly as in C. No CT
  forward or CT gradient exists. This is a pathology ablation, not a frozen-CT
  teacher. Return empty CT attention and zero diagnostic CT risk in the existing
  seven-output interface.
- B, `uniform_ct`: retain the same public-pretrained CT encoder, PA branch,
  CT projection, internal Value projection, attention output projection, query
  pooling and concat head. Replace learned spatial weights by `1 / N_CT`:
  `out_proj(mean(linear(ct_projection(tokens), Wv, bv), dim=1))`.
  Broadcast to the 25 query outputs before the unchanged query-pooling stage.
  CT and PA jointly train with fused Cox only. No trained unimodal checkpoint,
  auxiliary/alignment loss, QKNorm, CT Z-score or adaptive BN is introduced.
- Common settings: original paired CSV and five splits, 1280-D Top25 patch
  features, patch L2, ROI64, Cox32 with merged tail, dropout .25, Adam lr1e-4,
  CT backbone lr1e-5, weight decay5e-4, 30 epochs, patience10, seed123 per fold.
- Keep the original constructor/CLI behavior and old checkpoint keys unchanged
  when no control is selected. Persist the control in config and reconstruct it
  in checkpoint inference. Original-QK diagnostics must exclude these controls.
- Give both runs new suffixes and independent logs/results/checkpoints; never
  overwrite an existing run. The already-completed C remains unchanged.

## Execution checklist

- [x] Add failing `test_fusion_controls.py` tests: A exact zero-slot/head parity,
  CT independence in train/eval and absent CT encoder; B exact uniform internal
  Value/output-projection formula, query independence and CT sensitivity;
  matching initialization, fused-Cox gradients, fixed BN, strict checkpoint
  reload/cached inference and CLI guards against mixed interventions.
- [x] Implement local `fusion_control` in `path_query_ct.py` and `run.py`;
  adapt inference reconstruction and exclude controls in original score diagnosis.
- [x] Add `run_pathology_only.sh`, `run_uniform_ct.sh` and document both commands.
- [x] Run focused then all experiment-15 unittest tests with two CPU threads;
  shell syntax-check the launchers and review the final local diff.
- [x] Run a real ROI64 CUDA smoke: compare retained initialization to C, train
  and validate tiny paired subsets via the existing trainer/evaluator in /tmp,
  check gradients, unchanged BN, strict reload and cached/full prediction parity.
- [x] Start both nohup jobs in independent sessions, verify exact saved configs,
  721-patient/five-fold startup, actual PIDs and first successful updates.

Verification: all 43 unittest tests pass (76.6 seconds); both real-data smoke
runs complete one full 32-patient Cox update, eval, strict checkpoint reload and
cached/full prediction parity. Uniform-CT CT/PA/Value gradients are finite and
nonzero, Q/K score gradients zero, and all BN buffers remain unchanged. Smoke
peak allocated GPU memory is approximately 38 MiB for A and 1696 MiB for B.
Both full runs reached CUDA Fold 0, completed epoch 1 and saved best checkpoints;
saved configurations and cohort hashes match the original reference settings.
