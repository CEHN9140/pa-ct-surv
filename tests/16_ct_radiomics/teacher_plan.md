# Experiment16: pathology-query / CT-Radiomics-key-value teacher

User approved the previously proposed seven-group token design and requested
implementation inside test16. Main training, the completed Radiomics baseline
and test15 remain unchanged. No automatic commit/push/full-training launch.

## Model

- Same721 patients/251 DFS events and saved five folds. Read full107 Original
  features from test16/features cache, joined by CT ID and checked against PA ID,
  event/time/split. Read the corresponding EAGLE Top25 Virchow2-CLS [25,1280].
- Refit constant/Spearman>.90 filter and Z-score on each actual neural FIT set.
  Do not reuse the completed Cox model or its fitted preprocessing. Group the
  retained columns into Shape/FirstOrder/GLCM/GLRLM/GLSZM/GLDM/NGTDM; each group
  has an independent Linear(group_dim,256)-ReLU-Dropout.25 encoder.
- Fixed seven group positions; empty groups contain zero and are masked out of
  key/value attention. Require at least two nonempty groups. These are feature
  families, not CT locations or guaranteed independent biological processes.
- PA Patch L2 before one shared ABMIL projection1280->512. Query projection
  512->256. Standard one-head scaled-dot-product MHA, Q=25 PA tokens,
  K/V=7 CT tokens. Keep existing test15 attention-weight dropout0; projection,
  ABMIL attention/postprojection, query pooling and fusion-head dropout use .25.
  No QK normalization or feature L2 after modality encoding.
- Pool cross output25x256 with learned attention ->256. Independently retain
  original ABMIL PA pooling512->256. Concat512->256->128->1 risk head, same
  LayerNorm/ReLU/Dropout as existing ConcatFusion. Only fused Cox loss.
- All learnable modules newly initialized and jointly trained. The ABMIL
  unimodal risk head is removed/Identity because no auxiliary risk is optimized.
  Fixed patch embeddings and deterministic Radiomics are allowed inputs; no
  project-trained PA/CT/Ridge checkpoint initializes the teacher.
- Small optional concat control: average valid CT group embeddings and concat
  with the same PA representation/head. Same data/selection/training entry;
  no automatic additional control experiment.

## Data / training / evaluation

- Preload the small Top25 feature tensors once; no CT image reads needed for
  teacher fitting. Match file stem to PA ID; reject wrong bag size, nonfinite
  values, missing/duplicate/mismatched patients or labels. All data handling and
  FIT-only preprocessing live in teacher_data.py.
- Seed123 in Python; Adam all parameters lr1e-4, weight_decay5e-4. Training
  loader batch1; joint Cox32 patient forwards plus tail merged into preceding
  risk set. No gradient accumulation of independent per-patient Cox losses.
- Same outer5 / inner3 event-stratified roles as previous diagnostics, derived
  locally without importing other tests. For each inner split, 85% remaining
  FIT /15% STOP, with separate HELD unused for learning or selecting epochs.
  Each inner filter/scaler sees FIT only. Best STOP C-index, patience10, max30.
- Outer epoch count=floor median of3 STOP-selected epochs; refit preprocessing
  on all outer-training patients and a completely fresh network for that many
  epochs. No outer validation checkpoint/epoch/model selection. Fifteen inner
  and five outer fits per fusion; original patient split never regenerated.
- Training metrics are accumulated train-mode risks over the epoch. STOP and
  final Train/Val evaluate eval() over all relevant patients; full-risk-set Cox
  loss. Shared AUC36/60, KM/train median, HR, metrics mean/sample SD.
- Save weights plus feature preprocessing/group layout so inference is fully
  reproducible. Results/checkpoints/logs under separate teacher_* directories;
  large files and patient-level tables remain excluded by Git allowlist.

## Initial implementation verification (before Cross-Attention full training)

- [x] Tests first: IDs/labels/shuffled-table alignment, FIT-only selection and
  scaling, empty-group handling, input L2, MHA Q/K/V/manual formula/axes/masks,
  query and CT influence, gradient flow, Cox tail, nested split isolation and
  checkpoint/preprocessor roundtrip.
- [x] Implement data/model/training/launcher with defaults and concise docs.
- [x] Run focused tests and original test16 regression checks.
- [x] Audit all721 real inputs and inner/outer feature groups, then real-data
  short training/eval/reload smoke (not a scientific performance result).
- [x] Independent read-only code review, fix important issues, provide command.

Verification: nine teacher tests plus seven original Radiomics tests pass.
Real-data GPU smoke used20 FIT/10 STOP patients, at most two epochs per fit,
risk-set sizes8+12; selected checkpoint reload was bit-exact, fixed-epoch fit,
full Cox and KM generation succeeded. No complete five-fold training launched.
Source extraction signature (parameters/version/CT and Mask size/mtime) and
cohort/cache/patch feature hashes are checked. All new learned modules remain
jointly trainable; no project-trained unimodal model is loaded.

## Approved follow-up: Concat and matched PA-only

Cross-Attention five-fold training is now complete (see its completed.json).
User selected only Concat and PA-only controls; no risk-level fusion in this run.
Keep existing completed Cross-Attention outputs immutable. Launch two new runs
in parallel after tests and small real-data verification.

- Concat: existing seven FIT-encoded CT families -> valid-token mean256;
  concatenate PA256 ->512, same256/128 hidden risk head. No MHA/query pooling.
- PA-only: same patch L2, ABMIL1280->512, pooling and512->256; risk head
  256->256->128->1, same LayerNorm/ReLU/dropout. First input width changes from
  512 to256 because CT is absent. Native new initialization, same seed123;
  shared ABMIL starts identically, head weights need not be bit-identical across
  different input widths. No CT parameters, CT feature filtering/scaling or CT
  input dependence. Common paired cohort still audited for patient membership.
- Same optimizer, loss coefficient1, Cox32+merged tail, saved outer folds and
  local inner FIT/STOP/HELD roles, median STOP-selected epochs and fresh outer
  refit. Each independently selects its own epochs without outer outcomes.
- Reuse one training entry and checkpoint API; PA-only preprocessor=None and
  no unused scaler file. Log/output names concat and pa_only remain separate.
- Verify PA-only invariance to CT changes/None, correct head dimensions and
  gradient flow, native shared ABMIL initialization, STOP/fixed training and
  checkpoint roundtrip in both control modes; reload old cross checkpoint to
  guard backward compatibility. Update smoke-vs-full completion records and
  publish allowlist only for aggregate small results, never patient tables or
  checkpoints. Do not automatically commit/push.

Follow-up verification complete:19 tests pass (12 teacher/control,7 original
Radiomics); real20 FIT/10 STOP GPU smoke passes for each control including
selected-checkpoint reload and8+12 tail sizes. Five old Cross-Attention outer
models still reproduce metrics. Read-only review found no important defects.
Both full controls have now completed all20fits each. See
teacher_controls_validation.json for launch and completion metadata.
Post-training inference audit verified60fits/15outer-model reloads across all
three modes, computed2000 paired-patient bootstrap replicates and100CT
whole-vector permutations per fold for each teacher. Aggregate results are in
teacher_results/comparison; no risk-level fusion or new neural training in
these diagnostics.
