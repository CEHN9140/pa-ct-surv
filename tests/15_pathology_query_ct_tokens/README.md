# Experiment 15: pathology queries CT layer4 spatial tokens

Repository scope: code, launchers, tests, logs, aggregate metrics, diagnostic
reports and KM plots are versioned. Checkpoints, Python caches, patient-level
prediction tables and permutation/patient-risk NPZ files remain local.

## Residual teacher: jointly trained pathology and CT

```bash
bash tests/15_pathology_query_ct_tokens/run_residual_teacher.sh
```

Runs only the two user-requested penalties0 and0.01. The new ABMIL branch uses
the original25 Virchow2-CLS1280-D patches, patch L2 and dropout0.25. CT loads
only the configured PUBLIC MedicalNet ResNet18 weights; layer4 global512 ->
Linear512,128 -> ReLU -> Linear128,1 gives the scalar correction dc. The last
CT layer is zero-initialized. Risk is rp+dc, without sigmoid. Both branches
train jointly; BN statistics stay fixed and BN affine parameters are trainable.
No project-trained unimodal checkpoint initializes either branch. Zero final
CT weights give zero CT-encoder gradient on the first update; after the output
weights change, encoder gradients flow normally (verified in a two-step test).

Objective: `Cox(rp+dc) +0.3*Cox(rp) +lambda_r*mean(dc**2)`. No CT-only Cox.
The CT correction head follows the requested512->128->1 structure without an
additional dropout layer; ABMIL dropout remains0.25. Patient forwards accumulate
into32-person Cox risk sets; the final tail merges into the preceding set.
Other settings: ROI64, PA/head LR1e-4, backbone LR1e-5, Adam wd5e-4, seed123,
max30 epochs, patience10 inside outer training only.

Uses the nested-risk diagnostic's same721 patients and five outer splits.
It reads only source roles/predictions/config for evaluation provenance, never
the source's downstream-trained weights. Each penalty has three fresh inner
teachers with separate fit/early-stop/held roles. Select lambda by mean held
C-index; exact ties favor0.01. Before any outer evaluation, save the selected
lambda and each candidate's median inner-best epoch. Reinitialize and refit
each fixed candidate on all outer-training patients for its fixed epoch count.
This is40 fits in total:30 inner +10 outer, not just two ordinary5-fold runs.

Outputs: `results/residual_teacher/residual_r0/`, `residual_r0p01/` for fixed
secondary comparisons and `inner_selected/` for the primary procedure chosen
entirely inside training. The primary reuses selected candidate predictions,
not another training run. Checkpoints remain under their actual candidate
names. Each outer fold has prediction tables with rp/dc, KM/HR/IPCW AUC36/60,
loss components and `pair_changes.json`: strict errors corrected/broken and
all ranking gains/losses including ties. Corrections are compared both with
the joint model's rp alone and the independently trained nested pathology
baseline. Neither comparison by itself proves causal complementary information.
Training histories log online train C-index/loss components; final metrics use
whole-train/whole-val eval predictions. Final `train_loss`/`val_loss` are the
total objective; `*_fused_cox` preserves the unweighted prediction loss.

Log: `logs/residual_teacher.log`. Historical Concat/Cross-Attention used outer
validation stopping, so their results are descriptive references only. This
run does not retrain those controls and cannot establish a strict superiority
claim over them. Full results are exploratory internal CV. All existing main
training and previous experiment outputs remain unchanged.

Verified: real ROI64 smoke with66 fit patients,8 inner stop and8 inner held
patients consumes Cox groups32+34, updates PA and CT weights, keeps all BN
buffers unchanged and reproduces predictions exactly after checkpoint reload.
The smoke uses only /tmp outputs and is never reused by the complete run.

## High-dimensional CT increment and conditional Shape experiment

```bash
bash tests/15_pathology_query_ct_tokens/run_feature_increment.sh
```

This diagnostic reuses the completed nested-risk source cohort/checkpoints,
not the old outer-validation-selected baselines. All candidates share the same
pathology risk predictions and freely calibrate their pathology slope:

- M0: one-variable pathology-risk Cox calibration.
- M1: two-variable pathology/CT-risk Cox calibration (both slopes learned).
- M2_public: pathology risk plus public-pretrained CT layer4 global512 features.
- M2_finetuned: pathology risk plus the existing fold-restricted DFS CT global512.

M2 standardizes CT features, fits PCA16/32, standardizes PC scores and fits
Breslow Cox with ridge .01/.1/1/10/100/1000 on CT terms only. The pathology
slope is unpenalized consistently in every candidate. All transforms and grid
choices use inner training data only; selection uses mean within-inner-fold
C-index. Public extraction is outcome independent. For fine-tuned features,
one inner encoder supplies both calibration-training and calibration-validation
vectors; a single final outer encoder supplies final training/validation
vectors. Features from different encoders are never pooled into one PCA fit.

For clean M2 inner selection, generate pathology OOF risks AGAIN within each
calibration-training subset (two subfolds, separate fit/stopping/OOF sets), so
calibration-validation patients do not leak indirectly into the risk feature
generator. This is30 additional small pathology fits; no CT encoder retraining.
The final head uses existing honest outer-training OOF pathology risks, with
the source outer pathology predictions shared across all M0/M1/M2 comparisons.
Record1-epoch source refits as a stability limitation instead of changing the
pathology baseline only for a new candidate.

Predeclared conditional second stage: if neither M2 source has positive mean
delta, lower paired-bootstrap bound>0 and>=4/5 positive folds versus BOTH M0
and M1, automatically run Shape and volume probes. Shape uses the original
binary NIfTI masks in physical units: voxel volume, maximum3D diameter and
sphericity. M3 adds log-volume; M4 adds all three Shape measures with inner
ridge selection. Probes predict log-volume from512-D public/fine-tuned CT
features using inner standardization/Ridge selection on all512 dimensions
(no PCA for the volume probe), reporting outer log/raw-volume R2
and MAE. Shape is annotation-derived: positive incremental value would not by
itself prove the encoder can recover identical information from CT pixels.
No intensity/texture feature search or formal teacher modification is performed.

Log: `logs/feature_increment.log`. Results: `results/feature_increment/`:
each model has fold predictions, serialized Cox heads, transform/sample-count
provenance, selected parameters, KM/HR and five-fold mean/sample-SD metrics.
M2 comparisons and `stage_decision.json` record the predeclared gate; source
pathology epochs are in `source_pathology_stability.json`. If triggered, Shape
and volume outputs appear in that same result root. Stage1 metrics are saved
before Stage2. Caches stay local. Checkpoints under `checkpoints/feature_increment/`
are only the additional PA cross-fitting models. Existing source artifacts
are read-only. The default run uses both CT sources and all five outer folds;
partial folds/one-epoch options are only for integration verification.

These remain exploratory internal-CV comparisons. Bootstrap is conditional on
fitted models and does not include retraining/previous configuration selection.
A negative PCA16/32 linear probe does not exclude all higher-dimensional or
nonlinear complementary information. See `feature_increment_plan.md`.

Verification:59 experiment15 unit tests pass, including a regression check for
double-precision SVD volume probes with rank-deficient float32 input caches.
A real ROI64 outer-fold0
integration completes six new one-epoch PA cross-fits and all four M0/M1/M2
outputs. Head reload reproduces145 validation predictions; source encoders,
512-D caches, nested split roles and training-only transformation counts are
audited. Three original masks pass physical-unit Shape extraction; a separate
synthetic integration tests M3/M4, KM/HR and the full-dimensional volume probe.
No /tmp smoke checkpoint or prediction is reused by the full experiment.
The full five-fold diagnostic is complete, including the conditional Shape
stage. See `results/feature_increment/report.md` and `review_audit.json`.
All30 serialized survival heads reproduce their saved train/validation risks
and C-index/AUC/KM-group HR statistics. Source/cohort/encoder hashes and nested
patient-role isolation were checked. Original float32 Ridge volume probes
produced ill-conditioning warnings; `shape_increment.volume_probe` now uses
float64 plus SVD. Only those10 probes were recomputed from cached features,
under `results/feature_increment/volume_probe_numeric_review/`; original outputs
and all survival predictions remain unchanged. Use that review's metrics for
volume decoding. Its maximum per-fold change in log-volume R2 is below0.00026.
Shape has exploratory incremental value relative to M0, but fails the
predeclared stability criterion versus both references. This diagnostic does
not change the formal jointly trained teacher/CT-student design.

Official API references: [Cox ridge and per-feature penalties](https://scikit-survival.readthedocs.io/en/stable/api/generated/sksurv.linear_model.CoxPHSurvivalAnalysis.html),
[mask-derived PyRadiomics Shape definitions](https://pyradiomics.readthedocs.io/en/latest/features.html#module-radiomics.shape).

## Nested risk-increment diagnostic

```bash
bash tests/15_pathology_query_ct_tokens/run_risk_increment.sh
```

This is an independently trained unimodal-risk comparison, **not** a formal
teacher or student. It consumes no existing downstream-trained checkpoint.
The objective is `z_pathology + alpha * z_CT`, with one unrestricted scalar
alpha fitted by the existing event-averaged Breslow Cox loss. Positive and
negative alpha are both allowed; no validation-based alpha grid is used.

Keep the original 721 paired patients and five outer splits. Within each outer
training set, create three event-stratified inner OOF folds. For each inner
model, split the remaining patients into fit/stop sets (85/15), train only on
fit, and early-stop only on stop. Its inner OOF patients participate in neither
training, stopping nor risk-scaler fitting. Fit each risk mean/population SD on
fit-patient eval predictions, apply to OOF predictions, and learn alpha on the
complete outer-training OOF table. Select each modality's outer training length
by the median inner best epoch (rounded down). Initialize NEW outer models,
refit all outer-training patients for those fixed epochs, fit risk scalers only
on their outer-training predictions, then evaluate the outer validation set.
There is no outer early stopping, coefficient selection or scaler fitting.

Pathology uses Top25 1280-D features, per-patch L2 and ABMIL; CT uses ROI64,
public MedicalNet ResNet18, no pooled-feature L2 and fixed BN statistics with
trainable encoder parameters. Common settings: dropout .25, Cox32 (PA forward
accumulation, CT true batch32), tail merging, Adam lr1e-4/backbone1e-5,
wd5e-4, max30 inner epochs/patience10 and seed123 reset per independent fit.
Default total: 40 model fits, followed by scalar-risk calibration/evaluation.

Outputs:
- Log: `logs/nested_risk_increment.log`.
- Results: `results/nested_risk_increment/`, including `run_config.yaml`,
  `outer_splits.csv`, `alphas.csv`, `paired_comparison.json`, and completion marker.
- `results/nested_risk_increment/fold_i/`: split roles, standardized/raw OOF
  predictions and calibration YAML containing scaler/alpha/epoch provenance.
- `results/nested_risk_increment/{pathology,ct,late_fusion}/fold_i/`: train/val
  predictions and KM plots; each model's five-fold `metrics.csv` contains
  mean/sample-SD C-index/AUC alongside per-fold HR and other survival metrics.
- `checkpoints/nested_risk_increment/fold_i/`: independent inner and outer
  modality checkpoints, training histories and checkpoint-selection metadata.

The primary comparison is fusion versus the **newly retrained pathology model
under this same nested protocol**, not against the historically selected
0.7528 checkpoint. Outer train performance is in-sample. Conditional paired
patient bootstrap does not include retraining or past configuration-selection
uncertainty; this remains internal CV, not external validation. Failure of
scalar-risk fusion would not exclude complementary higher-dimensional CT
features. Existing teacher-only diagnostic discovery excludes this run type.
The launcher refuses to overwrite existing outputs. Default is all five outer
folds; partial `--outer_folds` is for verification and never prints a five-fold
summary. The implementation plan is in `risk_increment_plan.md`.
Verification: all 50 experiment-15 unit tests pass. A real ROI64 outer-fold0
smoke with two inner folds/one epoch completes all six fresh fits, OOF alpha
calibration and survival outputs. Strict checkpoint reload, disjoint roles,
fit-only risk scalers, unchanged BN, no outer early stopping and paired-metric
parity were audited. Smoke checkpoints remain in /tmp and are never reused.
The full40-fit diagnostic has completed. New-protocol pathology Val C-index
is0.738983 and scalar fusion0.739929 (delta+0.000947; conditional bootstrap
interval[−0.001803,+0.003962]). Detailed results, alpha values, protocol limits
and audits are in [`results/nested_risk_increment/report.md`](results/nested_risk_increment/report.md).

## Matched pathology-only and uniform-CT controls

These independently initialized ablations compare against the completed original
standard Top25 cross-attention run, not against the QKNorm or CT-Z-score variants.

```bash
bash tests/15_pathology_query_ct_tokens/run_pathology_only.sh
bash tests/15_pathology_query_ct_tokens/run_uniform_ct.sh
```

- A (`_pathologyonly`): the original ABMIL patient route and the exact original
  concat risk head, with a constant zero CT slot. Construct modules in the
  original order, then remove the CT encoder and unused query/attention modules,
  preserving identical retained PA/head initialization. No CT forward, CT
  parameters or CT gradients remain. This is a pathology ablation, not a
  frozen-encoder multimodal teacher. The zero diagnostic CT risk is only an
  interface placeholder; its logged auxiliary loss is not a CT-model metric.
- B (`_uniformct`): each patient retains their own paired CT. After the same CT
  projection, compute the original internal Value projection, spatial mean and
  attention output projection: `out_proj(mean(V, dim=CT_positions))`. Broadcast
  this vector to 25 outputs and retain the original query pooling and concat
  head. Spatial attention weights are fixed to `1/512`; pathology queries do
  not select CT positions. Both active branches jointly train with fused Cox.
  The pooled identical query outputs make query-pool weighting functionally
  redundant; it is retained to match the reference computation.

Both use the same 721-patient ROI64 paired CSV and five saved folds, Top25
1280-D features, pathology patch L2, no fusion L2, no alignment/auxiliary losses,
no QKNorm, original [0,1] CT inputs, fixed BN statistics, dropout .25, Cox32
with merged tail, Adam lr1e-4/backbone lr1e-5, wd5e-4, 30 epochs/patience10 and
seed123 reset per fold. Initial shared weights match exactly; removing a route
changes subsequent RNG consumption, so identical dropout masks/patient shuffles
across epochs are not claimed. No project-trained unimodal weights are loaded.
Separate nohup logs, results and checkpoints use the two suffixes and refuse
overwrites. The default model, original runs and main project files are retained.

Compare B versus A for incremental prediction from this CT representation, and
C versus B for the benefit of learned spatial attention. Failure of both B/C
to improve A would not establish biological redundancy of the raw modalities.
Use fold-matched comparisons; this remains exploratory internal CV.

Verification: all 43 local unittest tests pass. Real ROI64 32-patient Cox
updates pass for both controls, including CT/PA/Value gradients for B, absent
Q/K score gradients, fixed BN, strict checkpoint reload and cached/full risk
parity. Both full five-fold nohup jobs started on CUDA, loaded all 721 paired
patients and completed their first epoch with saved checkpoints. Saved configs
and cohort hashes match the reference; artifacts use the two suffixes above.
Both controls have now completed all five folds. Selected-checkpoint metrics,
paired uncertainty, per-fold stability, ranking corrections and checkpoint
audits are recorded in [`fusion_controls_analysis/report.md`](fusion_controls_analysis/report.md).
The original standard-attention query/output diversity audit, including all
721 validation patients and the exact final fusion route, is recorded in
[`interaction_output_audit/report.md`](interaction_output_audit/report.md).

## Standard attention + CT Z-score

Run: `bash tests/15_pathology_query_ct_tokens/run_ctzscore_standard.sh`.
This independent nohup run uses suffix `_ctzscore`, without `_qknorm`.
It keeps the original native PyTorch single-head cross-attention:
`softmax(internal_Q @ internal_K.T / sqrt(256)) @ internal_V`, followed by
the existing output projection. There is no Q/K L2, cosine attention or
learnable score scale. This is the original cross-attention component,
not an added full Transformer encoder/decoder stack.

Relative to the completed original Top25/L2/fused-Cox teacher, only CT input
changes: per-patient whole-ROI population Z-score with denominator floor 1e-6,
inside the local model before the encoder, identically in train/eval/inference.
Raw CT NPY files stay unchanged. BN running statistics remain fixed; CT and
pathology parameters jointly train from public CT/new pathology initialization.
The same cohort, folds, seed123, ROI64, Top25, pathology patch L2, Cox32,
dropout .25, lr1e-4/backbone1e-5, wd5e-4, epochs30 and patience10 are retained.
Existing QKNorm and original standard-attention runs are preserved.

Verification: 35 local tests pass, including manual standard scaled-QK parity,
native MHA type/checkpoint keys, identical predictions to a baseline given
manually standardized CT, and CLI acceptance without `--qk_norm`. A real ROI64
GPU smoke passes fused-Cox backward, both-branch gradients, fixed-BN audit,
strict checkpoint reload and cached prediction parity. The local inference
runner reconstructs the CT transform from config; original-only score
diagnostics exclude this changed input condition.
The standard-attention Z-score run is complete. Its selected-checkpoint metrics,
per-fold stability and full-validation read-only inference controls are recorded
in [`standard_ctzscore_analysis/report.md`](standard_ctzscore_analysis/report.md).

CT adaptation implementation plan (approved independent QKNorm controls):
- [x] Test per-case whole-ROI population Z-score, constant-input safety,
  batch independence, unchanged raw input, and identical train/eval handling.
- [x] Test fixed BN versus train-updated BN, no validation updates, trainable
  branches, strict reload and cached inference for both controls.
- [x] Implement local model/config flags and distinct launchers, keeping the
  existing QKNorm baseline, CSV, raw CT files and main training code unchanged.
- [x] Verify unit tests and real ROI64 GPU train/eval/checkpoint smoke checks.
- [x] Launch both five-fold runs in separate nohup sessions; verify saved
  configurations, actual GPU PIDs, startup logs and early checkpoints.

## Independent CT adaptation controls

Both controls use the completed Top25 QKNorm teacher as reference. They retain
the same ROI64 CSV, five saved splits, public CT pretraining, fresh pathology
and fusion initialization, patch L2, fused Cox only, Cox32 with tail merging,
dropout .25, lr 1e-4/backbone lr 1e-5, wd 5e-4, 30 epochs, patience 10 and
seed 123 reset at each fold. Teacher branches remain jointly trainable.

- B: `bash tests/15_pathology_query_ct_tokens/run_ctzscore.sh` launches the
  `_qknorm_ctzscore` run. Normalize each patient's entire CT ROI using its own
  mean and population standard deviation, with denominator clamped to 1e-6.
  This happens inside the local model before the CT encoder, identically in
  training, validation and checkpoint inference. No cohort statistics, labels
  or other patients enter the transform. Constant inputs become zero. Raw CT
  NPY files retain [0,1] range and are never modified. BN statistics remain fixed.
- C: `bash tests/15_pathology_query_ct_tokens/run_ctbnadapt.sh` launches the
  `_qknorm_ctbnadapt` run. Keep original [0,1] CT inputs, set the local backbone's
  `freeze_bn_stats=False`, and use ordinary BN3d training updates (existing
  momentum .1). Retain initial public running statistics, then update them
  only in model.train(); model.eval() uses the learned running statistics
  without updating them. BN affine parameters remain trainable as before.

The CT command-line flags are mutually exclusive and require Top25,
patch L2 and fused-only loss; the B/C launchers explicitly enable QKNorm.
No combined fourth condition is launched.
Neither flag adds checkpoint parameters. The local inference runner reconstructs
both behaviors from run_config.yaml, including correctly preserving BN buffers
when evaluating C. The main dataset loader and all other training files are
unchanged. Cox32 is a risk-set size; CT forwards remain one patient at a time,
so it does not make BN's patient batch size 32.

Verification: 33 local tests pass. Both controls pass real ROI64 GPU smoke
checks through the main trainer/evaluator, nonzero finite CT/pathology/QKV/scale
gradients, strict checkpoint reload and cached-inference parity. B had zero
BN-stat updates; C had exactly two for the two training forwards, and none
during validation/evaluation/cache inference. Temporary smoke outputs were
removed. Launchers use separate nohup logs/checkpoints/results and refuse to
overwrite existing runs.

Both independent runs were launched concurrently on GPU0 on 2026-11-11:
B PID 601244 and C PID 601247. Saved configurations, 721-patient cohort loading,
first-epoch logs and fold_0 checkpoints were verified. B preserves the public
BN running buffers; C checkpoints contain training-updated BN statistics.
Both five-fold runs are complete. Their selected-checkpoint metrics, training
stability, exact BN-update audit and read-only validation interventions are
summarized in [`ct_adaptation_analysis/report.md`](ct_adaptation_analysis/report.md).

QKNorm implementation plan (approved Top25 control):
- [x] Test the internal Q/K cosine formula, unchanged projected Value/output,
  scale invariance, zero-input safety, trainable scale and fused-Cox gradients.
- [x] Add optional local single-head QKNorm attention, retaining existing
  Q/K/V/output parameter names and default model behavior. Initialize the
  learnable scalar with log2(512^2-512); remove sqrt(d) scaling in this mode.
- [x] Connect run configuration, distinct run tag/launcher and checkpoint
  inference; exclude QKNorm runs from original-only score diagnosis.
- [x] Verify all local tests and a real-patient GPU training/evaluation smoke.
- [x] Launch the separate five-fold Top25/L2/fused-Cox run in the background;
  verify its process, GPU, configuration and loaded cohort; preserve older runs.

## QKNorm Top25 control

Run: `bash tests/15_pathology_query_ct_tokens/run_qknorm.sh`. The launcher uses
nohup and saves a separate run with suffix `_qknorm` under logs/, results/ and
checkpoints/. It uses the original 25 pathology queries, not all-query bags.

Only the attention score changes. After the internal Q/K/V linear projections,
normalize Q and K per token along the 256-D head dimension. Compute
`softmax(scale * normalize(Q) @ normalize(K).T) @ V`, then retain the original
output projection, query pooling and concat head. V is not normalized; no
sqrt(256) divisor remains in this mode. Attention dropout remains zero.
`scale` is a trainable scalar initialized to `log2(512^2-512)=17.997179`, adapting
the QKNorm paper's sequence-length initialization to the fixed 512 CT positions.
Reference: https://aclanthology.org/2020.findings-emnlp.379/ (Eqs. 3 and 5).
This uses its scoring mechanism, not its full translation architecture.

The original MHA projection initialization and checkpoint names remain intact;
QKNorm adds only `fusion.coattn.scale`. Original runs still use PyTorch's
unmodified scaled dot-product MHA. QKNorm checkpoints must be instantiated with
`qk_norm=True`; the local inference-check runner reads this from run_config.yaml.
The original-only score diagnostic explicitly excludes this different scoring
mode rather than applying its scaled-dot-product formulas to it.

All other settings follow the Top25 L2/fused-Cox baseline: same CSV/folds,
patch L2, no fusion-input L2, ROI64, Cox32 with tail merging, dropout .25,
lr 1e-4/backbone lr 1e-5, Adam wd 5e-4 (including the new scalar), 30 epochs,
patience 10 and seed 123. Public CT pretraining and newly initialized trainable
pathology/fusion modules are used for each fold. No downstream unimodal weights,
alignment loss or auxiliary survival loss are introduced.

Verification: all 26 local tests pass. A temporary real ROI64 four-patient GPU
smoke passed the main trainer/evaluator, nonzero finite QKV/scale/CT/pathology
gradients, a scale update, strict checkpoint reload and cached-inference parity.
It does not establish a performance improvement; that requires the new CV run
and subsequent query-shuffle/uniform-attention comparisons.

The five-fold GPU run was launched on 2026-11-11 (PID 449682), with configuration,
721 paired patients, the first training epochs and fold_0 checkpoint verified.
All five folds are now complete. Selected-checkpoint results and fixed-model
validation inference (entropy, uniform attention, 100 query/CT derangements per
fold) are recorded in [`qknorm_analysis/report.md`](qknorm_analysis/report.md).

All-patch Query implementation checklist (approved scope):
- [x] Verify every cohort slide's full features/coordinates against its Top25
  selected_indices, preserving labels, row order and saved CV assignments.
- [x] Put the exact Top25 first, followed by all remaining source rows once.
  Pool only the first 25 for the direct pathology patient feature; use the full
  compressed sequence for queries. Share trainable modules, without freezing.
- [x] Retain per-patient forwards and existing Cox32, evaluation and output
  interfaces; adapt cached inference for variable-length query sequences.
- [x] Add a separate all-query/fused-Cox launcher with pathology patch L2 on,
  other settings unchanged; no new alignment or auxiliary loss.
- [x] Verify tests, all 721 input mappings and real-case train/eval; do not
  launch training or push this change automatically.

## All-patch Query control

Run: `bash tests/15_pathology_query_ct_tokens/run_all_queries.sh` (nohup
background). This is a separate run, identified by `paqall`. Original Top25,
raw-feature and PAMT-alignment runs are preserved. The implementation has been
verified; the user has now completed all five folds. Results and subsequent
fixed-checkpoint inference checks are summarized in
[`all_query_analysis/report.md`](all_query_analysis/report.md).

Input CSV/folds remain `data/eagle/top25/all_label_roi64.csv`. Full features
come from `/home/gly001/cqj/data/lung_cancer/pathology/virchow/virchow2/`.
These are all available Virchow2-CLS patches from the same STAMP grid as the
original EAGLE Top25, not UNI or a different TRIDENT patch grid. They include
all saved valid patches, not every possible pixel region of the WSI.

`AllPatchQueryDataset` verifies each original Top25's `selected_indices`,
features, coordinates and source metadata against the full H5 before training.
It orders the exact Top25 first, in their original ranking order, followed by
every remaining source row once. CSV labels, patient pairing and fold
assignments stay unchanged; no separate label/split CSV is generated.

The full input [1,N,1280] is L2-normalized per patch, then compressed by the
shared 1280→512 projection. Only its first 25 rows enter the original ABMIL
attention/patient pathway (512→256). All N compressed rows become 256-D
Queries. CT layer4 still provides 512 spatial Key/Value tokens of 256 dimensions.
Cross-attention produces [1,N,256], attention-pools over N to 256, then concatenates
the direct Top25 pathology patient feature and predicts fused risk using the
existing 512→256→128→1 head. Attention outputs have shapes [1,N,512] and [1,N,1].
There is no padding or truncation; patients are forwarded separately and the
existing Cox32 groups merge any tail. Loss remains fused Cox only, with no
alignment or auxiliary branch loss. Patch L2, public MedicalNet initialization,
joint trainable branches, dropout, optimizer and stopping settings match the
original L2/no-alignment control. No project-trained unimodal checkpoint is used.

The run saves `query_data_audit.yaml` alongside `run_config.yaml` and the
existing per-fold outputs. Cached CT/query permutation inference also supports
variable N, retaining the recipient's direct Top25 patient feature.

Verification on 2026-11-11:
- All 721 paired patients pass exact Top25 feature/coordinate correspondence;
  271,635 query patches total, min 67, median 375, mean 376.7476, max 755.
  Dataframe and five saved CV splits match the original loader exactly.
- All 22 experiment unit tests pass, including gradient propagation to extra
  queries and both branches, original behavior for N=25, and variable-length
  cached inference parity.
- Four real ROI64 patients (two training, two validation; N=449/436/469/419)
  pass a temporary one-update GPU smoke check through the main trainer and
  evaluator, strict checkpoint reload, direct Top25 branch parity, 512 CT keys,
  cached prediction parity and CT/query swaps. Temporary outputs were removed.
  This smoke check verifies execution, not predictive performance or full
  Cox32 memory usage.

## Original Top25 architecture

Architecture for ROI64:
- EAGLE Top25 Virchow2 CLS [B,25,1280] → per-patch L2 → existing ABMIL
  compression [B,25,512]. The existing ABMIL path still produces a 256-D patient
  feature for the final concat and an auxiliary risk for interface compatibility.
- Map the 25 compressed pathology patches to 256-D Queries.
- MedicalNet-style ResNet18 layer4 produces [B,512,8,8,8]. Flatten spatial axes,
  not channels: [B,512 positions,512 channels]. Map each CT position to 256-D
  Keys/Values using one shared projection. No spatial average pooling before KV.
- Single-head cross-attention: Q=pathology, K=V=CT. Output [B,25,256]; the
  attention probability matrix is [B,25,512], softmax over the CT positions.
- Attention-pool the 25 CT retrieval outputs: Linear(256,128), Tanh,
  Dropout(.25), Linear(128,1), softmax over the 25 query outputs → weighted sum
  → 256-D pathology-guided CT feature. This matches the ordinary ABMIL pooling
  form; it is not a reproduction of MCAT's Transformer plus gated pooling.
  Concat with the ABMIL pathology patient feature → existing 512→256→128→1
  head (LayerNorm/ReLU/Dropout at its hidden layers).

Local forward tuple: (fused risk, CT diagnostic risk, PA diagnostic risk,
query-pooling weights [B,25,1], guided CT [B,256], PA patient [B,256],
cross-attention weights [B,25,512]). CT spatial positions and query-output
importance are distinct attention axes, each normalized by its own softmax.

No post-projection/feature L2, Transformer, positional encoding, residual query
addition, or extra self-attention. CT spatial token order is fixed but attention
does not explicitly model 3D adjacency. Tokens are overlapping CNN receptive
fields, not independent CT crops. Raw attention can be recovered by reshaping
its CT axis back to 8×8×8; do not treat this as the old single-branch WSI heatmap.

Training matches the latest v7 teacher controls: same paired CSV/five folds,
721 patients, per-patient forwards, joint Cox32 with tail merging,
lambda_ct=lambda_pa=0, Adam backbone lr=1e-5/other lr=1e-4, wd=5e-4,
dropout=.25, 30 epochs, patience=10, seed=123 reset each fold. CT backbone
fine-tunes with fixed BN running statistics. All classes/runner/output changes
are local to this experiment; original main-model fusions are unchanged.

Implementation and verification plan:
- [x] Test token shapes/direction, attention probability rows, invariance to CT
  token ordering without positional encoding, query sensitivity, model branch
  parity, single CT traversal, joint gradients, optimizer groups and strict load.
- [x] Implement local fusion/model and reuse train_pact/build_pact_optimizer,
  evaluate_survival, saved CV assignments and save_cv_results.
- [x] Run tests and a real-patient CPU forward/backward smoke with 512 CT tokens.
- [x] Launch a five-fold GPU run via run.sh; verify process/config/startup logs.

Run: `bash tests/15_pathology_query_ct_tokens/run.sh` (nohup background).
Outputs: logs/, checkpoints/, results/ beneath this directory. Existing runs
are never overwritten. Final evaluation exports train/val predictions, AUC36,
AUC60, HR and KM using the main evaluator. Checkpoints require the local
PathQueryCTModel class; generic main-project visualization/student loaders do
not construct this experimental model automatically.

## PAMT alignment extension (within experiment 15)

The completed baseline uses fused Cox only. Its saved results/checkpoints are
retained. The current run.sh launches a separate run with suffix
`_pamtalignk2_w1`: fused Cox + 1 × PAMT alignment. No CT/PA auxiliary survival
loss is added; the architecture, cohort, split, optimizer and early stopping
remain those of experiment 15.

Alignment follows the enabled contrastive branch of PAMT's main entry:
`gene_wsi_predict/train.py`, `utils_cox.py` and
`vit_model_gene_wsi_concat_label.py`. This adapts its gene/WSI alignment to
pathology/CT tokens; it does not reproduce PAMT's entire architecture.

Before cross-attention's internal Q/K/V projections, use projected pathology
P=[B,25,256] and CT C=[B,512,256]. Compute raw scores S=P@C.transpose(-2,-1).
For each pathology query, take log-softmax over all CT positions and sum the
negative log probabilities of its top 2 positions, then average over queries
and patients. This is numerically and gradient-equivalent to PAMT's sorted
scores plus CrossEntropyLoss with two target entries set to 1. Keep target
mass 2; do not divide by 2. No additional L2, temperature or sqrt(dim) scaling
is used. Uniform scores over 512 positions give alignment loss 2*log(512)
(approximately 12.47665).

These are within-patient, similarity-selected pseudo-positive positions, not
anatomically annotated matches or cross-patient contrastive pairs. The loss
can encourage concentrated matching; improvement must be checked in results.

The optional eighth forward output is alignment loss; checkpoint parameter
keys and risk predictions for the same weights are unchanged. Alignment is
averaged over patients in each joint Cox group before adding it to Cox. Logs
include training total/fused Cox/alignment and validation fused Cox/alignment/
total. `Val Loss` and final metrics losses remain fused Cox for comparison
with the baseline. Best checkpoint and early stopping still use Val C-index.

Verification: test_pamt_alignment.py checks equality of source-form loss and
gradients, target mass, patient averaging, CT token permutation invariance,
and checkpoint/prediction compatibility. test_path_query_ct.py retains the
baseline architecture and gradient checks. New full training is not started
automatically by editing these files.

The alignment extension also passed a real four-patient CPU smoke: MedicalNet
initialization, Cox plus alignment backward/update, best-checkpoint strict
reload, metadata and finite final train/validation evaluation. Temporary smoke
outputs were removed. Both test files passed (six tests total); run.sh syntax
and the modified Python modules compiled successfully. These checks establish
implementation compatibility, not predictive performance of the new loss.

## Fixed-checkpoint inference checks

Approved design: evaluate the completed baseline and PAMT-aligned runs on each
fold's own validation patients. Repeat 100 within-fold patient derangements
(no self-donors), using the same donor orders for the two interventions and
both model variants. Labels always stay attached to the recipient patient.

- CT shuffle: replace only CT spatial tokens with another patient's tokens.
- Query-only shuffle: replace only the pathology patch features feeding the
  query projection. Keep the recipient's CT tokens and direct pathology
  patient feature unchanged. This does not shuffle labels or just token order.
- Cache deterministic fusion inputs in memory after eval-mode encoding; call
  the existing fusion module for every intervention. Verify both full-model
  and cached baseline risks against saved validation predictions first.
- Export repeat-level C-index, IPCW AUC36/AUC60, prediction changes, donor
  indices and risks. Average repeats within each fold, then report mean and
  sample SD across five folds; permutation variation is not a patient CI.
- Save under inference_checks/, with one log under logs/. No training,
  checkpoint changes or overwriting the completed training outputs.

Implementation/verification plan:
- [x] Test reproducible derangements, unchanged direct pathology input,
  correct CT/query substitution, cached/full-forward parity and fold-first SD.
- [x] Implement inference_checks.py and its nohup launcher.
- [x] Run unit tests and a real ROI64/Top25 checkpoint smoke on the training GPU.
- [x] Start the requested background inference and verify progress in its log.

Four inference tests passed, together with the six existing architecture/loss
tests. Both variants passed a real four-patient ROI64/Top25 GPU check: full
predictions matched saved predictions within 4e-8, and cached fusion differed
from full inference by at most 1.2e-7. Checkpoint SHA256 hashes were unchanged.
CPU and GPU convolutions showed larger numerical differences in the smoke;
the production checks therefore run on the original GPU settings and enforce
baseline parity before computing shuffle effects.

Run: `bash tests/15_pathology_query_ct_tokens/run_inference_checks.sh`.
Log: `logs/inference_checks.log`. Outputs: `inference_checks/<training_run>/`,
with fold-specific donor indices/predictions and repeat/fold/summary CSVs.
`inference_checks/comparison.csv` and `completed.json` are written after both
model variants finish. The launcher refuses to overwrite an existing check.

## Query-stage diagnostic

Additional read-only analysis is saved in query_stage_diagnostic/. Both
variants were evaluated on all five validation folds (721 patients), with ten
query-only patient derangements per fold, seed 123. Queries, actual attention,
retrieved CT token outputs, attention-pooled CT features and final risks were
compared while keeping recipient CT and direct pathology features fixed.
Saved checkpoints were unchanged and baseline risk parity was verified.

For each fold, feature/risk RMSE is normalized by the corresponding baseline
between-patient RMS spread; these are different denominators and must not be
interpreted as a percentage of information retained between stages.

Without alignment, mean normalized attention entropy was 0.9999994; query
replacement changed guided CT features by only 0.00238 times their baseline
between-patient spread. With PAMT, the corresponding values were 0.97333 and
0.24553, but final risk changes remained 0.01448 times baseline risk SD and
the completed survival evaluation did not improve. This localizes weak query
sensitivity upstream of the prediction head in the no-alignment model; it
does not establish the biological amount of complementary CT information or
prove that a particular training modification will improve generalization.

## Original attention score diagnosis

`diagnose_scores.py` and `test_score_diagnosis.py` audit the no-alignment
variant only. Results and interpretation are in score_diagnosis/report.md.
The analysis verifies PyTorch/manual output parity, feature scales/spatial
variation, optimizer inclusion, actual training-mode Cox gradients, a
zero-data-gradient Adam decay control on independent parameter copies, and
uniform-value aggregation. It does not step an optimizer on trained models.
All trained parameter/buffer values and checkpoint hashes were unchanged.

## No pathology patch L2 control

`bash tests/15_pathology_query_ct_tokens/run_no_l2.sh` starts the requested
five-fold raw-input control in the background. It passes
`--no_pa_l2_normalize --alignment_weight 0` and uses a separate `paraw` tag.
Only input patch normalization changes relative to the original fused-only
baseline: raw patches feed the same ABMIL compression, whose outputs supply
both the pathology patient feature and the cross-attention queries.

The model keeps L2 enabled by default for old runs/checkpoints; inference
loading reads pa_l2_normalize from the run config. The no-L2 control uses
ROI64, Cox32/tail merging, dropout .25, 30 epochs, patience 10, lr 1e-4,
CT backbone lr 1e-5, wd 5e-4 and seed 123. Public MedicalNet weights initialize
the backbone; all remaining modules are newly initialized. No project-trained
unimodal weights, alignment or auxiliary survival losses are used.

Verification: three unit tests passed, including both attention axes and pooling gradients. Real ROI64/Top25 inputs produced exactly [1,25,512] cross-attention and [1,25,1] pooling probabilities. A four-patient CPU smoke loaded MedicalNet, completed fused-only Cox backward/update, restored the best checkpoint strictly, and evaluated finite train/validation metrics. Temporary smoke outputs were removed.
