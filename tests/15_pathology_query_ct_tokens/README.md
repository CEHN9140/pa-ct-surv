# Experiment 15: pathology queries CT layer4 spatial tokens

Repository scope: code, launchers, tests, logs, aggregate metrics, diagnostic
reports and KM plots are versioned. Checkpoints, Python caches, patient-level
prediction tables and permutation/patient-risk NPZ files remain local.

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
