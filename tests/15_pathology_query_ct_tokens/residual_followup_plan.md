# Residual teacher follow-up: epoch audit and matched pathology controls

Scope: execute the two checks requested in the pasted review, within test15.
Do not add fusion modules, force a minimum epoch after seeing outer results,
initialize teachers from trained unimodal weights, or overwrite source runs.

1. Audit every saved inner history, especially outer folds0/3. Reproduce raw
   best epochs, median refit epochs, first/best/last stop C-index, PA/fused Cox
   and mean-square CT correction trajectories. Check stop event counts and
   number of comparable pairs. Recompute within-teacher pair totals. Compare
   raw epoch selection with one predeclared retrospective alternative: maximum
   trailing3-epoch mean stop C-index (complete windows only). Report candidate
   epochs only: later checkpoint weights/predictions do not exist, so this
   cannot establish an improvement or justify changing completed results.

2. Add a matched PA-only control: exact existing ABMIL1280->512->256->1,
   Top25/patch L2/dropout.25. Construct the same initial ResidualTeacher so PA
   weights AND post-initialization RNG state match, then remove CT modules.
   There is no CT forward, frozen encoder, or source trained weight loading.
   Optimize Cox(rp)+0.3*Cox(rp)=1.3*Cox(rp), matching the residual objective
   when CT is absent (important with Adam's coupled weight decay).

   Reuse the original source's same721 patients, saved outer splits, exact
   inner fit/stop/held roles, LR/wd/seed and Cox32+tail. Train three inner PA
   fits per outer fold, independently choose best stop C-index with patience10,
   and fresh-refit outer training for median selected epochs (protocol control).
   Also fresh-refit for the previously INNER-selected teacher's saved epoch
   count (budget control); if equal, reuse the exact same PA refit. These two
   answer different questions and neither is selected by outer validation.
   Pre-save both epoch schedules before any outer control prediction.

Files: diagnose_residual_epochs.py; residual_pathology_control.py;
test_residual_followup.py; run_residual_pathology_control.sh. Extend only the
local fit_teacher helper with a default-preserving model_factory argument and
accurate initialization metadata. Main training remains unchanged.

Outputs: results/residual_followup/epoch_diagnosis (read-only source analysis),
results/residual_pathology_control/{protocol_matched,teacher_epoch_matched}
and separate checkpoints/log. Per-fold predictions/KM/AUC/HR, paired patient
Bootstrap and corrected/broken rankings versus the fixed INNER-selected teacher.
Do not treat comparisons as proof of a causal CT effect or generalize away
finite-sample/optimization uncertainty. Conditional Bootstrap excludes retraining.

Checklist:
- [x] Failing tests: control initialization/RNG parity and removal of CT,
  first-update PA gradient parity, rolling-window no incomplete windows,
  epoch selection without outer scores and existing residual compatibility.
- [x] Implement and run epoch audit; report numerical evidence without changing
  original checkpoint selection or interpreting mean-square correction as SD.
- [x] Implement paired controls, source hashes/roles/prediction checks and
  default-preserving local trainer factory. Verify all test15 tests plus
  real-data checkpoint/prediction and parameter-update smoke checks.
- [x] Start the authorized PA-only follow-up with nohup+setsid, verify live
  GPU/PID/first epoch; retain the exact source teacher/PA outputs unchanged.

Epoch audit is complete: original median rules reproduced; retrospective rolling3
primary candidate epochs3/27/10/3/22 do not establish a performance improvement.
All71 test15 tests passed. Independent read-only review found no blocking or
important issues. Real public-initialization/RNG, no-CT and checkpoint smoke
passed. Complete PA-only controls started with nohup+setsid: PID951823, PPID1,
CUDA active, first inner epoch/checkpoint confirmed. Control results are pending.
