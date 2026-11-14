# CT-only auxiliary Shape supervision: matched A/B/C diagnostic

Scope: A=Cox baseline, B=Cox+sphericity MSE, C=Cox+all3 Shape MSE. Only CT-only
diagnostics in test15; no teacher/student or main-flow changes. B/C initialize
freshly from PUBLIC MedicalNet plus the original new risk head, never from a
project-trained CT/PA checkpoint. Read-only A evaluation/caches are controls.

Keep721 cases/251 DFS events/source outer5 and inner FIT/STOP/HELD roles. CT is
existing64^3 ROI,[0,1],no augmentation/L2; MedicalNet ResNet18 layer4 GAP512.
Risk head unchanged512->256->ReLU->Dropout.25->1. BN running stats fixed,affine
trainable. Adam backbone1e-5,risk/Shape heads1e-4,wd5e-4;true batch32 with
tail merged,no dropping;seed123 reset perfit,max30/STOP patience10.

Add Linear512->K (K=1 sphere or3 log-volume/diameter/sphere), no dropout. Use
fork_rng during auxiliary-head construction so base initialization AND subsequent
shuffle/dropout RNG equal A. Risk forward comes before auxiliary projection.
The B sphere row is copied from the same canonical3-row auxiliary initialization
used in C,so the shared target starts from identical auxiliary weights too.
Standardize Shape labels using ONLY neural FIT patients (outer refit:whole
outer train),population mean/std; persist buffers with checkpoint. MSE averages
over patients and K. Train total=Cox+lambda*MSE;lambda0 returns exactly Cox,
without a zero-weight auxiliary graph or auxiliary gradients/weight decay.
Aux head belongs to headLR group,not backboneLR. All network parameters train.

Predeclare lambda grid[0,.05,.2]. Reuse verified A's15 inner+5 outer controls
for lambda0 after exact model/optimizer/short-training parity checks; no zero
candidate is warm-started. CPU parity is exact;CUDA parameters allow1e-8
roundoff while requiring identical risks/STOP epoch. The original source itself
showed9.3e-10 parameter variation on repeat,with identical predictions.
For each nonzero lambda,fit all3 inner networks from
public weights: FIT updates,STOP DFS C-index selects epoch,HELD C-index selects
lambda by mean across3. Exact ties prefer smaller lambda. Fixed outer epochs
are floor median selected STOP epochs of the chosen lambda. Write choices
before outer evaluation. Chosenlambda0 reuses A outputs/control features only;
otherwise fresh public outer refit on all outertrain. Never stop/tune on outer.
Each B/C branch entails30 inner fits plus up to5 outer refits; run in parallel
after GPU memory/update tests. Each owns results/checkpoints/log, no launcher log.

Epoch records: train/STOP total loss,Cox,Shape MSE,C-index (train online,STOP
eval);final whole-train/val eval Cox NLL,C-index,AUC36/60,KM/HR,aux-head target
R2/MAE. Keep source A mean/SD and patient-paired2000 Bootstrap deltas. Aux-head
selection is never based on Shape recovery or outer survival results.

Posttraining diagnostic: extract selected inner/outer512-D features in eval with
BN buffers unchanged. For A use existing validated feature_increment caches.
For B/C lambda0 use those A caches; otherwise load the completed auxiliary
checkpoint ONLY for inference,not initialization. Inner Probe tuning uses each
appropriate inner encoder's single feature basis; final Probe uses its outer
encoder. Do not concatenate differently trained feature axes.

Raw linear Ridge Probe targets all3 Shape;target-wise alpha from fixed6-point
grid by mean inner MSE. Conditional sphere Probe: fit unpenalized linear
nuisance g(trueV,trueD)->trueS on each Probe training part only, then predict
S-g(V,D) from512-D CT. Apply same nuisance to held labels; never refit on held.
Report residual R2/MAE plus oracle-VD baseline and reconstructed S error. This
is a conditional recovery diagnostic,not a mask-free deployed predictor.
All Probe alpha/nuisance fits stay in outertrain. Lambda is another hyperparameter
selected from those same inner folds;inner Probe scores are tuning diagnostics,
not independent estimates. Only outer performance is the reported validation.
Residualization removes only a LINEAR VD relationship,not all dependence. Raw
residual R2 and reconstructed-S R2 have different denominators. A/lambda0 has no
trained auxiliary head:direct-head errors are N/A. C averages3 target errors,
so its per-target coefficient is lambda/3;B/C is not a causal target-addition test.

Targets derive from masks/current geometry. Existing CT ROI localization also
requires masks. Upstream intensity uncertainty remains. Target choice uses prior
exploration on this cohort;new nested CV does not erase that historical bias.
No automatic teacher extension follows a positive CT-only result.

Files: ct_shape_aux_model.py (network,loss,optimizer,selection);ct_shape_aux_train.py
(audited source I/O,training,nested orchestration);ct_shape_aux_probe.py
(cache/readouts/report);test_ct_shape_aux.py;run_ct_shape_aux.sh. Use isolated
.phenotype_runtime, no UNI modification. Save actual scaler/training-role/selection
provenance and source hashes;fail if caches differ. Do not overwrite old results.

- [x] RED tests: init/RNG/risk parity;lambda0 optimizer/update parity;auxiliary
      gradients/BN;train-only population scaler/reload;headLR;innerlambda/epoch
      selection;train-only conditional nuisance and residual recovery.
- [x] Implement model,trainer,nested selection,raw/conditional Probes/reports.
- [x] Pass focused/full tests,public-init/GPU true32+tail smoke,source checkpoint
      inference parity;independent review;check memory for B/C parallel launch.
- [x] Start B/C background jobs with dedicated logs;verify active CUDA epochs.
- [x] Report launch/status and paths accurately;full runs are not yet results.

Prelaunch evidence:103 tests passed in77.294s. Source audit validates721 Shape
labels,15 A inner candidates and20 actual checkpoint-backed feature caches.
Real GPU smoke (temporary artifacts only):two-epoch source/aux0 risks and STOP
choice identical;max CT parameter difference7.28e-12,less than source's own
9.31e-10 repeat noise. Both B/C66-case nonzero fits used32+34 risk sets,updated
stem andaux head,kept BN buffers unchanged,reloaded exactly. Source A32-patient
checkpoint predictions match old CSV. Peak allocated~2044MiB.

The A-vs-identical-aux readout smoke reproduces all5-fold rawShape probes:
R² sphere0.28944,linear-VD-residual sphere0.00165;both copies match exactly.
One-target Ridge predict returns1D in this runtime:fixed shape handling in
select_probe and residual evaluation;end-to-end single-target regression test
guards against unintended[N,N] broadcasting. Existing multi-target tests pass.

Launch verified: B PID1450276,C PID1450277,each PPID1 and CUDA memory~4744MiB.
Both completed their first inner fit and advanced to inner1;dedicated logs have
train/STOP total,Cox,Shape MSE and C-index. No teacher/student process started.
Existing unrelated process141305 was left untouched. Full nested experiments
are RUNNING,not complete;only temporary validation/probe controls have finished.
