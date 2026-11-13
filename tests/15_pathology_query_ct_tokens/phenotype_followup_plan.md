# Shape plus First-order and MedicalNet phenotype probes

Goal: implement the two requested diagnostics, not the later predicted-phenotype
Cox experiment. Keep the proposed jointly trained teacher -> CT student method
and all main code unchanged. Reuse721 patients/251 events/original outer folds.
User's supplied design authorizes inline implementation and background execution.

1. Conditional prognosis: fit PA+Shape+FirstOrder (9 phenotype covariates) with
   the existing IncrementCox and honest nested PA risks. Standardize within
   training, freely calibrate PA slope, penalize all9 CT terms with one ridge
   chosen from the unchanged .01/.1/1/10/100/1000 grid by inner C-index. Main
   comparison PA+Shape, secondary PA alone. Retain all existing models/results.
   Save five-fold predictions, NLL/AUC/KM/HR, paired2000-patient bootstrap,
   mean/sampleSD, per-fold changes and selected parameters.
2. Linear phenotype probes: cached layer4 GAP512 before risk head, no L2, eval.
   Compare public-pretrained versus fold-restricted DFS-finetuned representations.
   Targets: log native volume, maximum3D diameter, sphericity; tumor mean, STD,
   P10, median, P90, entropy. Log-volume is the original Shape predictor, not
   raw volume; explicitly label its units. Fit StandardScaler+linear Ridge,
   float64/SVD, intercept; select each target's ridge by mean inner MSE,
   exact ties favor stronger ridge. No DFS labels enter probe fitting/scoring.
   Outer final probe uses outer model's features for train and val. Inner
   tuning uses that inner model's features for train and held patients. Never
   concatenate features from separately trained encoders as a single basis.
   Preserve source CT inner FIT/STOP/HELD roles and OOF membership checks.
3. Provenance: strictly cache-only, no fallback extraction/training. Check NPZ
   IDs,512-D finite values, source public/finetuned metadata, actual checkpoint
   hashes, existing cohort/features/config hashes, five outer folds and15 inner
   partitions. Check at end that all source files remain unchanged. All fitted
   scalers/heads are local diagnostics, not teacher/student initialization.
4. Report each target's five-fold R2/MAE/RMSE and train-mean baseline errors,
   plus pooled outer-prediction R2 as a separate descriptive statistic. Per-fold
   predictions stay ignored locally. Negative R2 is valid; low linear R2 does
   not prove absence of nonlinear information. Even high R2 does not prove
   that recovered information improves DFS. Stage3 requires a future strictly
   cross-fitted implementation; do not run it automatically.

Files: phenotype_probe.py (fit/select/evaluate linear probes);
phenotype_followup.py (cache audit and requested diagnostics);
test_phenotype_followup.py; run_phenotype_followup.sh. Independent CPU-only
results/phenotype_followup and logs/phenotype_followup.log. Use existing
.phenotype_runtime with NumPy1.26.4; do not modify UNI or main data.

Verification and execution:
- [x] Write tests for probe scaling, target-wise selection, train-mean baseline,
      held-patient isolation, corrupt/missing cache failures and target columns;
      run RED before implementation.
- [x] Implement cache audit, prognosis follow-up and probe evaluation. Source
      M0/Shape are read-only references, with explicit patient/label checks.
- [x] Pass focused and complete experiment15 tests; test real cached probes
      and Shape reference parity. Review all leakage boundaries.
- [x] Launch only these diagnostics via nohup/setsid, verify host progress.
- [x] Verify completion, all9x2x5 target metrics, saved probes/predictions,
      source hashes, all5 prognosis folds and paired comparisons; report results.

Interpretation: this follow-up is exploratory after multiple earlier analyses.
Bootstrap intervals condition on existing fitted models and do not include
retraining/prior exploration uncertainty. Phenotype targets depend on masks,
current ROI64 and upstream-uncertain clipped input intensities. Probe inference
uses CT-derived512-D vectors only; target masks are not predictor inputs.

Pre-run evidence:5 focused tests and all90 test15 tests passed (75.371s).
Actual load_inputs verified21 cache sets,15 CT/PA role partitions and128 source
files/checkpoints. Fold0 public and finetuned9-target probes, including full
inner tuning, completed with finite metrics in1.91s. No source writes.

Completion: all5 prognosis folds and90 probe target/fold metrics verified.
All10 saved probe pipelines and5 Cox heads reproduce predictions; train-only
scaler means,90 target-wise ridge choices,5 Cox choices and all145 source hashes
passed checks. The independent reviewer found no important/critical issues;
its suggestion to include original OOF/late-fusion/summary sources in the hash
manifest was applied before execution. See completion_review.json.
PA+Shape+FirstOrder Val0.7527+/-0.0387; versus Shape Δ+0.0007,95% conditional
paired interval[-0.0063,+0.0079],2/5 positive folds. Stage3 was not started.
