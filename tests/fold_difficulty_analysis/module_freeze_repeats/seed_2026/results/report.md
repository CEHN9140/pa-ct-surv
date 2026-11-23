# Epoch3 branch-freezing intervention

Raw UNI full bags, original721/251/fivefold. Each scope starts newly; common3Epoch warmup, exact last warmup state and RNG reused across two branches, Adam reset. Frozen modules remain train-mode including Dropout; Freeze-C gradient flows through frozen head to AB. No input/loss/architecture change.

Single event-stratified80/20 FIT/STOP inside each Outer Train, seed123. Each inner arm runs20continuation Epochs without early stopping; STOP selects continuation0..20, tie chooses earlier. Outer Train separately warms up3new Epochs, branches and runs the fixed selected budget. Outer Val is final-only; reported Train C/Loss is eval-mode at same final checkpoint.

| Arm | Train C | Val C | Gap | AUC36 | AUC60 |
|---|---:|---:|---:|---:|---:|
| baseline | 0.8348 ± 0.0702 | 0.7322 ± 0.0327 | 0.1027 ± 0.0976 | 0.7665 ± 0.0059 | 0.7552 ± 0.0337 |
| freeze_c | 0.8352 ± 0.0643 | 0.7361 ± 0.0336 | 0.0991 ± 0.0950 | 0.7705 ± 0.0086 | 0.7590 ± 0.0371 |

Historical0.7248/0.7458 are not matched comparators: this baseline resets Adam after3Epochs and uses a different fixed continuation window. Freeze effects are not a unique causal decomposition of modules; changing trainability changes optimization and interactions. Frozen Dropout remains stochastic in training. Reduced gap alone is not improvement. Conditional1000paired patient intervals omit refitting, selection, overlapping Train folds, one comparison and prior exploration. Internal exploratory validation only.

All branch initial weight/RNG hashes and patient-order prefixes, frozen tensors, candidate selection and20final Train/Val C-indices were verified. No project-trained unimodal checkpoint initializes these models. All private predictions/weights remain local. No teacher/student change or automatic follow-up experiment follows.

Training seed=2026; inner split seed remains123. Only baseline/freeze_c are run.
