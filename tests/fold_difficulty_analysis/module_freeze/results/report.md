# Epoch3 branch-freezing intervention

Raw UNI full bags, original721/251/fivefold. Each scope starts newly; common3Epoch warmup, exact last warmup state and RNG reused across three branches, Adam reset. Frozen modules remain train-mode including Dropout; Freeze-C gradient flows through frozen head to AB. No input/loss/architecture change.

Single event-stratified80/20 FIT/STOP inside each Outer Train, seed123. Each inner arm runs20continuation Epochs without early stopping; STOP selects continuation0..20, tie chooses earlier. Outer Train separately warms up3new Epochs, branches and runs the fixed selected budget. Outer Val is final-only; reported Train C/Loss is eval-mode at same final checkpoint.

| Arm | Train C | Val C | Gap | AUC36 | AUC60 |
|---|---:|---:|---:|---:|---:|
| baseline | 0.8502 ± 0.0600 | 0.7310 ± 0.0390 | 0.1193 ± 0.0936 | 0.7699 ± 0.0142 | 0.7481 ± 0.0375 |
| freeze_ab | 0.7826 ± 0.0083 | 0.7261 ± 0.0535 | 0.0564 ± 0.0590 | 0.7651 ± 0.0314 | 0.7329 ± 0.0652 |
| freeze_c | 0.8420 ± 0.0377 | 0.7398 ± 0.0377 | 0.1022 ± 0.0721 | 0.7796 ± 0.0126 | 0.7579 ± 0.0299 |

Historical0.7248/0.7458 are not matched comparators: this baseline resets Adam after3Epochs and uses a different fixed continuation window. Freeze effects are not a unique causal decomposition of modules; changing trainability changes optimization and interactions. Frozen Dropout remains stochastic in training. Reduced gap alone is not improvement. Conditional1000paired patient intervals omit refitting, selection, overlapping Train folds, two comparisons and prior exploration. Internal exploratory validation only.

All branch initial weight/RNG hashes and patient-order prefixes, frozen tensors, candidate selection and30final Train/Val C-indices were verified. No project-trained unimodal checkpoint initializes these models. All private predictions/weights remain local. No teacher/student change or automatic follow-up experiment follows.
