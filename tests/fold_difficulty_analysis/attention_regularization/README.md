# Attention regularization: controlled survival adaptation

Independent test of baseline Cox, normalized-entropy penalty and effective-patch-fraction hinge on original Raw UNI1024/full WSI ABMIL. No input L2, patch selection, new network modules, EMA or project-trained checkpoint initialization. Model/optimizer/data source stay unchanged:722178 parameters, dropout.25, Cox batch32 with merged tail, Adam1e-4/WD5e-4, seed123.

Entropy penalty per patient: max(0,1-H(p)/log(N)); singleton bags contribute differentiable zero. Neff fraction q=1/(N*sum(p²)); penalty max(0,q_min-q)². The Cox batch uses Cox+lambda*mean(patient penalty). All event/censored patients contribute to the regularizer; Cox itself remains event-averaged. Eval loss is unregularized Cox for comparability. Dropout is train-only. Active regularizer exposure is recorded.

Source: [AEM, MICCAI2025](https://papers.miccai.org/miccai-2025/0056-Paper5183.html), [official code](https://github.com/dazhangyu123/AEM). This follows the user's normalized-entropy formula adapted to survival, not an exact reproduction of the paper's classification loss/scale or Cosine Weight Annealing. The first experiment uses constant weights and no additional annealing to isolate the two penalties. ASMIL/EMA is not implemented.

## Validation protocol

Keep the original721 patients/251 events and CSV five outer folds. Within each Outer Train, a fixed event-stratified80/20 FIT/STOP split (seed123) is common to all candidates. This is a single inner holdout, not threefold inner CV. Inner max30 Epoch/patience10; STOP C-index selects candidate and Epoch. Outer data are never used for early stopping or parameter choice.

Seven inner candidates per fold: baseline; entropy lambda.01/.1; Neff q_min.05/.1 crossed with lambda1/10. No-reg candidate is also available to both regularized families, reusing the baseline inner run. Exact ties prefer weaker/no regularization and earlier Epoch. This is a small predefined grid, not evidence that the grid or q_min is universally optimal.

Each of baseline/entropy/neff then freshly trains full Outer Train for its inner-selected Epoch, with outer early stopping disabled. The fixed-budget last state is the final checkpoint; the main loop's training-reference best state is explicitly overwritten. Train-only reference monitoring is labelled distinctly. Outer Val receives one final inference pass that also captures Attention statistics. Three methods x five outer folds:35 inner fits+15 outer refits=50 neural fits. All start from the same new initialization, never trained unimodal weights.

Final outputs include Train/Val C/Loss,36/60-month AUC,KM/HR, mean entropy/Neff/q/Top128 mass, selected settings and active regularizer exposures. Fixed eval FIT/STOP curves preserve CPU/CUDA RNG and do not access Outer Val. Historical validation-selected best scores are not the matched baseline for this nested protocol.

This remains exploratory internal validation on a repeatedly studied cohort. Inner selected curves are not unbiased evidence of generalized improvement. Conditional patient-bootstrap intervals omit retraining, model selection and prior exploration. A failed small grid cannot rule out all entropy/Neff methods; reduced Train-Val gap or increased q alone is not success.

## Execution and artifacts

run.sh starts a pipeline with at most three outer jobs concurrently. Each job executes seven inner candidates and three fixed-budget outer refits. Logs fold_0/training.log through fold_4/training.log; progress validation/pipeline_status.json. Inner checkpoint/curve directories and outer/<family> outputs stay inside the corresponding fold. Patient mappings, predictions, raw features and weights remain local.

After all50 fits, report.py independently validates counts/splits/risks/checkpoints, creates comparison and stability figures, then publishes this experiment only via an explicit filename whitelist. Other staged changes, branch divergence, source drift or verification failure stop publication safely. No force push. Previous experiment data and main source are unchanged.
