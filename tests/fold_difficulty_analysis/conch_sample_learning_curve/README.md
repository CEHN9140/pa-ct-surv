# CONCH patient-number learning curve

Isolated diagnostic; no changes to the proposed teacher/student or main training.

Compare scratch ABMIL with frozen TITAN + Ridge using exactly matched CONCHv1.5 bags / slide features, unchanged 721 patients / 251 events / five outer folds.

- Subsets: 128, 256, 384, full Outer Train. Three train-only stratified, nested draws (seeds 123,7,42); full is not triplicated.
- Neural seed and three-fold inner selection seed: 123.
- ABMIL: inner selection over fixed epoch candidates; fresh outer refit, no outer early stopping. Adam 1e-4, WD5e-4, dropout .25, Cox64 with tail merge. Every epoch covers all FIT patients exactly once.
- TITAN: frozen existing features; FIT-only scaling + Ridge selection; identical inner roles. Five full-point fits reused after label/role/prediction checks; smaller cohorts fitted anew.
- Repeats averaged within fold before five-fold summaries; fold SD is descriptive, not a confidence interval. No information-ceiling claim.

Run from repository root:

```bash
bash tests/fold_difficulty_analysis/conch_sample_learning_curve/launch.sh
```

The launcher runs GPU ABMIL and CPU Ridge in parallel and builds the report only if both succeed. It refuses to overwrite existing scientific job directories. Logs, patient lists, predictions, feature arrays and weights stay under ignored `results/private/`. Public outputs contain aggregate metrics only.

Expected workload: 200 neural fits, 1350 new Ridge inner attempts, 45 new Ridge outer fits, 5 reused full Ridge fits. This is sampling repetition, not a neural random-seed robustness experiment.
