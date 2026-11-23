# Matched CONCH patient-number learning curve

Sampling repeats are averaged within each fold; values below are mean ± SD across five fold means. Error bars are fold SD, not confidence intervals. Full point is fitted once per fold.

|Model|Patients|Val C-index|Val EE|Train–Val gap|
|---|---:|---:|---:|---:|
|abmil|128.0|0.7188 ± 0.0521|0.5860|0.0590|
|abmil|256.0|0.7227 ± 0.0524|0.5875|0.0705|
|abmil|384.0|0.7267 ± 0.0496|0.5846|0.0525|
|abmil|576.8|0.7444 ± 0.0488|0.6012|0.0455|
|titan|128.0|0.7221 ± 0.0563|0.6028|0.0726|
|titan|256.0|0.7304 ± 0.0504|0.6025|0.0523|
|titan|384.0|0.7351 ± 0.0453|0.6091|0.0314|
|titan|576.8|0.7410 ± 0.0523|0.6162|0.0179|

## Protocol and limits

Original 721 patients / 251 events / five outer folds. Train-only event-stratified nested subsets of 128, 256 and 384; sampling seeds 123, 7, 42. Neural initialization and inner split seed stay 123. Same subset and inner roles for both models.

ABMIL: fresh initialization; full CONCHv1.5 bags; 3 inner folds select among epochs [1,2,3,5,8,10,15,20,30]. Adam 1e-4, weight decay 5e-4, dropout 0.25, Cox batch64 with tail merge. Outer Val is evaluated only after fresh refit to selected epoch. Full ABMIL is refitted under this epoch protocol, rather than reused from prior fixed-step results.

TITAN: same frozen cached slide features; train-only StandardScaler and inner-selected Ridge penalty. The five full-cohort Ridge fits are reused only after exact cohort, inner-role, model-prediction and label checks. 45 new outer fits and 1350 inner attempts.

This measures sample efficiency of complete pipelines. More patients also change updates per epoch and risk-set composition. Selected epochs may differ. Three repeated samples share patients and Outer Val; they are not independent replicates. These repeatedly explored internal folds cannot establish an information ceiling or a single causal explanation. Missing IPCW AUC values are retained, not silently removed.
