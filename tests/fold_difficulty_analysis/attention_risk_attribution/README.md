# Attention changes, patch risk contributions and survival ordering

Independent frozen diagnostic following the user's design. No new loss, architecture, training, model selection or automatic publication. Raw complete UNI features, original721 patients/251 events/saved five folds, existing bs32 last checkpoints and cached Epoch5 attention/risk. All model modules run in eval mode.

Two questions stay separate: actual Epoch5→late risk changes; and attention-only changes holding late patch representations and risk head fixed. The second starts at early attention with late representations, not the actual early model.

## Exact Integrated Gradients

For h_i from the late projection, a0=Epoch5 attention and a1=late attention, z(t)=sum_i (a0_i+t*(a1_i-a0_i))*h_i. The eval risk head is Linear→ReLU→Dropout(identity)→Linear. For each ReLU unit its integrated activation derivative is (ReLU(u1)-ReLU(u0))/(u1-u0), with the constant-segment limit when equal. This gives an exact straight-line IG allocation to each attention coordinate, rather than numerical quadrature. Unit checks compare against independent dense autograd integration across nonlinear crossings.

c_i=(a1_i-a0_i)*h_i·integral(gradient_z risk)dt. Every patient's sum_i c_i must equal the frozen attention-only risk change; saved arrays are independently checked in report.py. Float32 attention sums are renormalized to remove softmax rounding; native risks/weights must reproduce stored snapshots before interpretation. Single-patch attribution is representation/path dependent, not an invariant patch importance or a causal deletion effect.

## Patient pairs

Use survival-comparable pairs, including event vs censor at equal time, half-credit for tied risks. For every pair (earlier event patient i, comparator j), attention-only margin change=sum(c_i)-sum(c_j). Both corrected and broken pairs are reported. Actual early→late broken pairs and their overlap with the frozen intervention are recorded separately; overlap is not a causal explanation of all temporal decline. Patients with no comparable pair have undefined normalized change and are excluded from outcome-based review ordering/correlations.

## Native technical proxies and review

Every WSI contributes8 largest absolute-IG patches and8 least-absolute-IG controls, disjoint (smaller equal groups for very short bags). High absolute IG includes positive and negative risk changes, not just harmful patches. Verify feature and coordinate array order, geometry metadata and original SVS dimensions. Read256px at the exact stored native pyramid scale/FOV (512 level0 pixels in this cohort); transparent out-of-slide pixels are white and valid coverage is recorded.

Measure white/black fractions, eroded-tissue Laplacian variance, optical density RGB summaries and saturation. These are technical proxies: sharpness also depends on tissue structure; OD is not H&E deconvolution. Within-WSI matching controls common slide/scanner/stain factors, not tissue type, spatial heterogeneity or all confounders. Report within-WSI group contrasts and fold-specific descriptive association with pair-credit changes. Extreme-group QC is not WSI-wide artifact prevalence.

Prepare15 representative cases (two worst actual temporal pair changes and one stable control per fold); group labels A/B and tile order are shuffled deterministically. Panels and WSI locations plus a blank reviewer/review.csv await a real reviewer. Tissue diagnoses and stable prognostic histology are not automatically inferred. Outcome-based case selection is exploratory and unsuitable for independent confirmation.

## Artifacts

run.sh launches pipeline.py: run.py→quality.py→report.py. Status: status.json; live log: validation/pipeline.log. Unit/smoke checks gate launch; source hashes are pinned. Existing outputs are not overwritten. All patient arrays, IDs, coordinates, original-patch panels and review keys stay under private/. Aggregate tables/report/plot go under results/. Main flow, previous results and source WSI/features/checkpoints are read-only.

Intervals resample paired patients conditional on fixed models, omit retraining/selection/prior exploration, and are not multiplicity-adjusted. Correlations cannot establish that quality caused overfitting. This experiment cannot by itself distinguish unstable biological associations from insufficient supervision or missing prognostic information. Actual pathology review remains pending.

Latest scope: actual Epoch5/Epoch10/best/late risks are all retained. Exact IG uses both Epoch5 and best-attention references with the same late feature/head. Primary QC/review selection remains Epoch5-reference absolute IG; each selected patch also records its best-reference contribution, so this is not a separate best-reference Top-K quality search. Worst/stable review case selection uses actual Epoch5-to-late pair changes; attention-only harm is reported separately. No train/validation tissue-association study is claimed at this stage.

Column distinction: patients.best_to_late_delta is the frozen counterfactual (best attention with late H/head→late attention); pairs.best_to_late_credit_change is actual best-model→actual late-model pair credit. They refer to different interventions and must not be combined as an additive decomposition.
