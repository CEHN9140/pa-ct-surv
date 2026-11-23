"""Verify each seed using the original audit, then summarize seed-specific paired gains."""
from pathlib import Path
from unittest.mock import patch
import json
import numpy as np
import pandas as pd
from repeat import load_backend,ORIGINAL

BASE=Path(__file__).resolve().parent;SEEDS=[7,42,2026]


def summarize():
    backend=load_backend('freeze_repeat_report_backend','report.py');all_metrics=[];seed_rows=[]
    for seed in SEEDS:
        directory=BASE/f'seed_{seed}'
        with patch.object(backend,'BASE',directory),patch.object(backend,'ARMS',['baseline','freeze_c']):backend.summarize()
        frame=pd.read_csv(directory/'results/fold_metrics.csv')
        if len(frame)!=10 or set(frame.arm)!=set(['baseline','freeze_c']):raise ValueError('Two-arm report count differs')
        proof=json.loads((directory/'results/validation.json').read_text());assert proof['passed'];proof.update(arms=2,training_seed=seed,split_seed=123)
        (directory/'results/validation.json').write_text(json.dumps(proof,indent=2))
        path=directory/'results/report.md';text=path.read_text().replace('three branches','two branches').replace('30final','20final').replace('two comparisons','one comparison')
        path.write_text(text+f'\nTraining seed={seed}; inner split seed remains123. Only baseline/freeze_c are run.\n')
        frame=pd.read_csv(directory/'results/fold_metrics.csv');frame['seed']=seed;all_metrics.append(frame)
        pivot=frame.pivot(index='fold',columns='arm',values='val_cindex');delta=pivot.freeze_c-pivot.baseline
        seed_rows.append(dict(seed=seed,baseline_val_mean=pivot.baseline.mean(),freeze_c_val_mean=pivot.freeze_c.mean(),mean_delta=delta.mean(),positive_folds=int(delta.gt(0).sum())))
    output=BASE/'results';output.mkdir(exist_ok=True);pd.concat(all_metrics,ignore_index=True).to_csv(output/'fold_metrics.csv',index=False);summary=pd.DataFrame(seed_rows);summary.to_csv(output/'seed_summary.csv',index=False)
    historical=pd.read_csv(ORIGINAL/'results/fold_metrics.csv');historical=historical[historical.arm.isin(['baseline','freeze_c'])].copy();historical['seed']=123
    pd.concat([historical,*all_metrics],ignore_index=True).to_csv(output/'with_historical_seed123.csv',index=False)
    lines=['# Freeze-C prespecified training-seed replication','',
        'Fixed original Outer fivefolds and fixed inner event-stratified80/20 split seed123. New training seeds7/42/2026; same Raw UNI, ABMIL, Cox32, Dropout.25, lr1e-4, WD5e-4,3Epoch common fork and0..20continuation selection. Only baseline and freeze_c. All warmups start newly; no project baseline checkpoint initialization. Per-seed audits reuse the original tensor/RNG/order/prediction checks, with exactly two arms.',
        '', '| Training seed | Baseline Val C | Freeze-C Val C | Paired mean delta | Positive folds |','|---|---:|---:|---:|---:|']
    for r in seed_rows:lines.append(f"| {r['seed']} | {r['baseline_val_mean']:.4f} | {r['freeze_c_val_mean']:.4f} | {r['mean_delta']:+.4f} | {r['positive_folds']}/5 |")
    lines+=['','Seed123 is retained as the previously explored reference; new seeds are reported separately. Repeated seeds/folds are not new independent patients, so no15-fold independent-sample significance or pooled raw-risk C-index is claimed. Conditional per-seed bootstrap does not cover retraining, selection or prior exploration. No freeze-epoch/lambda/architecture search is performed. A positive mean alone does not establish reproducible benefit or identify head C as a unique overfitting cause.']
    (output/'report.md').write_text('\n'.join(lines)+'\n');(output/'validation.json').write_text(json.dumps(dict(passed=True,new_training_seeds=SEEDS,outer_folds_per_seed=5,arms=2,split_seed=123,independent_patient_replications=False),indent=2))


if __name__=='__main__':summarize()
