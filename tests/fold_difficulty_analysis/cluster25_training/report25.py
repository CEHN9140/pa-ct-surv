"""Descriptive matched sampling and historical EAGLE comparisons; no model selection."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sksurv.metrics import concordance_index_censored
BASE=Path(__file__).resolve().parent;ROOT=BASE.parents[2]

def five_folds(path):
    t=pd.read_csv(path);t=t[t.fold.astype(str).isin([str(i) for i in range(5)])].copy()
    t['fold']=t.fold.astype(int)
    if len(t)!=5 or set(t.fold)!=set(range(5)):raise ValueError('Incomplete or duplicate folds')
    return t.sort_values('fold').reset_index(drop=True)

def build():
    roots={'Cluster25':BASE/'results','Cluster128':BASE.parent/'cluster128_training/results',
        'Full UNI':ROOT/'results/pact_v6/pathology/trident_features_uni_v1_cv5_roi64_abmil_cox_dropout0.25_epochs30_coxbs32_lr1e-4_wd5e-4'}
    for lr in ['1e-4','1e-3']:
        roots['EAGLE Top25 lr='+lr]=ROOT/f'results/pact_v7/pathology/eagle_top25_virchow2_cls_raw_cv5_roi64_abmil_cox_dropout0.25_epochs30_coxbs32_lr{lr}_wd5e-4'
    c=pd.read_csv(ROOT/'data/trident/features_uni_v1/all_label_roi64.csv',dtype={'pa_id':str})
    assert len(c)==721 and c.event.sum()==251
    tables={name:five_folds(path/'metrics.csv') for name,path in roots.items()}
    for fold in range(5):
        expected=c[c.split==fold]
        for name,path in roots.items():
            p=pd.read_csv(path/f'fold_{fold}/val_predictions.csv',dtype={'case_id':str})
            assert p.case_id.is_unique and set(p.case_id)==set(expected.pa_id)
            p=p.set_index('case_id').loc[expected.pa_id]
            assert np.array_equal(p['dfs.status'],expected.event)
            assert np.array_equal(p['dfs.month'].to_numpy(np.float32),expected.time.to_numpy(np.float32))
            score=concordance_index_censored(p['dfs.status'].astype(bool),p['dfs.month'],p.risk_score)[0]
            assert np.isclose(score,tables[name].loc[fold,'val_cindex'],atol=1e-12,rtol=0),name
    rows=[];deltas=[];metrics=['train_cindex','val_cindex','gap','auc_36m','auc_60m']
    for name,t in tables.items():
        row=dict(method=name)
        for k in metrics:row.update({k+'_mean':float(t[k].mean(skipna=False)),k+'_sd':float(t[k].std(ddof=1,skipna=False))})
        rows.append(row)
        if name!='Cluster25':
            for fold in range(5):deltas.append(dict(fold=fold,reference=name,cluster25=tables['Cluster25'].loc[fold,'val_cindex'],reference_cindex=t.loc[fold,'val_cindex'],delta=tables['Cluster25'].loc[fold,'val_cindex']-t.loc[fold,'val_cindex']))
    pd.DataFrame(rows).to_csv(BASE/'results/comparison_summary.csv',index=False)
    pd.DataFrame(deltas).to_csv(BASE/'results/fold_comparison.csv',index=False)
    text=['# Raw UNI cluster-balanced25','', 'Completed five folds. Same protocol as Cluster128; only training patch budget changed128→25. Eight fixed per-WSI clusters, newly sampled each epoch; Raw UNI inputs. Full WSI validation and final Train evaluation. Adam1e-4/WD5e-4, dropout.25, Cox32, max30 epochs, patience10, seed123. Outer Val selects best epoch; exploratory internal CV, not Nested CV.','', '|Method|Train C|Val C|AUC36|AUC60|','|---|---:|---:|---:|---:|']
    for r in rows:text.append('|'+r['method']+'|'+'|'.join(f'{r[k+"_mean"]:.4f} ± {r[k+"_sd"]:.4f}' for k in ['train_cindex','val_cindex','auc_36m','auc_60m'])+'|')
    text+=['','EAGLE uses CHIEF selection and Virchow2 CLS features; its comparison changes encoder and selection and cannot isolate patch sampling. Cluster25 vs Cluster128 is the intended sampling-budget contrast. These historical controls were not freshly retrained. No claim that sampled patches are DFS-specific. All predictions, checkpoints and logs remain local.']
    (BASE/'results/report.md').write_text('\n'.join(text)+'\n')
    (BASE/'results/validation.json').write_text(json.dumps(dict(status='completed',folds=5,patients=721,events=251,train_budget=25,validation='full_wsi',cohort_alignment_checked_all_references=True),indent=2))

if __name__=='__main__':build()
