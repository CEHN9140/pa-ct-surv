"""Average repeat draws within fold, then summarize five folds (SD, not CI)."""
import json,os
import numpy as np
import pandas as pd
from curve_data import HERE,dump,prepare

def aggregate(rows,sizes,metrics):
    expected={(m,f,s,r) for m in ['abmil','titan'] for f in range(5) for s in sizes for r in (range(1) if s=='full' else range(3))}
    if len(rows)!=len(expected) or set(zip(rows.model,rows.fold,rows.size_label,rows.repeat))!=expected:raise ValueError('Incomplete or duplicate learning-curve conditions')
    folds=[]
    for (model,fold,size),part in rows.groupby(['model','fold','size_label']):
        folds.append(dict(model=model,fold=fold,size_label=size,n_train=float(part.n_train.mean()),repeats=len(part),**{k:float(part[k].mean()) if part[k].notna().all() else np.nan for k in metrics}))
    folds=pd.DataFrame(folds);summary=[]
    for (model,size),part in folds.groupby(['model','size_label']):
        row=dict(model=model,size_label=size,n_train=float(part.n_train.mean()))
        for k in metrics:
            row[k+'_mean']=float(part[k].mean()) if part[k].notna().all() else np.nan
            row[k+'_sd']=float(part[k].std(ddof=1)) if part[k].notna().all() else np.nan
        summary.append(row)
    return folds,pd.DataFrame(summary)

def build():
    cfg,c,jobs=prepare();out=HERE/'results'
    audits=[]
    for j in jobs:
        roles=[]
        for model in ['abmil','titan']:
            directory=out/'private'/model/j['key']
            part=pd.read_csv(directory/'inner_roles.csv',dtype={'case_id':str})
            assert set(part.case_id)==set(c.iloc[j['indices']].pa_id) and part.case_id.is_unique
            roles.append(part.set_index('case_id').sort_index())
            for name,ix in [('train',j['indices']),('val',np.flatnonzero(c.split.eq(j['fold'])))]:
                pred=pd.read_csv(directory/f'{name}_predictions.csv',dtype={'case_id':str},float_precision='round_trip')
                assert pred.case_id.is_unique and set(pred.case_id)==set(c.iloc[ix].pa_id)
                aligned=pred.set_index('case_id').loc[c.iloc[ix].pa_id]
                assert np.array_equal(aligned.time,c.iloc[ix].time) and np.array_equal(aligned.event,c.iloc[ix].event) and np.isfinite(aligned.risk).all()
        assert roles[0].equals(roles[1])
        audit=pd.read_csv(out/'private/abmil'/j['key']/'fit_audit.csv')
        assert len(audit)==4 and audit.initial_state_sha256.nunique()==1
        assert (audit.patient_exposures%audit.n_fit==0).all()
        assert audit.loc[audit.role=='outer','evaluation_calls'].iloc[0]==0
        audits.extend(dict(key=j['key'],**r) for r in audit.to_dict('records'))
    pd.DataFrame(audits).to_csv(out/'fit_audit.csv',index=False)
    for model in ['abmil','titan']:assert json.loads((out/f'{model}_validation.json').read_text())['status']=='completed'
    rows=pd.concat([pd.read_csv(out/f'{m}_metrics.csv',dtype={'size_label':str}) for m in ['abmil','titan']],ignore_index=True)
    metrics=['train_cindex','val_cindex','gap','auc_36m','auc_60m','train_ee','val_ee','val_ec','val_ee_gap24']
    folds,summary=aggregate(rows,[str(n) for n in cfg['sample_sizes']]+['full'],metrics)
    folds.to_csv(out/'fold_means.csv',index=False);summary.to_csv(out/'summary.csv',index=False)
    os.environ.setdefault('MPLCONFIGDIR',str(out/'private/matplotlib'))
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(13,3.6),constrained_layout=True)
    for ax,metric,title in zip(axes,['val_cindex','val_ee','gap'],['Validation C-index','Validation EE C-index','Train - validation gap']):
        for m,label in [('abmil','Scratch ABMIL'),('titan','Frozen TITAN + Ridge')]:
            p=summary[summary.model==m].sort_values('n_train');ax.errorbar(p.n_train,p[metric+'_mean'],yerr=p[metric+'_sd'],marker='o',capsize=3,label=label)
        ax.set(xlabel='Training patients',ylabel=title);ax.grid(alpha=.2)
    axes[0].legend(fontsize=8);fig.savefig(out/'learning_curve.png',dpi=180);plt.close(fig)
    text=['# Matched CONCH patient-number learning curve','', 'Sampling repeats are averaged within each fold; values below are mean ± SD across five fold means. Error bars are fold SD, not confidence intervals. Full point is fitted once per fold.','', '|Model|Patients|Val C-index|Val EE|Train–Val gap|','|---|---:|---:|---:|---:|']
    for r in summary.sort_values(['model','n_train']).itertuples():text.append(f'|{r.model}|{r.n_train:.1f}|{r.val_cindex_mean:.4f} ± {r.val_cindex_sd:.4f}|{r.val_ee_mean:.4f}|{r.gap_mean:.4f}|')
    text+=['','## Protocol and limits','', 'Original 721 patients / 251 events / five outer folds. Train-only event-stratified nested subsets of 128, 256 and 384; sampling seeds 123, 7, 42. Neural initialization and inner split seed stay 123. Same subset and inner roles for both models.','', 'ABMIL: fresh initialization; full CONCHv1.5 bags; 3 inner folds select among epochs [1,2,3,5,8,10,15,20,30]. Adam 1e-4, weight decay 5e-4, dropout 0.25, Cox batch64 with tail merge. Outer Val is evaluated only after fresh refit to selected epoch. Full ABMIL is refitted under this epoch protocol, rather than reused from prior fixed-step results.','', 'TITAN: same frozen cached slide features; train-only StandardScaler and inner-selected Ridge penalty. The five full-cohort Ridge fits are reused only after exact cohort, inner-role, model-prediction and label checks. 45 new outer fits and 1350 inner attempts.','', 'This measures sample efficiency of complete pipelines. More patients also change updates per epoch and risk-set composition. Selected epochs may differ. Three repeated samples share patients and Outer Val; they are not independent replicates. These repeatedly explored internal folds cannot establish an information ceiling or a single causal explanation. Missing IPCW AUC values are retained, not silently removed.']
    (out/'report.md').write_text('\n'.join(text)+'\n')
    dump(out/'validation.json',dict(status='completed',conditions=100,abmil_fits=200,new_titan_outer_fits=45,reused_titan_full_fits=5,aggregation='repeat within fold, then five fold means',no_main_flow_changes=True))

if __name__=='__main__':build()
