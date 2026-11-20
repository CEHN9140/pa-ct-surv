"""Aggregate diagnostics; publish only cohort-level results, never patient arrays."""
import itertools
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sksurv.metrics import concordance_index_censored
from diagnostics import risk_change,feature_change,attention_change

BASE=Path(__file__).resolve().parent


def summarize(base=BASE):
    base=Path(base);output=base/'results';output.mkdir(exist_ok=True)
    curves=[];finals=[];budgets=[];risks=[];features=[];phase_rows=[];initial=[]
    for batch in [32,64,128]:
        condition=base/f'bs{batch}'
        status=json.loads((condition/'status.json').read_text())
        if status['status']!='completed' or status['completed_folds']!=5:raise ValueError('Incomplete fivefold condition')
        final=pd.read_csv(condition/'results/metrics.csv');final=final[pd.to_numeric(final.fold,errors='coerce').notna()].copy();final['batch']=batch
        if len(final)!=5:raise ValueError('Incomplete final fold metrics')
        finals.append(final)
        for fold in range(5):
            directory=condition/f'diagnostics/fold_{fold}'
            frame=pd.read_csv(directory/'private_cohort.csv',dtype={'pa_id':str})
            curve=pd.read_csv(directory/'epoch_metrics.csv');curve['batch']=batch
            audit=json.loads((directory/'audit.json').read_text());last=audit['last_epoch'];best=audit['best_epoch']
            initial.append(audit['initial_state_sha256'])
            expected_epochs=set(range(last+1))
            if len(curve)!=2*(last+1) or set(curve.epoch)!=expected_epochs:raise ValueError('Missing checkpoint evaluations')
            updates_per_epoch=audit['optimizer_updates']//last
            if updates_per_epoch*last!=audit['optimizer_updates']:raise ValueError('Inconsistent optimizer count')
            budgets.append(dict(batch=batch,fold=fold,train_patients=audit['train_patients'],val_patients=audit['val_patients'],epochs=last,best_epoch=best,
                optimizer_updates=audit['optimizer_updates'],updates_per_epoch=updates_per_epoch,patient_exposures=last*audit['train_patients']))
            # Validate every saved risk vector and selected-checkpoint predictions independently.
            for epoch in expected_epochs:
                with np.load(directory/f'private/epoch_{epoch:03d}.npz',allow_pickle=False) as saved:
                    r=saved['risk']
                for split,mask in [('train',frame.split.ne(fold)),('val',frame.split.eq(fold))]:
                    score=concordance_index_censored(frame.loc[mask,'event'].to_numpy(bool),frame.loc[mask,'time'].to_numpy(np.float32),r[mask])[0]
                    recorded=curve.loc[(curve.epoch==epoch)&curve.split.eq(split),'cindex'].iloc[0]
                    if not np.isclose(score,recorded,atol=1e-12,rtol=0):raise ValueError('Saved risk C-index mismatch')
            for split,mask in [('train',frame.split.ne(fold)),('val',frame.split.eq(fold))]:
                predictions=pd.read_csv(condition/f'results/fold_{fold}/{split}_predictions.csv',dtype={'case_id':str}).set_index('case_id')
                with np.load(directory/f'private/epoch_{best:03d}.npz',allow_pickle=False) as saved:
                    np.testing.assert_allclose(predictions.loc[frame.loc[mask,'pa_id'],'risk_score'],saved['risk'][mask],atol=1e-7,rtol=1e-6)
            curves.append(curve)
            checkpoints=sorted({1,3,5,7,10,last,best})
            for left,right in itertools.combinations(checkpoints,2):
                with np.load(directory/f'private/epoch_{left:03d}.npz',allow_pickle=False) as a,np.load(directory/f'private/epoch_{right:03d}.npz',allow_pickle=False) as b:
                    for split,mask in [('train',frame.split.ne(fold)),('val',frame.split.eq(fold))]:
                        result=risk_change(a['risk'][mask],b['risk'][mask],frame.loc[mask,'time'].to_numpy(np.float32),frame.loc[mask,'event'].to_numpy(bool))
                        row=dict(batch=batch,fold=fold,split=split,early_epoch=left,late_epoch=right,**result);risks.append(row)
                        for label,early,late in [('1_to_5',1,5),('5_to_10',5,10),('10_to_last',10,last),('1_to_last',1,last)]:
                            if (left,right)==(early,late):phase_rows.append(dict(phase=label,**row))
            for label,left,right in [('1_to_last',1,last),('5_to_10',5,10),('10_to_last',10,last)]:
                if left==right:continue
                with np.load(directory/f'private/epoch_{left:03d}.npz',allow_pickle=False) as a,np.load(directory/f'private/epoch_{right:03d}.npz',allow_pickle=False) as b:
                    for split,mask in [('train',frame.split.ne(fold)),('val',frame.split.eq(fold))]:
                        row=dict(batch=batch,fold=fold,split=split,phase=label,early_epoch=left,late_epoch=right)
                        for key in ['pool512','post256']:row.update({key+'_'+k:v for k,v in feature_change(a[key][mask],b[key][mask]).items()})
                        row.update(attention_change(a,b,np.flatnonzero(mask)));features.append(row)
    if len(set(initial))!=1:raise ValueError('Initial model weights differ')
    curve=pd.concat(curves,ignore_index=True);final=pd.concat(finals,ignore_index=True)
    baseline=pd.read_csv(base.parent/'full_patch_epoch_controls/abmil/results/metrics.csv').iloc[:5]
    control=final[final.batch==32].sort_values('fold')
    np.testing.assert_allclose(control[['train_cindex','val_cindex']],baseline[['train_cindex','val_cindex']],atol=1e-10,rtol=0,
                               err_msg='Diagnostic instrumentation changed main-flow batch32 results')
    curve.to_csv(output/'learning_curves.csv',index=False);final.to_csv(output/'fold_metrics.csv',index=False)
    pd.DataFrame(budgets).to_csv(output/'training_budgets.csv',index=False)
    pd.DataFrame(risks).to_csv(output/'risk_pair_changes.csv',index=False)
    pd.DataFrame(phase_rows).to_csv(output/'risk_phase_changes.csv',index=False)
    pd.DataFrame(features).to_csv(output/'feature_attention_changes.csv',index=False)
    summary=[]
    for batch,rows in final.groupby('batch'):
        record=dict(batch=batch)
        for key in ['train_cindex','val_cindex','gap','auc_36m','auc_60m']:
            record[key+'_mean']=float(rows[key].mean(skipna=False));record[key+'_std']=float(rows[key].std(skipna=False))
        summary.append(record)
    pd.DataFrame(summary).to_csv(output/'batch_summary.csv',index=False)
    matched=[]
    for fold in range(5):
        group=curve[curve.fold==fold]
        for left,right in itertools.combinations([32,64,128],2):
            for split in ['train','val']:
                a=group[(group.batch==left)&group.split.eq(split)].set_index('optimizer_updates')
                b=group[(group.batch==right)&group.split.eq(split)].set_index('optimizer_updates')
                for update in sorted((set(a.index)&set(b.index))-{0}):
                    matched.append(dict(fold=fold,split=split,batch_left=left,batch_right=right,optimizer_updates=update,
                        epoch_left=int(a.loc[update,'epoch']),epoch_right=int(b.loc[update,'epoch']),
                        cindex_left=a.loc[update,'cindex'],cindex_right=b.loc[update,'cindex'],delta_cindex=b.loc[update,'cindex']-a.loc[update,'cindex']))
    pd.DataFrame(matched).to_csv(output/'matched_update_comparisons.csv',index=False)
    with plt.rc_context({'font.size':11,'axes.labelweight':'bold'}):
        fig,axes=plt.subplots(2,2,figsize=(12,8))
        for batch in [32,64,128]:
            for ax,split,key,title in [(axes[0,0],'train','cindex','Train C-index (eval)'),(axes[0,1],'val','cindex','Val C-index'),
                                      (axes[1,0],'val','risk_sd','Val risk SD'),(axes[1,1],'val','attention_entropy','Val normalized attention entropy')]:
                rows=curve[curve.batch.eq(batch)&curve.split.eq(split)]
                stats=rows.groupby('epoch')[key].agg(['mean','std','count']);stats=stats[stats['count']==5]
                ax.plot(stats.index,stats['mean'],label=f'Cox batch {batch}',linewidth=2)
                ax.fill_between(stats.index,stats['mean']-stats['std'],stats['mean']+stats['std'],alpha=.12)
                ax.set(xlabel='Epoch',ylabel=title);ax.grid(alpha=.2)
        axes[0,0].legend();fig.tight_layout();fig.savefig(output/'diagnostic_curves.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(3,5,figsize=(19,10),sharex=True,sharey=True)
    for row,batch in enumerate([32,64,128]):
        for fold in range(5):
            ax=axes[row,fold]
            for split in ['train','val']:
                part=curve[curve.batch.eq(batch)&curve.fold.eq(fold)&curve.split.eq(split)]
                ax.plot(part.epoch,part.cindex,label=split.title(),linewidth=2)
            ax.set_title(f'Cox {batch} / Fold {fold}');ax.set_xlabel('Epoch');ax.set_ylabel('Eval C-index');ax.grid(alpha=.2)
    axes[0,0].legend();fig.tight_layout();fig.savefig(output/'fold_learning_curves.png',dpi=160);plt.close(fig)
    lines=['# Cox batch, risk trajectory and representation diagnostics','',
        'Completed: Raw UNI/full patches, original ABMIL, five saved folds, batch32/64/128. Max30 Epoch, patience10, Adam1e-4/WD5e-4, dropout.25, seed123. Added evaluation preserves RNG; batch32 selected-model Train/Val C-index reproduces the original main-flow control.',
        '', 'The primary budget is a maximum of30 patient passes, not equal optimizer updates. Early stopping changes actual exposure. Matched-update comparisons are reported separately at exact common completed-epoch update counts; those comparisons have different exposure. No isolated causal effect of batch size is claimed.',
        '', '| Cox batch | Train C-index | Val C-index | AUC36m | AUC60m |','|---|---:|---:|---:|---:|']
    for row in summary:lines.append('| '+str(row['batch'])+' | '+' | '.join(f"{row[k+'_mean']:.4f} ± {row[k+'_std']:.4f}" for k in ['train_cindex','val_cindex','auc_36m','auc_60m'])+' |')
    lines+=['','## Early/late risk changes','', '| Batch | Phase | Split | Spearman | Risk SD ratio | Affine R² | Delta C |','|---|---|---|---:|---:|---:|---:|']
    phases=pd.DataFrame(phase_rows)
    for (batch,phase,split),rows in phases.groupby(['batch','phase','split']):
        lines.append(f"| {batch} | {phase} | {split} | {rows.spearman.mean():.4f} | {rows.risk_sd_ratio.mean():.3f} | {rows.affine_r2.mean():.4f} | {rows.delta_cindex.mean():+.4f} |")
    lines+=['','Risk SD growth with near-perfect affine fit/Spearman supports mainly rescaling for that interval. Material rank changes or corrected/broken comparable pairs preclude saying the change is only amplitude. A positive affine rescaling cannot itself improve ranking, apart from numerical tie-tolerance effects.',
        '', '## Representation/attention drift','', '| Batch | Phase | Split | Pool512 CKA | Post256 CKA | Attention JS | Top128 Jaccard |','|---|---|---|---:|---:|---:|---:|']
    for (batch,phase,split),rows in pd.DataFrame(features).groupby(['batch','phase','split']):
        lines.append(f"| {batch} | {phase} | {split} | {rows.pool512_linear_cka.mean():.4f} | {rows.post256_linear_cka.mean():.4f} | {rows.attention_js_mean.mean():.5f} | {rows.attention_top128_jaccard_mean.mean():.4f} |")
    lines+=['','CKA/attention drift describes frozen-checkpoint behavior, not biological correspondence or evidence that later features contain no DFS information. Attention entropy near1 also does not imply numerically identical weights.',
        '', 'All Train/Val diagnostic curves use the same fixed model in eval mode. Original online Train metrics remain in each training log. Each plotted mean±SD includes all5 folds only; later epochs with early-stopped folds are excluded from the plot, and counts remain available in CSV.',
        '', 'Absolute Train/Val Cox losses use different risk-set sizes and are not directly comparable as a gap; compare each split over time.', '', 'These are exploratory internal-validation results on a repeatedly inspected cohort. They cannot establish an intrinsic UNI information ceiling or prove that event count is the sole bottleneck. No new architecture, L2, label changes or teacher/student warm-starting were introduced. Private patient risks/attention/embeddings and all weights stay local.']
    (output/'report.md').write_text('\n'.join(lines)+'\n')
    (output/'validation.json').write_text(json.dumps(dict(passed=True,conditions=3,completed_folds=15,all_saved_risks_recomputed=True,
        common_initialization=True,batch32_mainflow_reproduced=True,private_patient_outputs_excluded_from_publication=True),indent=2))


if __name__=='__main__':summarize()
