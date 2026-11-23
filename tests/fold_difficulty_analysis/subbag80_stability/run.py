"""Fixed best/late models,30 paired80% subbags; no training or method selection."""
from pathlib import Path
import argparse,json,sys
import h5py
import numpy as np
import pandas as pd
import torch
import yaml
from scipy.stats import spearmanr
from sksurv.metrics import concordance_index_censored

BASE=Path(__file__).resolve().parent;ROOT=BASE.parents[2];SOURCE=BASE.parent/'cox_epoch_diagnostics/bs32'
sys.path[:0]=[str(ROOT),str(BASE.parent/'attention_risk_attribution')]
from model.build import Pa_Model
from attribution import comparable_pairs,pair_changes
from subbag import cached_predictions


def main(smoke=False):
    torch.set_num_threads(4);out=BASE/('validation/smoke' if smoke else 'private');out.mkdir(parents=True,exist_ok=False)
    frame=pd.read_csv(ROOT/'data/trident/features_uni_v1/all_label_roi64.csv',dtype={'pa_id':str})
    if len(frame)!=721 or frame.event.sum()!=251:raise ValueError('Cohort differs')
    status=dict(status='running',training=False);records=[];patient_stats=[]
    try:
        for fold in range(1 if smoke else 5):
            status.update(fold=fold);(BASE/'status.json').write_text(json.dumps(status))
            directory=SOURCE/f'checkpoints/fold_{fold}';meta=yaml.safe_load((directory/'checkpoint_metadata.yaml').read_text())
            selected=np.arange(len(frame))
            if smoke:
                selected=np.concatenate([frame.index[(frame.split.eq(fold) if role=='val' else frame.split.ne(fold))&frame.event.eq(event)][:2] for role in ['train','val'] for event in [0,1]])
            original=pd.read_csv(SOURCE/f'diagnostics/fold_{fold}/private_cohort.csv',dtype={'pa_id':str})
            if original.pa_id.tolist()!=frame.pa_id.tolist():raise ValueError('Original patient order differs')
            np.testing.assert_array_equal(original.event,frame.event);np.testing.assert_array_equal(original.time.to_numpy(np.float32),frame.time.to_numpy(np.float32))
            np.testing.assert_array_equal(original.split,frame.split)
            models={}
            for phase in ['best','last']:
                model=Pa_Model(feature_dim=1024,abmil_dropout=.25,l2_normalize=False).cuda().eval()
                model.load_state_dict(torch.load(directory/f'{phase}_model.pth',map_location='cuda',weights_only=True));models[phase]=model
            predictions={key:[] for key in models}
            expected={}
            for phase in models:
                epoch=meta['best_epoch' if phase=='best' else 'last_epoch']
                with np.load(SOURCE/f'diagnostics/fold_{fold}/private/epoch_{epoch:03d}.npz',allow_pickle=False) as f:expected[phase]=f['risk'][selected]
            for pos,index in enumerate(selected):
                row=frame.iloc[index]
                with h5py.File(row.pa_path) as f:raw=torch.from_numpy(f['features'][:]).cuda()
                for phase,model in models.items():
                    values=cached_predictions(model,raw,row.pa_id)
                    if abs(float(values[0])-float(expected[phase][pos]))>2e-5:raise ValueError('Full checkpoint risk replay failed')
                    if pos==0:
                        from subbag import subset_indices
                        with torch.no_grad():direct=float(model(raw[subset_indices(len(raw),row.pa_id,0)])[0])
                        if abs(direct-float(values[1]))>2e-5:raise ValueError('Direct subbag replay failed')
                    predictions[phase].append(values)
                if (pos+1)%100==0 or pos==len(selected)-1:print(f'Fold {fold}: paired sampling {pos+1}/{len(selected)}',flush=True)
            current=frame.iloc[selected].reset_index(drop=True)
            for phase,values in predictions.items():
                array=np.asarray(values,np.float32);np.savez_compressed(out/f'fold_{fold}_{phase}.npz',indices=selected,risks=array)
                train=current.split.ne(fold).to_numpy();scale=float(array[train,0].std())
                if scale<=0:raise ValueError('No Train risk scale')
                for split,mask in [('train',train),('val',~train)]:
                    r=array[mask];group=current[mask];e=group.event.to_numpy(bool);t=group.time.to_numpy(np.float32)
                    a,b=comparable_pairs(t,e);i,j=np.triu_indices(len(group),1)
                    full_c=concordance_index_censored(e,t,r[:,0])[0]
                    for repeat in range(30):
                        x=r[:,repeat+1];old=r[:,0];pairs,_=pair_changes(old,x,t,e)
                        oldsign=np.sign(np.where(abs(old[i]-old[j])<=1e-8,0,old[i]-old[j]));newsign=np.sign(np.where(abs(x[i]-x[j])<=1e-8,0,x[i]-x[j]))
                        strict_flip=(oldsign*newsign)<0
                        records.append(dict(fold=fold,phase=phase,split=split,repeat=repeat,full_cindex=full_c,subbag_cindex=concordance_index_censored(e,t,x)[0],
                            spearman=float(spearmanr(old,x).statistic),all_pair_strict_flip=float(strict_flip.mean()),
                            comparable_credit_change_fraction=float(pairs.credit_change.ne(0).mean()),
                            comparable_strict_flip=float(((pairs.credit_before==0)&(pairs.credit_after==1)|((pairs.credit_before==1)&(pairs.credit_after==0))).mean()),
                            normalized_rmse=float(np.sqrt(np.mean((x-old)**2))/scale),train_risk_sd=scale))
                    for row,values in zip(group.itertuples(),r):patient_stats.append(dict(fold=fold,phase=phase,split=split,pa_id=row.pa_id,raw_subbag_sd=float(values[1:].std(ddof=1)),normalized_subbag_sd=float(values[1:].std(ddof=1)/scale)))
            pd.DataFrame(records).to_csv(out/'repeat_metrics.csv',index=False);pd.DataFrame(patient_stats).to_csv(out/'patient_sd.csv',index=False)
        summarize(out,smoke);status.update(status='completed',folds=1 if smoke else 5)
    except Exception as error:status.update(status='failed',error=f'{type(error).__name__}: {error}');raise
    finally:(BASE/'status.json').write_text(json.dumps(status,indent=2))


def summarize(private,smoke):
    metrics=pd.read_csv(private/'repeat_metrics.csv');patients=pd.read_csv(private/'patient_sd.csv',dtype={'pa_id':str})
    cohort=pd.read_csv(ROOT/'data/trident/features_uni_v1/all_label_roi64.csv',dtype={'pa_id':str})
    if len(metrics)!=(120 if smoke else 600):raise ValueError('Missing phase/role/repeat metrics')
    for fold in range(1 if smoke else 5):
        for phase in ['best','last']:
            with np.load(private/f'fold_{fold}_{phase}.npz',allow_pickle=False) as f:indices=f['indices'];risks=f['risks']
            if risks.shape!=(len(indices),31) or (not smoke and len(indices)!=721):raise ValueError('Incomplete risk matrix')
            frame=cohort.iloc[indices];scale=float(risks[frame.split.ne(fold).to_numpy(),0].std())
            for split in ['train','val']:
                mask=frame.split.eq(fold).to_numpy() if split=='val' else frame.split.ne(fold).to_numpy();r=risks[mask];g=frame[mask]
                t=g.time.to_numpy(np.float32);e=g.event.to_numpy(bool);full_c=concordance_index_censored(e,t,r[:,0])[0]
                rows=metrics[metrics.fold.eq(fold)&metrics.phase.eq(phase)&metrics.split.eq(split)].sort_values('repeat')
                if rows.repeat.tolist()!=list(range(30)):raise ValueError('Replicate IDs differ')
                for repeat,row in enumerate(rows.itertuples()):
                    pair,_=pair_changes(r[:,0],r[:,repeat+1],t,e)
                    c=concordance_index_censored(e,t,r[:,repeat+1])[0]
                    if abs(row.full_cindex-full_c)>1e-12 or abs(row.subbag_cindex-c)>1e-12:raise ValueError('C-index recomputation failed')
                    if abs(row.comparable_credit_change_fraction-pair.credit_change.ne(0).mean())>1e-12:raise ValueError('Pair flips differ')
                    if abs(row.spearman-float(spearmanr(r[:,0],r[:,repeat+1]).statistic))>1e-12:raise ValueError('Spearman differs')
                    strict=float((((pair.credit_before==0)&(pair.credit_after==1))|((pair.credit_before==1)&(pair.credit_after==0))).mean())
                    if abs(row.comparable_strict_flip-strict)>1e-12:raise ValueError('Comparable strict flips differ')
                    i,j=np.triu_indices(len(r),1);d0=r[i,0]-r[j,0];d1=r[i,repeat+1]-r[j,repeat+1]
                    flips=((d0>1e-8)&(d1< -1e-8))|((d0< -1e-8)&(d1>1e-8))
                    if abs(row.all_pair_strict_flip-float(flips.mean()))>1e-12:raise ValueError('All-pair flips differ')
                    if abs(row.train_risk_sd-scale)>1e-7:raise ValueError('Train reference scale differs')
                    if abs(row.normalized_rmse-float(np.sqrt(np.mean((r[:,repeat+1]-r[:,0])**2))/scale))>1e-7:raise ValueError('Normalized RMSE differs')

                sd=patients[patients.fold.eq(fold)&patients.phase.eq(phase)&patients.split.eq(split)]
                if sd.pa_id.astype(str).tolist()!=g.pa_id.astype(str).tolist():raise ValueError('Patient SD role/order differs')
                np.testing.assert_allclose(sd.raw_subbag_sd,r[:,1:].std(axis=1,ddof=1),rtol=2e-6,atol=1e-7)
                np.testing.assert_allclose(sd.normalized_subbag_sd,sd.raw_subbag_sd/scale,rtol=2e-6,atol=1e-7)

    output=private/'report' if smoke else BASE/'results';output.mkdir(exist_ok=True)
    fold=metrics.groupby(['fold','phase','split']).agg(full_cindex=('full_cindex','first'),subbag_cindex_mean=('subbag_cindex','mean'),subbag_cindex_sd=('subbag_cindex','std'),
        spearman_mean=('spearman','mean'),comparable_flip_mean=('comparable_strict_flip','mean'),credit_change_fraction=('comparable_credit_change_fraction','mean'),all_pair_flip_mean=('all_pair_strict_flip','mean'),normalized_rmse=('normalized_rmse','mean')).reset_index()
    sd=patients.groupby(['fold','phase','split'])[['raw_subbag_sd','normalized_subbag_sd']].mean().reset_index();fold=fold.merge(sd,on=['fold','phase','split'])
    fold['delta_c']=fold.subbag_cindex_mean-fold.full_cindex;fold.to_csv(output/'fold_metrics.csv',index=False)
    summary=fold.groupby(['phase','split']).mean(numeric_only=True).drop(columns='fold').reset_index();summary.to_csv(output/'summary.csv',index=False)
    contrast=fold.pivot(index=['fold','split'],columns='phase',values=['normalized_subbag_sd','spearman_mean','comparable_flip_mean','normalized_rmse','delta_c'])
    pd.DataFrame({key:contrast[key]['last']-contrast[key]['best'] for key in ['normalized_subbag_sd','spearman_mean','comparable_flip_mean','normalized_rmse','delta_c']}).to_csv(output/'late_minus_best.csv')
    lines=['# Frozen80% subbag sampling stability','',
        'No neural training. Original Raw UNI full WSI, five saved bs32 best/last models, eval mode. Each patient uses30deterministic80%-without-replacement subsets; identical subsets across checkpoints/folds, original patch order retained. Ceil(0.8N), singleton retained.',
        '', 'In eval ABMIL, projection/attention logits are patch independent; cached subset-softmax/pooling is equivalent to direct subset forwards. Unit tests and direct real-GPU checks verify this. Full risks must reproduce original snapshot predictions. Risk SD uses30subbag predictions; normalized SD/RMSE divides by that checkpoint full-Train risk SD. Spearman, C-index, all-pair strict flips and comparable-pair credit/strict changes are distinct metrics.',
        '', '| Phase | Role | Full C | Subbag mean C | Mean Spearman | Comparable flip fraction | Normalized subbag SD |','|---|---|---:|---:|---:|---:|---:|']
    for r in summary.itertuples():lines.append(f'| {r.phase} | {r.split} | {r.full_cindex:.4f} | {r.subbag_cindex_mean:.4f} | {r.spearman_mean:.4f} | {r.comparable_flip_mean:.4f} | {r.normalized_subbag_sd:.4f} |')
    lines+=['','Repeated subbags are correlated technical replicates, not30independent clinical validations. Train patients recur across folds. Best checkpoints used Outer Val selection. This diagnoses sensitivity, not its causal role in overfitting. No sampling proportion/lambda is chosen from Outer Val and no consistency training is automatically started. Historical strict baseline0.7248 applies only to its exact inner-holdout protocol; no new training comparison is made here.']
    (output/'report.md').write_text('\n'.join(lines)+'\n');(output/'validation.json').write_text(json.dumps(dict(passed=True,training=False,folds=1 if smoke else 5,repeats=30,paired_subsets=True,patient_phase_observations=len(patients),full_risks_reproduced=True,direct_subset_checks=True,saved_risks_metrics_independently_recomputed=True),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');main(p.parse_args().smoke)
