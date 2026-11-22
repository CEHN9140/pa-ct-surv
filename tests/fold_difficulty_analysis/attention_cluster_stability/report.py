"""Recompute class accounting and exploratory fold-local Train/Val margin stability."""
from pathlib import Path
import argparse,hashlib,json,sys
import h5py,joblib
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE=Path(__file__).resolve().parent;ROOT=BASE.parents[2];ATTR=BASE.parent/'attention_risk_attribution';PREVIOUS=ATTR/'best_reference'
sys.path[:0]=[str(ATTR),str(PREVIOUS)]
from attribution import comparable_pairs
from cluster_tools import aggregate_bag,sample_indices,context_coefficients,changed_margin,patient_contexts


def summarize(smoke=False):
    private=BASE/('validation/smoke' if smoke else 'private');out=private/'report' if smoke else BASE/'results';out.mkdir(exist_ok=True)
    assigned=json.loads((private/'assigned.json').read_text());qc_status=json.loads((private/'qc_completed.json').read_text())
    assert assigned['passed'] and qc_status['passed'];k=assigned['classes_per_fold'];nfolds=assigned['folds']
    cohort=pd.read_csv(ROOT/'data/trident/features_uni_v1/all_label_roi64.csv',dtype={'pa_id':str})
    source=pd.read_csv(PREVIOUS/'private/patients.csv',dtype={'pa_id':str}).set_index(['fold','pa_id'])
    table=pd.read_csv(private/'patient_clusters.csv',dtype={'pa_id':str})
    if not smoke and (assigned['fold_patient_observations']!=3605 or nfolds!=5 or k!=20):raise ValueError('Full cohort/fold/class counts changed')
    if table.duplicated(['fold','pa_id','cluster']).any() or len(table)!=assigned['fold_patient_observations']*k:raise ValueError('Missing class-patient rows')
    for fold in range(nfolds):
        if not smoke and set(table[table.fold.eq(fold)].pa_id)!=set(cohort.pa_id):raise ValueError('Missing assignment patients')
        fit=pd.read_csv(private/f'fold_{fold}/fit_patients.csv',dtype={'pa_id':str});model=joblib.load(private/f'fold_{fold}/dictionary.joblib')
        audit=json.loads((private/f'fold_{fold}/fit_audit.json').read_text())
        if hashlib.sha256(model.cluster_centers_.tobytes()).hexdigest()!=audit['centers_sha256']:raise ValueError('Saved dictionary differs')
        expected=set(cohort[cohort.split.ne(fold)].pa_id)
        if set(fit.pa_id)-expected or (not smoke and set(fit.pa_id)!=expected):raise ValueError('Fit leakage or dropped Train patients')
        if not (fit.saved_split.ne(fold).all() and not audit['validation_used_for_fit']):raise ValueError('Val guided dictionary')
        for row in fit.itertuples():
            meta=source.loc[(fold,row.pa_id)];idx=np.fromstring(row.sample_indices,sep=',',dtype=int)
            np.testing.assert_array_equal(idx,sample_indices(int(meta.n_patches),32 if smoke else 128,row.pa_id))
            if row.selected!=len(idx) or row.total_weight!=(32 if smoke else 128):raise ValueError('Sampling weights differ')
        contexts={}
        for role in ['train','val']:
            ids=table[table.fold.eq(fold)&table.split.eq(role)].pa_id.drop_duplicates().tolist()
            meta=patient_contexts(source.loc[[(fold,p) for p in ids]])
            for patient in ids:contexts[patient]=meta.loc[(fold,patient)]
        for patient,rows in table[table.fold.eq(fold)].groupby('pa_id'):
            rows=rows.sort_values('cluster');old=contexts[patient]
            if rows.split.nunique()!=1 or rows.split.iloc[0]!=old.split:raise ValueError('Role changed')
            with np.load(private/f'fold_{fold}/{patient}/assignment.npz',allow_pickle=False) as f:labels=f['labels'];distance=f['distance']
            with np.load(PREVIOUS/f'private/fold_{fold}/{patient}/best_attribution.npz',allow_pickle=False) as f:a=f['best_attention'];b=f['late_attention'];c=f['contributions']
            with np.load(private/f'technical/{patient}.npz',allow_pickle=False) as f:white=f['white']
            # Count/mass/IG/context reconstruction is independent of the aggregate CSV.
            reconstruct=aggregate_bag(labels,a,b,c,np.zeros(len(c)),white,old,k)
            for key in ['patch_count','patch_fraction','best_mass','late_mass','delta_mass','ig_sum','abs_ig','harm_support','harm_oppose','repair_support','repair_oppose','mean_thumbnail_white']:
                np.testing.assert_allclose(rows[key],reconstruct[key],rtol=1e-8,atol=1e-8,equal_nan=True)
            if len(labels)!=old.n_patches or abs(rows.ig_sum.sum()-old.attention_risk_delta)>1e-8*max(1.,abs(old.attention_risk_delta)):raise ValueError('IG/count completeness')
        # Fresh predictions on fixed Train and Val examples, never refit dictionary.
        frame=table[table.fold.eq(fold)][['pa_id','split']].drop_duplicates()
        for split in ['train','val']:
            ids=frame[frame.split.eq(split)].pa_id.tolist()
            for patient in ids[:1]+ids[-1:]:
                row=cohort.set_index('pa_id').loc[patient]
                with h5py.File(row.pa_path) as f:
                    idx=sample_indices(len(f['features']),32,patient);raw=f['features'][idx]
                with np.load(private/f'fold_{fold}/{patient}/assignment.npz',allow_pickle=False) as f:labels=f['labels'][idx]
                np.testing.assert_array_equal(model.predict(raw),labels)
    prototype_coverage=[]
    for fold in range(nfolds):
        proto=pd.read_csv(private/f'fold_{fold}/prototypes.csv',dtype={'pa_id':str})
        for cluster in range(k):
            rows=proto[proto.cluster.eq(cluster)]
            if rows.pa_id.duplicated().any() or len(rows)>4:raise ValueError('Prototype distinct-patient rule differs')
            prototype_coverage.append(dict(fold=fold,cluster=cluster,distinct_train_prototypes=len(rows),shortfall_from_four=4-len(rows)))
    pd.DataFrame(prototype_coverage).to_csv(out/'prototype_coverage.csv',index=False)
    summaries=[];rng=np.random.default_rng(123);bands=[]
    for fold in range(nfolds):
        current=table[table.fold.eq(fold)];train_ids=current[current.split.eq('train')].pa_id.drop_duplicates().tolist()
        scale=float(source.loc[[(fold,p) for p in train_ids]].frozen_best_attention_risk.std(ddof=0))
        if scale<=0:raise ValueError('No positive Train risk scale')
        for split in ['train','val']:
            group=current[current.split.eq(split)];ids=group.pa_id.drop_duplicates().tolist();meta=source.loc[[(fold,p) for p in ids]]
            matrix=group.pivot(index='pa_id',columns='cluster',values='ig_sum').loc[ids].to_numpy()
            coefficients=context_coefficients(meta.frozen_best_attention_risk.to_numpy(),meta.frozen_late_attention_risk.to_numpy(),meta.time.to_numpy(),meta.event.to_numpy(bool))
            n_pairs=len(comparable_pairs(meta.time.to_numpy(),meta.event.to_numpy(bool))[0]);direct=changed_margin(matrix,coefficients,n_pairs,scale)
            aggregate=group.groupby('cluster',sort=True)
            for cluster,rows in aggregate:
                helpful=(rows.repair_support.sum()+rows.harm_oppose.sum())/n_pairs/scale
                harmful=(rows.harm_support.sum()+rows.repair_oppose.sum())/n_pairs/scale
                if abs(helpful-harmful-direct[cluster])>1e-8:raise ValueError('Endpoint contextual margins do not reconcile')
                summaries.append(dict(fold=fold,split=split,cluster=int(cluster),patients=len(ids),patients_present=int(rows.patch_count.gt(0).sum()),
                    mean_patch_fraction=rows.patch_fraction.mean(),mean_best_mass=rows.best_mass.mean(),mean_late_mass=rows.late_mass.mean(),mean_delta_mass=rows.delta_mass.mean(),
                    mean_ig_sum=rows.ig_sum.mean(),normalized_helpful_margin=helpful,normalized_harmful_margin=harmful,normalized_net_changed_margin=float(direct[cluster]),
                    mean_uni_norm=rows.mean_uni_norm.mean(),mean_thumbnail_white=rows.mean_thumbnail_white.mean(),mean_assignment_distance=rows.mean_assignment_distance.mean(),
                    comparable_pairs=n_pairs,train_risk_scale=scale))
            if split=='val':
                e=meta.event.to_numpy(bool);t=meta.time.to_numpy(float);r0=meta.frozen_best_attention_risk.to_numpy();r1=meta.frozen_late_attention_risk.to_numpy();draws=[]
                for repeat in range(200 if smoke else 1000):
                    for attempt in range(1000):
                        ix=rng.integers(len(ids),size=len(ids));npairs=len(comparable_pairs(t[ix],e[ix])[0])
                        if npairs:break
                    else:raise ValueError('No comparable bootstrap pairs')
                    coeff=context_coefficients(r0[ix],r1[ix],t[ix],e[ix]);draws.append(changed_margin(matrix[ix],coeff,npairs,scale))
                draws=np.asarray(draws);std=draws.std(0,ddof=1);valid=std>1e-14
                standardized=np.zeros_like(draws);standardized[:,valid]=abs((draws[:,valid]-direct[valid])/std[valid])
                critical=float(np.quantile(standardized.max(1),.95))
                for cluster in range(k):bands.append(dict(fold=fold,cluster=cluster,bootstrap_sd=float(std[cluster]),pointwise_low=float(np.quantile(draws[:,cluster],.025)),pointwise_high=float(np.quantile(draws[:,cluster],.975)),
                    simultaneous_low=float(direct[cluster]-critical*std[cluster]),simultaneous_high=float(direct[cluster]+critical*std[cluster]),repeats=len(draws)))
    summary=pd.DataFrame(summaries)
    concentration=[]
    for (fold,split),rows in summary.groupby(['fold','split']):
        absolute=rows.normalized_net_changed_margin.abs();total=absolute.sum();shares=absolute/total if total>0 else absolute*0
        summary.loc[rows.index,'context_absolute_share']=shares
        concentration.append(dict(fold=fold,split=split,max_absolute_share=float(shares.max()),effective_context_classes=float(1/np.square(shares).sum()) if total>0 else 0.,positive_classes=int(rows.normalized_net_changed_margin.gt(0).sum()),negative_classes=int(rows.normalized_net_changed_margin.lt(0).sum())))
    pd.DataFrame(concentration).to_csv(out/'class_margin_concentration.csv',index=False)
    summary.to_csv(out/'fold_cluster_summary.csv',index=False)
    ci=pd.DataFrame(bands);ci.to_csv(out/'val_conditional_bands.csv',index=False)
    tr=summary[summary.split.eq('train')].set_index(['fold','cluster']);va=summary[summary.split.eq('val')].set_index(['fold','cluster'])
    contrast=pd.DataFrame(dict(train_margin=tr.normalized_net_changed_margin,val_margin=va.normalized_net_changed_margin,
        train_delta_mass=tr.mean_delta_mass,val_delta_mass=va.mean_delta_mass,train_prevalence=tr.mean_patch_fraction,val_prevalence=va.mean_patch_fraction)).reset_index()
    contrast=contrast.merge(ci,on=['fold','cluster'],validate='one_to_one');contrast['train_positive_val_negative']=(contrast.train_margin>0)&(contrast.val_margin<0)
    contrast['val_simultaneous_upper_negative']=contrast.simultaneous_high<0
    contrast.to_csv(out/'cluster_contrasts.csv',index=False)
    # Post-hoc centroid correspondences label dictionaries only; no assignments/risk data are changed.
    align=[];reference=joblib.load(private/'fold_0/dictionary.joblib')
    for fold in range(nfolds):
        model=joblib.load(private/f'fold_{fold}/dictionary.joblib');distance=cdist(reference.cluster_centers_,model.cluster_centers_)
        left,right=linear_sum_assignment(distance)
        for a,b in zip(left,right):
            radius=np.sqrt((reference.training_radius_[a]**2+model.training_radius_[b]**2)/2)
            align.append(dict(fold=fold,reference_fold0_cluster=int(a),local_cluster=int(b),raw_distance=float(distance[a,b]),
                distance_over_radius=float(distance[a,b]/radius) if radius>0 else np.nan,mutual_nearest=bool(np.argmin(distance[a])==b and np.argmin(distance[:,b])==a)))
    pd.DataFrame(align).to_csv(out/'descriptive_centroid_correspondence.csv',index=False)
    qc=pd.read_csv(private/'random_patch_quality.csv',dtype={'pa_id':str});qrows=[]
    if qc.duplicated(['pa_id','patch_index']).any():raise ValueError('Duplicate random QC tile')
    if set(qc.pa_id)!=set(table.pa_id):raise ValueError('Missing QC patients')
    for patient,rows in qc.groupby('pa_id'):
        fold=int(table[table.pa_id.eq(patient)].fold.iloc[0]);n=int(source.loc[(fold,patient)].n_patches)
        np.testing.assert_array_equal(np.sort(rows.patch_index),sample_indices(n,8,patient))
    for fold in range(nfolds):
        for patient,sample in qc.groupby('pa_id'):
            assignment=private/f'fold_{fold}/{patient}/assignment.npz'
            if not assignment.exists():continue
            with np.load(assignment,allow_pickle=False) as f:labels=f['labels'][sample.patch_index.to_numpy(int)]
            role=source.loc[(fold,patient)].split
            for row,label in zip(sample.to_dict('records'),labels):qrows.append(dict(fold=fold,split=role,cluster=int(label),**row))
    qt=pd.DataFrame(qrows);qt.to_csv(private/'random_qc_classes.csv',index=False)
    fields=['white_fraction','black_fraction','laplacian_variance','od_mean','saturation_mean']
    patient_q=qt.groupby(['fold','split','cluster','pa_id'])[fields].mean().reset_index()
    quality=patient_q.groupby(['fold','split','cluster']).agg(patient_observations=('pa_id','size'),**{key:(key,'mean') for key in fields}).reset_index()
    coverage=qt.groupby(['fold','split','cluster']).agg(native_patch_count=('pa_id','size')).reset_index()
    valid=patient_q.groupby(['fold','split','cluster']).agg(laplacian_patients=('laplacian_variance','count'),od_patients=('od_mean','count')).reset_index()
    grid=summary[['fold','split','cluster']].copy();quality=grid.merge(quality,on=['fold','split','cluster'],how='left').merge(coverage,on=['fold','split','cluster'],how='left').merge(valid,on=['fold','split','cluster'],how='left')
    for key in ['patient_observations','native_patch_count','laplacian_patients','od_patients']:quality[key]=quality[key].fillna(0).astype(int)
    quality.to_csv(out/'random_qc_by_class.csv',index=False)
    with plt.rc_context({'font.size':11,'axes.labelweight':'bold'}):
        fig,ax=plt.subplots(figsize=(9,7))
        for fold,g in contrast.groupby('fold'):ax.scatter(g.train_margin,g.val_margin,label=f'Fold {fold}',s=45,alpha=.8)
        ax.axhline(0,color='gray',lw=1);ax.axvline(0,color='gray',lw=1);ax.set(xlabel='Train normalized changed-pair margin contribution',ylabel='Val normalized changed-pair margin contribution')
        ax.legend();ax.grid(alpha=.2);fig.tight_layout();fig.savefig(out/'train_val_cluster_margins.png',dpi=180);plt.close(fig)
    counts=contrast.groupby('fold').agg(train_positive_val_negative=('train_positive_val_negative','sum'),conditional_val_upper_negative=('val_simultaneous_upper_negative','sum')).reset_index()
    counts.to_csv(out/'fold_pattern_counts.csv',index=False)
    lines=['# Fold-local UNI classes and Best-to-Late attribution stability','',
        f"Completed{nfolds} Train-only dictionaries, {assigned['fold_patient_observations']}fold-patient observations, {qc_status['random_native_patches']}outcome-independent native QC patches and {qc_status['oof_maps']}OOF WSI maps. No ABMIL training, no feature suppression or formal model change.",
        '', '## Predefined geometry and fit roles','',
        'Raw UNI1024 Euclidean MiniBatchKMeans,20classes per fold, up to128fixed patches per Train patient with equal total patient weight, seed123, batch2048, max_iter100, n_init3. No PCA, L2, Zscore, DFS labels or Val data enter dictionary fit. Fold-local labels are nominal, not tissue types. Sampled fit points are not all WSI patches; all Train/Val patches are assigned afterward without centroid updates.',
        '', '## What the signed margin metric means','',
        'For a changed comparable pair, patch IG from the event patient contributes+c to risk margin and comparator IG contributes-c. Credit-loss/credit-gain context keeps support and opposition. The normalized net metric sums signed class margin changes on all credit-changing pairs, weighted by absolute credit change (ties0.5), divides by the split comparable-pair count and the same full-Train reference risk SD. It is not delta C-index, a unique C-index allocation, or proof of a causal/biological relation. A positive patient risk shift alone is never treated as helpful. One class can support both harm and correction in different pair contexts.',
        '', '| Fold | Local classes with Train positive / Val negative margin | Classes with conditional simultaneous Val upper below0 |','|---|---:|---:|']
    for r in counts.itertuples():lines.append(f'| {r.fold} | {r.train_positive_val_negative}/{k} | {r.conditional_val_upper_negative}/{k} |')
    lines+=['','Val bands use1000paired patient resamples per fold, recomputing comparable pairs/credit transitions with fixed risks/IG/dictionary. The max-standardized band approximately covers20class contrasts within a fold; it does not cover fivefold exploration, dictionary refits, model retraining, best-epoch selection or prior repeated analysis. Best epochs were selected on Outer Val; a negative best-conditioned pattern is not independent validation. Train patients recur across four models; no significance from pooled Train rows is claimed.',
        '', '## Prevalence, attention and geometry checks','',
        'fold_cluster_summary.csv separates class prevalence, best/late attention mass, delta mass, signed IG, normalized contextual helpful/harmful margins, UNI norm, assignment distance and coarse all-patch thumbnail white fraction. Attention masses are compositional (sum to1), so class changes are dependent. class_margin_concentration.csv indicates whether contextual allocations are diffuse or concentrated; an allocation share is not causal importance. Changes can reflect class prevalence/domain differences and the projection/head-dependent attribution path, not just unstable morphology. Per-class means of norm/white/distance condition on presence; absent classes have zero mass and undefined technical mean.',
        '', '## Crossfold labels','',
        'descriptive_centroid_correspondence.csv provides post-hoc Raw Euclidean Hungarian matches to fold0 plus mutual-nearest and distance/radius diagnostics. It never changes class assignment or model predictions and does not establish identical biology. It is not used to pool class outcomes or declare that the same numeric class repeats across folds. Forced one-to-one matches can be weak and dictionaries can split/merge phenotypes.',
        '', '## Technical proxies and visualizations','',
        'Eight deterministic random patches per patient are selected without consulting outcomes, IG or class, read once at the native TRIDENT FOV, then assigned to each fold dictionary. random_qc_by_class.csv reports sparse patient-balanced conditional technical summaries; sample counts and missingness matter. Laplacian/OD/white proxies depend on local tissue structure and cannot label blur, stain artifact or tissue. OOF maps show local class colors on original thumbnails. Prototype panels show up to four nearest sampled Train patches from distinct patients per class; prototype_coverage.csv records rare-class shortfalls, not DFS-selected regions or histology diagnoses. All maps/native images/IDs/centers stay private.',
        '', '## Decision boundary','',
        'Train-positive/Val-negative classes are exploratory candidates for inspection, not a license to delete or downweight those patches and report improvement on the same Val set. No model performance gain is established. If most classes share the same sign contrast or allocations are diffuse, the result does not isolate one unstable phenotype. If a candidate appears, validate a predefined intervention with independent selection/evaluation; if no repeatable pattern is supported, stop this limited attention diagnostic rather than search K or tune cluster geometry on Val. Without reliable tissue labels this does not identify necrosis/stroma/tumor morphology, and no doctor is required to execute these computations.',
        '', 'Checks: Train fit roles, deterministic sampled indices/weights, immutable saved centroids, fresh Train/Val assignment samples, all class patch/mass/IG/context sums, both endpoint margin identity and independent QC sample assignment were verified.']
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    (out/'validation.json').write_text(json.dumps(dict(passed=True,folds=nfolds,classes_per_fold=k,fold_patient_observations=assigned['fold_patient_observations'],train_only_fit_verified=True,
        class_accounting_recomputed=True,context_margin_identity=True,neural_training=False,histology_labels=False,crossfold_class_outcomes_pooled=False),indent=2))
    print('Class stability report and independent recomputation complete',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');summarize(p.parse_args().smoke)
