"""Train-only Raw UNI dictionaries and explicitly margin-based class accounting."""
from pathlib import Path
import hashlib,sys
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans

ROOT=Path(__file__).resolve().parents[3];BASE=Path(__file__).resolve().parent
sys.path[:0]=[str(BASE.parent/'attention_risk_attribution'),str(BASE.parent/'attention_risk_attribution/best_reference')]
from attribution import comparable_pairs,pair_changes
from best_tools import context_scores,contextual_counts


def sample_indices(n,cap,patient):
    seed=int.from_bytes(hashlib.sha256(('cluster123:'+str(patient)).encode()).digest()[:8],'little')
    return np.sort(np.random.default_rng(seed).choice(n,min(n,cap),replace=False))


def fit_dictionary(frame,fold,load,n_clusters=20,cap=128):
    samples=[];weights=[];patients=[];indices=[];audit=[]
    for row in frame[frame.split.ne(fold)].itertuples():
        raw=np.asarray(load(row),np.float32)
        if raw.ndim!=2 or not len(raw) or not np.isfinite(raw).all():raise ValueError('Invalid Raw UNI bag')
        chosen=sample_indices(len(raw),cap,row.pa_id);samples.append(raw[chosen]);weights.extend([cap/len(chosen)]*len(chosen))
        patients.extend([row.pa_id]*len(chosen));indices.extend(chosen)
        audit.append(dict(pa_id=row.pa_id,saved_split=int(row.split),selected=len(chosen),total_weight=float(cap),sample_indices=','.join(map(str,chosen))))
    x=np.concatenate(samples);weight=np.asarray(weights,np.float32)
    if len(x)<n_clusters:raise ValueError('Too few training patches')
    model=MiniBatchKMeans(n_clusters=n_clusters,random_state=123,batch_size=min(2048,len(x)),max_iter=100,n_init=3,max_no_improvement=20,reassignment_ratio=.01)
    model.fit(x,sample_weight=weight);labels=model.predict(x);distance=np.sum((x-model.cluster_centers_[labels])**2,axis=1)
    radius=[];prototypes=[];patients=np.asarray(patients);indices=np.asarray(indices)
    for k in range(n_clusters):
        selected=np.flatnonzero(labels==k);radius.append(float(np.sqrt(np.average(distance[selected],weights=weight[selected]))) if len(selected) else np.nan)
        seen=set()
        for i in selected[np.argsort(distance[selected],kind='stable')]:
            if patients[i] in seen:continue
            seen.add(patients[i]);prototypes.append(dict(cluster=k,pa_id=patients[i],patch_index=int(indices[i]),distance=float(np.sqrt(distance[i]))))
            if len(seen)==4:break
    model.training_radius_=np.asarray(radius)
    return model,pd.DataFrame(audit),pd.DataFrame(prototypes)


def aggregate_bag(labels,best,late,contributions,norm,white,context,k):
    labels=np.asarray(labels,int);best=np.asarray(best,float);late=np.asarray(late,float);c=np.asarray(contributions,float)
    if any(len(x)!=len(labels) for x in [best,late,c,norm,white]) or (labels<0).any() or (labels>=k).any():raise ValueError('Misaligned class arrays')
    count=np.bincount(labels,minlength=k);rows=dict(cluster=np.arange(k),patch_count=count,patch_fraction=count/len(labels))
    for name,values in dict(best_mass=best,late_mass=late,ig_sum=c,abs_ig=np.abs(c),**context_scores(c,context)).items():rows[name]=np.bincount(labels,weights=values,minlength=k)
    rows['delta_mass']=rows['late_mass']-rows['best_mass']
    for name,values in [('mean_uni_norm',norm),('mean_thumbnail_white',white)]:
        rows[name]=np.divide(np.bincount(labels,weights=values,minlength=k),count,out=np.full(k,np.nan),where=count>0)
    return pd.DataFrame(rows)


def context_coefficients(before,after,time,event):
    before=np.asarray(before);after=np.asarray(after);a,b=comparable_pairs(time,event)
    def credit(r):
        d=r[a]-r[b];return np.where(abs(d)<=1e-8,.5,(d>0).astype(float))
    weights=np.abs(credit(after)-credit(before))
    return np.bincount(a,weights=weights,minlength=len(before)).astype(float)-np.bincount(b,weights=weights,minlength=len(before)).astype(float)


def changed_margin(components,coefficients,n_pairs,train_scale):
    if not np.isfinite(train_scale) or train_scale<=0:raise ValueError('Train-only risk scale must be positive')
    components=np.asarray(components,float)
    if len(coefficients)!=len(components):raise ValueError('Patient coefficient alignment')
    return np.asarray(coefficients)@components/n_pairs/train_scale if n_pairs else np.full(components.shape[1],np.nan)


def patient_contexts(observations):
    pairs,_=pair_changes(observations.frozen_best_attention_risk.to_numpy(),observations.frozen_late_attention_risk.to_numpy(),observations.time.to_numpy(),observations.event.to_numpy(bool))
    result=observations.copy();counts=contextual_counts(pairs,len(observations))
    for column in counts:result[column]=counts[column].to_numpy()
    return result
