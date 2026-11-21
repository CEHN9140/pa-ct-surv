"""Exploratory patient attribution; a changed pair belongs to both endpoints."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

BASE=Path(__file__).resolve().parent;SOURCE=BASE.parent


def participation(old,new,time,event):
    n=len(old);i,j=np.triu_indices(n,1)
    valid=((time[i]<time[j])&event[i])|((time[j]<time[i])&event[j])|((time[i]==time[j])&(event[i]!=event[j]))
    i,j=i[valid],j[valid]
    earlier=np.where((time[i]<time[j])|((time[i]==time[j])&event[i]),i,j)
    later=np.where(earlier==i,j,i)
    def score(r):
        gap=r[earlier]-r[later]
        return np.where(np.abs(gap)<=1e-8,.5,(gap>0).astype(float))
    a,b=score(old),score(new)
    corrected=(a==0)&(b==1);broken=(a==1)&(b==0)
    degree=np.bincount(np.r_[i,j],minlength=n)
    def count(mask):return np.bincount(np.r_[i[mask],j[mask]],minlength=n)
    corrected_nodes=count(corrected);broken_nodes=count(broken)
    before_error=np.bincount(np.r_[i,j],weights=np.r_[1-a,1-a],minlength=n)
    after_error=np.bincount(np.r_[i,j],weights=np.r_[1-b,1-b],minlength=n)
    rate=np.divide(after_error-before_error,degree,out=np.full(n,np.nan,dtype=float),where=degree>0)
    assert corrected_nodes.sum()==2*corrected.sum() and broken_nodes.sum()==2*broken.sum()
    return dict(degree=degree,corrected=corrected_nodes,broken=broken_nodes,net_harm_rate=rate,total_change_rate=np.divide(corrected_nodes+broken_nodes,degree,out=np.full(n,np.nan,dtype=float),where=degree>0),
        early_error_rate=np.divide(before_error,degree,out=np.full(n,np.nan,dtype=float),where=degree>0),
        late_error_rate=np.divide(after_error,degree,out=np.full(n,np.nan,dtype=float),where=degree>0)),(i,j,corrected,broken)


def attention_metrics(a,b,index):
    start,end=a['offsets'][index:index+2]
    p=a['attention'][start:end].astype(float);q=b['attention'][start:end].astype(float);p/=p.sum();q/=q.sum()
    entropy=lambda x:float(-(x*np.log(np.maximum(x,1e-30))).sum()/np.log(len(x))) if len(x)>1 else 0.
    effective=lambda x:float(1/np.square(x).sum())
    middle=(p+q)/2
    js=.5*np.sum(p*np.log(np.maximum(p,1e-30)/np.maximum(middle,1e-30))+q*np.log(np.maximum(q,1e-30)/np.maximum(middle,1e-30)))
    ea,eb=effective(p),effective(q);k=min(128,len(p))
    pa=a['post256'][index];pb=b['post256'][index]
    return dict(n_patches=len(p),attention_js=float(js),entropy_early=entropy(p),entropy_late=entropy(q),
        entropy_change=entropy(q)-entropy(p),effective_early=ea,effective_late=eb,effective_log_ratio=float(np.log(eb/ea)),
        top128_mass_early=float(np.partition(p,-k)[-k:].sum()),top128_mass_late=float(np.partition(q,-k)[-k:].sum()),
        embedding_relative_drift=float(np.linalg.norm(pb-pa)/(np.linalg.norm(pa)+1e-12)))


def run():
    patients=[];concentration=[]
    profiles=pd.read_csv(SOURCE.parent/'patient_error_profiles.csv',dtype={'pa_id':str}).set_index('pa_id')
    for batch in [32,64,128]:
        for fold in range(5):
            directory=SOURCE/f'bs{batch}/diagnostics/fold_{fold}'
            frame=pd.read_csv(directory/'private_cohort.csv',dtype={'pa_id':str})
            last=json.loads((directory/'audit.json').read_text())['last_epoch']
            train=frame.split.ne(fold).to_numpy();indices=np.flatnonzero(frame.split.eq(fold))
            group=frame.iloc[indices]
            for phase,left,right in [('5_to_10',5,10),('10_to_last',10,last)]:
                with np.load(directory/f'private/epoch_{left:03d}.npz',allow_pickle=False) as a,np.load(directory/f'private/epoch_{right:03d}.npz',allow_pickle=False) as b:
                    a={key:a[key] for key in ['risk','offsets','attention','post256']};b={key:b[key] for key in ['risk','offsets','attention','post256']}
                    old,new=a['risk'][indices].astype(float),b['risk'][indices].astype(float)
                    nodes,edges=participation(old,new,group.time.to_numpy(np.float32),group.event.to_numpy(bool))
                    z_old=(a['risk'][indices]-a['risk'][train].mean())/a['risk'][train].std()
                    z_new=(b['risk'][indices]-b['risk'][train].mean())/b['risk'][train].std()
                    for pos,index in enumerate(indices):
                        row=frame.iloc[index];patient=row.pa_id
                        record=dict(batch=batch,fold=fold,phase=phase,early_epoch=left,late_epoch=right,pa_id=patient,event=int(row.event),time=float(row.time),
                            risk_z_change=float(z_new[pos]-z_old[pos]),abs_risk_z_change=float(abs(z_new[pos]-z_old[pos])),**{k:float(v[pos]) for k,v in nodes.items()},
                            **attention_metrics(a,b,index))
                        if patient in profiles.index:
                            for key in ['feature_norm_mean','mean_pair_error']:record['historical_'+key]=float(profiles.loc[patient,key])
                        if patient in profiles.index:record['source_batch']=str(profiles.loc[patient,'source_batch'])
                        patients.append(record)
                    # Post-hoc concentration: compare pair incidence against all-pair and degree-only coverage.
                    k=int(np.ceil(len(group)*.2));top=np.argsort(-nodes['net_harm_rate'],kind='stable')[:k]
                    degree_top=np.argsort(-nodes['degree'],kind='stable')[:k]
                    i,j,corrected,broken=edges
                    for method,selected in [('net_harm_top20pct',top),('degree_top20pct',degree_top)]:
                        included=np.isin(i,selected)|np.isin(j,selected)
                        concentration.append(dict(batch=batch,fold=fold,phase=phase,method=method,selected_patients=k,total_patients=len(group),
                            all_pair_coverage=float(included.mean()),broken_pair_coverage=float(included[broken].mean()) if broken.any() else np.nan,
                            corrected_pair_coverage=float(included[corrected].mean()) if corrected.any() else np.nan))
    frame=pd.DataFrame(patients);frame.to_csv(BASE/'private/patient_changes.csv',index=False)
    pd.DataFrame(concentration).to_csv(BASE/'pair_concentration.csv',index=False)
    columns=['attention_js','entropy_change','effective_log_ratio','embedding_relative_drift','abs_risk_z_change','n_patches','historical_feature_norm_mean','historical_mean_pair_error','early_error_rate','time']
    correlations=[]
    for (batch,phase),rows in frame.groupby(['batch','phase']):
        target=rows.groupby('fold').net_harm_rate.rank(pct=True)
        for key in columns:
            feature=rows.groupby('fold')[key].rank(pct=True)
            valid=feature.notna()&target.notna()&rows.degree.gt(0)
            rho=spearmanr(feature[valid],target[valid]).statistic
            by_fold=[spearmanr(g[key],g.net_harm_rate,nan_policy='omit').statistic for _,g in rows.groupby('fold')]
            correlations.append(dict(batch=batch,phase=phase,feature=key,fold_rank_spearman=float(rho),positive_folds=int(np.sum(np.asarray(by_fold)>0)),
                negative_folds=int(np.sum(np.asarray(by_fold)<0)),patients=int(valid.sum())))
    pd.DataFrame(correlations).to_csv(BASE/'associations.csv',index=False)
    overlap=[]
    for phase in ['5_to_10','10_to_last']:
        pivot=frame[frame.phase.eq(phase)].pivot(index='pa_id',columns='batch',values='net_harm_rate')
        for left,right in [(32,64),(32,128),(64,128)]:
            both=pivot[[left,right]].dropna()
            overlap.append(dict(phase=phase,batch_left=left,batch_right=right,patients=len(both),spearman=float(spearmanr(both[left],both[right]).statistic),
                same_sign_fraction=float((np.sign(both[left])==np.sign(both[right])).mean())))
    pd.DataFrame(overlap).to_csv(BASE/'cross_batch_consistency.csv',index=False)
    groups=[]
    for keys,rows in frame[frame.degree.gt(0)].groupby(['batch','phase','event']):
        groups.append(dict(batch=keys[0],phase=keys[1],event=keys[2],patients=len(rows),mean_net_harm=float(rows.net_harm_rate.mean()),
            fraction_net_harmed=float(rows.net_harm_rate.gt(0).mean()),fraction_net_helped=float(rows.net_harm_rate.lt(0).mean()),
            mean_entropy_change=float(rows.entropy_change.mean()),mean_abs_z_risk_change=float(rows.abs_risk_z_change.mean())))
    pd.DataFrame(groups).to_csv(BASE/'event_group_summary.csv',index=False)
    print('Saved patient attribution privately and five aggregate diagnostic tables')


if __name__=='__main__':run()
