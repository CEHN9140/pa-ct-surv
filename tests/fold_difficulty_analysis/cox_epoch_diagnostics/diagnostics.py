"""Fixed-model full-patient measurements; all patient arrays stay private."""
import math
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sksurv.metrics import concordance_index_censored

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from loss.survival_loss import cox_loss


def risk_change(old,new,time,event):
    old,new=np.asarray(old,float),np.asarray(new,float)
    time,event=np.asarray(time),np.asarray(event,bool)
    i,j=np.triu_indices(len(old),1)
    left=((time[i]<time[j])&event[i])|((time[i]==time[j])&event[i]&~event[j])
    right=((time[j]<time[i])&event[j])|((time[i]==time[j])&event[j]&~event[i])
    a=np.concatenate([i[left],j[right]]);b=np.concatenate([j[left],i[right]])
    def credit(risk):
        difference=risk[a]-risk[b]
        return np.where(np.abs(difference)<=1e-8,.5,(difference>0).astype(float))
    before,after=credit(old),credit(new)
    variance=np.var(old);slope=np.cov(old,new,ddof=0)[0,1]/variance if variance>0 else np.nan
    affine=slope*(old-old.mean())+new.mean()
    r2=1-np.sum((new-affine)**2)/np.sum((new-new.mean())**2) if np.var(new)>0 else np.nan
    rank_before=old[i]-old[j];rank_after=new[i]-new[j]
    flip=((rank_before>1e-8)&(rank_after< -1e-8))|((rank_before< -1e-8)&(rank_after>1e-8))
    return dict(spearman=float(spearmanr(old,new).statistic),risk_sd_old=float(old.std()),risk_sd_new=float(new.std()),
        risk_sd_ratio=float(new.std()/old.std()) if old.std()>0 else np.nan,affine_slope=float(slope),affine_r2=float(r2),
        all_pair_strict_flip_fraction=float(flip.mean()),comparable_pairs=len(a),corrected_pairs=int(((before==0)&(after==1)).sum()),
        broken_pairs=int(((before==1)&(after==0)).sum()),tie_related_changed_pairs=int(((before!=after)&((before==.5)|(after==.5))).sum()),
        delta_cindex=float((after-before).mean()) if len(a) else np.nan)


def feature_change(old,new):
    x=np.asarray(old,dtype=np.float64);y=np.asarray(new,dtype=np.float64)
    x=x-x.mean(0);y=y-y.mean(0)
    cross=np.linalg.norm(x.T@y,'fro')**2
    denom=np.linalg.norm(x.T@x,'fro')*np.linalg.norm(y.T@y,'fro')
    return dict(linear_cka=float(cross/denom) if denom else np.nan,
                centered_relative_l2=float(np.linalg.norm(y-x)/(np.linalg.norm(x)+1e-12)))


def attention_change(old,new,indices):
    rows=[]
    for index in indices:
        a,b=old['offsets'][index:index+2];p=old['attention'][a:b].astype(float);q=new['attention'][a:b].astype(float)
        p=p/p.sum();q=q/q.sum();middle=(p+q)/2
        js=.5*(np.sum(p*np.log(np.maximum(p,1e-30)/np.maximum(middle,1e-30)))+np.sum(q*np.log(np.maximum(q,1e-30)/np.maximum(middle,1e-30))))
        k=min(128,len(p));top_p=np.argpartition(p,-k)[-k:];top_q=np.argpartition(q,-k)[-k:]
        rows.append((js,len(np.intersect1d(top_p,top_q))/len(np.union1d(top_p,top_q))))
    return dict(attention_js_mean=float(np.mean([r[0] for r in rows])),attention_top128_jaccard_mean=float(np.mean([r[1] for r in rows])))


def evaluate_checkpoint(model,source,fold,epoch,updates,directory,capture=False):
    directory=Path(directory);private=directory/'private';private.mkdir(parents=True,exist_ok=True)
    frame=source.samples;device=next(model.parameters()).device;training=model.training
    gpu=[device.index] if device.type=='cuda' else []
    risks=[];entropies=[];effective=[];maxima=[];offsets=[0];all_attention=[];pools=[];embeddings=[];captured={}
    hook=model.mil.post_compression.register_forward_pre_hook(lambda module,inputs:captured.update(pool=inputs[0]))
    try:
        with torch.random.fork_rng(devices=gpu),torch.no_grad():
            model.eval()
            for x in source.bags:
                risk,embedding,weights=model(x.to(device))
                risk=float(risk.item());w=weights.detach().cpu().numpy().reshape(-1)
                if not np.isfinite(risk) or not np.isfinite(w).all():raise RuntimeError('Nonfinite checkpoint evaluation')
                risks.append(risk);entropies.append(float(-(w*np.log(np.maximum(w,1e-30))).sum()/math.log(len(w))) if len(w)>1 else 0.)
                effective.append(float(1/np.square(w).sum()));maxima.append(float(w.max()))
                if capture:
                    all_attention.append(w);offsets.append(offsets[-1]+len(w))
                    pools.append(captured['pool'].detach().cpu().numpy().reshape(-1));embeddings.append(embedding.detach().cpu().numpy().reshape(-1))
            risk=np.asarray(risks,np.float32);time=frame.time.to_numpy(np.float32);event=frame.event.to_numpy(bool)
            payload=dict(risk=risk)
            if capture:payload.update(attention=np.concatenate(all_attention),offsets=np.asarray(offsets,np.int64),pool512=np.stack(pools),post256=np.stack(embeddings))
            np.savez_compressed(private/f'epoch_{epoch:03d}.npz',**payload)
            result=[]
            for split,indices in [('train',np.flatnonzero(frame.split.ne(fold))),('val',np.flatnonzero(frame.split.eq(fold)))]:
                r,t,e=risk[indices],time[indices],event[indices]
                result.append(dict(fold=fold,epoch=epoch,optimizer_updates=updates,split=split,patients=len(indices),events=int(e.sum()),
                    cindex=float(concordance_index_censored(e,t,r)[0]),cox_loss=float(cox_loss(torch.from_numpy(r),torch.from_numpy(t),torch.from_numpy(e).float())),
                    risk_sd=float(r.std()),attention_entropy=float(np.mean(np.asarray(entropies)[indices])),effective_patches=float(np.mean(np.asarray(effective)[indices])),
                    max_attention=float(np.mean(np.asarray(maxima)[indices]))))
            return result
    finally:
        hook.remove();model.train(training)
