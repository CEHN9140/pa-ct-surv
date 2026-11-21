"""Separate weight-distribution concentration from patch-weight assignment."""
import numpy as np
from scipy.optimize import brentq
from scipy.special import softmax


def effective(weights):
    p=np.asarray(weights,float);p=p/p.sum()
    return float(1/np.square(p).sum())


def rank_preserving_profile(late,profile):
    late=np.asarray(late);values=np.asarray(profile)
    if late.shape!=values.shape or late.ndim!=1 or not np.isfinite(late).all() or not np.isfinite(values).all():
        raise ValueError('Aligned finite weight vectors required')
    result=np.empty_like(values)
    result[np.argsort(late,kind='stable')]=np.sort(values,kind='stable')
    return result


def match_effective(late,target):
    p=np.asarray(late,float)
    if p.ndim!=1 or not len(p) or not np.isfinite(p).all() or (p<=0).any():raise ValueError('Strictly positive native weights required')
    p=p/p.sum();n=len(p)
    if not 1<=target<=n+1e-7:raise ValueError('Invalid target Neff')
    if np.ptp(p)==0:
        if abs(target-n)>1e-7:raise ValueError('Uniform native weights cannot be sharpened without breaking ties')
        return p,1.
    if abs(target-effective(p))<1e-10:return p,1.
    if abs(target-n)<1e-9:return np.ones(n)/n,float('inf')
    logp=np.log(p)
    def residual(log_temperature):return effective(softmax(logp/np.exp(log_temperature)))-target
    if residual(-16)>1e-7 or residual(16)< -1e-7:raise ValueError('Target not attainable with rank-preserving temperature')
    root=brentq(residual,-16,16,xtol=1e-12)
    return softmax(logp/np.exp(root)),float(np.exp(root))


def patch_stats(attention,offsets):
    attention=np.asarray(attention,float);offsets=np.asarray(offsets,int)
    if offsets[0]!=0 or offsets[-1]!=len(attention) or (np.diff(offsets)<=0).any():raise ValueError('Invalid offsets')
    rows=[]
    for begin,end in zip(offsets[:-1],offsets[1:]):
        p=attention[begin:end];p=p/p.sum();n=len(p);k=min(128,n)
        positive=p>0;entropy=float(-(p[positive]*np.log(p[positive])).sum()/np.log(n)) if n>1 else 0.
        neff=effective(p)
        rows.append((n,neff,neff/n,float(np.partition(p,-k)[-k:].sum()),entropy))
    values=np.asarray(rows)
    return {key:values[:,i] for i,key in enumerate(['patches','effective','effective_ratio','top128_mass','entropy'])}
