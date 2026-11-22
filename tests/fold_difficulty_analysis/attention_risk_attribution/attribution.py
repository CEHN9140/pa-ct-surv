"""Exact straight-line attention IG and descriptive survival-pair/technical QC tools."""
import numpy as np
import pandas as pd
import torch
from torch import nn
from scipy.ndimage import laplace,binary_erosion


def attention_path(h,early,late,post,head):
    if post.training or head.training:raise ValueError('Frozen eval mode required')
    if len(post)!=3 or not isinstance(post[0],nn.Linear) or not isinstance(post[1],nn.ReLU) or not isinstance(post[2],nn.Dropout) or not isinstance(head,nn.Linear) or head.out_features!=1:
        raise ValueError('Exact integration requires Linear/ReLU/Dropout(eval)/Linear scalar head')
    h=h.detach().double();a=early.detach().to(h).reshape(-1);b=late.detach().to(h).reshape(-1)
    if h.ndim!=2 or len(a)!=len(h) or len(b)!=len(h) or not torch.isfinite(h).all():raise ValueError('Invalid aligned patch representations')
    for weights in [a,b]:
        if not torch.isfinite(weights).all() or (weights<0).any() or abs(float(weights.sum())-1)>2e-5:raise ValueError('Attention must be a probability vector')
    # Renormalize float32 softmax roundoff; neither patch order nor ranking changes.
    a=a/a.sum();b=b/b.sum();w=post[0].weight.detach().double();bias=post[0].bias.detach().double();v=head.weight.detach().double().reshape(-1)
    z0=a@h;z1=b@h;u0=w@z0+bias;u1=w@z1+bias;du=u1-u0
    active=(u0>0).double();moving=du!=0
    active[moving]=(u1[moving].relu()-u0[moving].relu())/du[moving]
    integrated_gradient=w.T@(v*active)
    contributions=(b-a)*(h@integrated_gradient)
    intercept=head.bias.detach().double().reshape(()) if head.bias is not None else h.new_zeros(())
    old=v@u0.relu()+intercept;new=v@u1.relu()+intercept
    return dict(contributions=contributions,risk_early_attention=float(old),risk_late_attention=float(new),risk_delta=float(new-old),
        completeness_error=abs(float(contributions.sum()-(new-old))))


def comparable_pairs(time,event):
    time=np.asarray(time);event=np.asarray(event,bool);i,j=np.triu_indices(len(time),1)
    left=event[i]&((time[i]<time[j])|((time[i]==time[j])&~event[j]))
    right=event[j]&((time[j]<time[i])|((time[j]==time[i])&~event[i]))
    return np.r_[i[left],j[right]],np.r_[j[left],i[right]]


def pair_changes(old,new,time,event):
    old=np.asarray(old,float);new=np.asarray(new,float);a,b=comparable_pairs(time,event)
    def credit(x):
        margin=x[a]-x[b];return np.where(abs(margin)<=1e-8,.5,(margin>0).astype(float))
    before=credit(old);after=credit(new);change=after-before
    pairs=pd.DataFrame(dict(event_index=a,other_index=b,credit_before=before,credit_after=after,credit_change=change,
        event_delta=new[a]-old[a],other_delta=new[b]-old[b],margin_delta=(new[a]-new[b])-(old[a]-old[b])))
    degree=np.bincount(np.r_[a,b],minlength=len(old));total=np.bincount(np.r_[a,b],weights=np.r_[change,change],minlength=len(old))
    broken=np.bincount(np.r_[a[change<0],b[change<0]],minlength=len(old));corrected=np.bincount(np.r_[a[change>0],b[change>0]],minlength=len(old))
    patients=pd.DataFrame(dict(patient_index=np.arange(len(old)),comparable_degree=degree,pair_credit_change=total,
        mean_pair_credit_change=np.divide(total,degree,out=np.full(len(old),np.nan,dtype=float),where=degree>0),broken_pairs=broken,corrected_pairs=corrected))
    return pairs,patients


def select_controls(contributions,count=8):
    order=np.argsort(np.abs(contributions),kind='stable');k=min(count,len(order)//2)
    return (order[-k:][::-1],order[:k]) if k else (order[:0],order[:0])


def technical_qc(rgb):
    rgb=np.asarray(rgb)
    if rgb.ndim!=3 or rgb.shape[2]!=3 or rgb.dtype!=np.uint8:raise ValueError('RGB uint8 patch required')
    x=rgb.astype(float);maximum=x.max(2);minimum=x.min(2)
    white=(minimum>=220)&((maximum-minimum)/np.maximum(maximum,1)<=.1);black=maximum<=30
    gray=x@np.array([.299,.587,.114]);od=-np.log((x+1)/256)
    tissue=(~white)&(~black);inside=binary_erosion(tissue,iterations=2,border_value=0);sharp=laplace(gray,mode='reflect')
    result=dict(white_fraction=float(white.mean()),black_fraction=float(black.mean()),tissue_fraction=float(tissue.mean()),
        laplacian_variance=float(np.var(sharp[inside])) if inside.sum()>=32 else np.nan,
        od_mean=float(od[tissue].mean()) if tissue.sum()>=32 else np.nan,
        saturation_mean=float(((maximum-minimum)/np.maximum(maximum,1))[tissue].mean()) if tissue.sum()>=32 else np.nan)
    for j,color in enumerate(['r','g','b']):result['od_'+color]=float(od[:,:,j][tissue].mean()) if tissue.sum()>=32 else np.nan
    return result
