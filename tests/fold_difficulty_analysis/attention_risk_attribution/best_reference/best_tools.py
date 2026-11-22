"""Contextual margin accounting and non-histological same-WSI matching."""
import numpy as np
import pandas as pd


def contextual_counts(pairs,n):
    result={}
    for name,weights in [('harm',(-pairs.credit_change.to_numpy()).clip(0)),('repair',pairs.credit_change.to_numpy().clip(0))]:
        for role,column in [('event','event_index'),('other','other_index')]:
            result[name+'_'+role]=np.bincount(pairs[column].to_numpy(int),weights=weights,minlength=n).astype(float)
    return pd.DataFrame(result)


def context_scores(contributions,counts):
    c=np.asarray(contributions,float);positive=c.clip(0);negative=(-c).clip(0)
    return dict(harm_support=negative*counts.harm_event+positive*counts.harm_other,
        harm_oppose=positive*counts.harm_event+negative*counts.harm_other,
        repair_support=positive*counts.repair_event+negative*counts.repair_other,
        repair_oppose=negative*counts.repair_event+positive*counts.repair_other)


def matched_controls(scores,features,white,count=8,caliper=.1):
    scores=np.asarray(scores,float);features=np.asarray(features,float);white=np.asarray(white,float);n=len(scores)
    if features.ndim!=2 or len(features)!=n or len(white)!=n or not np.isfinite(scores).all() or not np.isfinite(features).all() or not np.isfinite(white).all():raise ValueError('Invalid aligned matching inputs')
    k=min(count,n//2);order=np.argsort(scores,kind='stable');high=order[::-1][:k];high=high[scores[high]>0]
    low=order[:n//2];low=np.setdiff1d(low,high);used=set();rows=[]
    norms=np.linalg.norm(features,axis=1);unit=features/np.maximum(norms[:,None],1e-30)
    for index in high:
        available=np.array([i for i in low if i not in used and abs(white[i]-white[index])<=caliper+1e-12],int)
        if not len(available):continue
        similarities=unit[available]@unit[index];chosen=int(available[np.argmax(similarities)]);used.add(chosen)
        rows.append(dict(high_index=int(index),low_index=chosen,white_difference=float(abs(white[index]-white[chosen])),uni_cosine=float(unit[index]@unit[chosen]),
            high_score=float(scores[index]),low_score=float(scores[chosen])))
    return pd.DataFrame(rows,columns=['high_index','low_index','white_difference','uni_cosine','high_score','low_score'])


def white_fraction(rgb,coords,side,dimensions):
    rgb=np.asarray(rgb);maximum=rgb.max(2).astype(float);minimum=rgb.min(2).astype(float)
    white=(minimum>=220)&((maximum-minimum)/np.maximum(maximum,1)<=.1)
    h,w=white.shape;width,height=dimensions;coords=np.asarray(coords)
    x0=np.clip(np.floor(coords[:,0]*w/width).astype(int),0,w);x1=np.clip(np.ceil((coords[:,0]+side)*w/width).astype(int),0,w)
    y0=np.clip(np.floor(coords[:,1]*h/height).astype(int),0,h);y1=np.clip(np.ceil((coords[:,1]+side)*h/height).astype(int),0,h)
    integral=np.pad(white.astype(np.int64).cumsum(0).cumsum(1),((1,0),(1,0)));area=(x1-x0)*(y1-y0)
    if (area<=0).any():raise ValueError('Patch lies outside thumbnail')
    return (integral[y1,x1]-integral[y0,x1]-integral[y1,x0]+integral[y0,x0])/area
