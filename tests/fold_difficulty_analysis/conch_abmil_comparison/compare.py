"""Group-specific paired-patient bootstrap, never independent-pair resampling."""
import numpy as np


def pair_credit_matrix(frame,group,reference_column='reference_risk'):
    t=frame['dfs.month'].to_numpy();e=frame['dfs.status'].to_numpy(bool)
    mask=e[:,None]&((t[:,None]<t[None,:])|((t[:,None]==t[None,:])&~e[None,:]))
    if group.startswith('EE'):
        mask &= e[None,:]
        if group=='EE_gap>24':mask &= (t[None,:]-t[:,None])>24
        elif group!='EE':raise ValueError('Unknown group')
    elif group=='EC':mask &= ~e[None,:]
    elif group!='all':raise ValueError('Unknown group')
    def credit(column):
        r=frame[column].to_numpy();margin=r[:,None]-r[None,:]
        return np.where(abs(margin)<=1e-8,.5,(margin>0).astype(float))
    return mask,np.where(mask,credit('risk_score')-credit(reference_column),0.)


def paired_group_interval(frames,group,repeats=2000,seed=123,reference_column='reference_risk'):
    observed=[];distributions=[]
    for fold,frame in enumerate(frames):
        mask,difference=pair_credit_matrix(frame,group,reference_column)
        if not mask.any():raise ValueError('No comparable subgroup pairs')
        observed.append(float(difference[mask].mean()))
        rng=np.random.default_rng(np.random.SeedSequence([seed,fold]));n=len(frame)
        draws=rng.integers(n,size=(repeats,n))
        counts=np.stack([np.bincount(row,minlength=n) for row in draws]).astype(float)
        denominator=np.einsum('bi,bi->b',counts@mask.astype(float),counts)
        numerator=np.einsum('bi,bi->b',counts@difference,counts)
        values=np.full(repeats,np.nan);np.divide(numerator,denominator,out=values,where=denominator>0)
        distributions.append(values)
    distribution=np.mean(distributions,axis=0);valid=distribution[np.isfinite(distribution)]
    if len(valid)<repeats/2:raise ValueError('Too few complete patient-bootstrap replicates')
    return dict(group=group,delta_cindex=float(np.mean(observed)),lower=float(np.quantile(valid,.025)),upper=float(np.quantile(valid,.975)),
                positive_folds=int((np.asarray(observed)>0).sum()),valid_repeats=len(valid))
