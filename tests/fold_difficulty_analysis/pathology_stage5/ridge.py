"""Raw all-patch mean and train-only nested StandardScaler + Ridge-Cox."""

import warnings

import h5py
import numpy as np
from scipy.linalg import LinAlgWarning
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import concordance_index_censored


def pool_feature(path,feature_dim=1024,block_size=512):
    with h5py.File(path,'r') as handle:
        features=handle['features']
        if features.ndim!=2 or not len(features) or features.shape[1]!=feature_dim:
            raise ValueError(f'Expected nonempty UNI [N,{feature_dim}] in {path}')
        total=np.zeros(feature_dim,dtype=np.float64)
        for start in range(0,len(features),block_size):
            block=np.asarray(features[start:start+block_size],dtype=np.float64)
            if not np.isfinite(block).all():
                raise ValueError(f'Nonfinite raw features in {path}')
            total+=block.sum(axis=0)
        return total/len(features),len(features)


def fit_ridge(x,y,alpha):
    model=Pipeline([('scale',StandardScaler()),
                    ('cox',CoxPHSurvivalAnalysis(alpha=float(alpha),ties='breslow',n_iter=1000,tol=1e-9))])
    # A failed/unfinished numerical fit must never become a candidate with a score.
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        model.fit(x,y)
    if not np.isfinite(model.named_steps['cox'].coef_).all() or not np.isfinite(model.predict(x)).all():
        raise FloatingPointError('Nonfinite Ridge-Cox coefficients/predictions')
    return model


def select_alpha(x,y,seed=123,alphas=(.1,1.,10.,100.,1000.,10000.)):
    splitter=StratifiedKFold(3,shuffle=True,random_state=seed)
    folds=list(splitter.split(x,y['event']))
    roles=np.full(len(x),-1,dtype=int)
    for inner,(_,held) in enumerate(folds): roles[held]=inner
    rows=[]
    for alpha in alphas:
        for inner,(fit,held) in enumerate(folds):
            error=None
            score=np.nan
            try:
                model=fit_ridge(x[fit],y[fit],alpha)
                prediction=model.predict(x[held])
                if not np.isfinite(prediction).all(): raise FloatingPointError('Nonfinite inner prediction')
                score=concordance_index_censored(y['event'][held],y['time'][held],prediction)[0]
                if not np.isfinite(score): raise FloatingPointError('Nonfinite inner C-index')
            except (RuntimeWarning,ConvergenceWarning,LinAlgWarning,np.linalg.LinAlgError,FloatingPointError) as exc:
                error=f'{type(exc).__name__}: {exc}'
            except ValueError as exc:
                if 'search direction contains NaN or infinite values' not in str(exc): raise
                error=f'{type(exc).__name__}: {exc}'
            if error:
                print(f'  REJECT alpha={alpha:g} inner={inner}: {error}',flush=True)
            rows.append(dict(alpha=float(alpha),inner_fold=inner,cindex=float(score),converged=error is None,error=error,
                             n_fit=len(fit),n_held=len(held),n_fit_events=int(y['event'][fit].sum())))
        candidate=[r for r in rows if r['alpha']==float(alpha)]
        eligible=all(r['converged'] for r in candidate)
        for record in candidate: record['candidate_eligible']=eligible
        print(f'  alpha={alpha:g} | '+(f'three-fold mean C-index={np.mean([r["cindex"] for r in candidate]):.6f}'
              if eligible else 'INELIGIBLE: at least one failed inner fit'),flush=True)
    means={float(alpha):np.mean([r['cindex'] for r in rows if r['alpha']==float(alpha)]) for alpha in alphas
           if all(r['candidate_eligible'] for r in rows if r['alpha']==float(alpha))}
    if not means: raise RuntimeError(f'No complete three-fold candidates: {rows}')
    best=max(means.values())
    chosen=max(alpha for alpha,score in means.items() if np.isclose(score,best,rtol=0,atol=1e-12))
    return chosen,rows,roles


def fit_outer(x,y,seed=123,alphas=(.1,1.,10.,100.,1000.,10000.)):
    alpha,rows,roles=select_alpha(x,y,seed,alphas)
    return fit_ridge(x,y,alpha),alpha,rows,roles
