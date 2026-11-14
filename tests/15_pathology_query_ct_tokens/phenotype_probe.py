"""Linear phenotype decoding; no survival labels or encoder fitting."""

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def fit_probe(x,y,alphas):
    x,y=np.asarray(x,dtype=np.float64),np.asarray(y,dtype=np.float64)
    alphas=np.asarray(alphas,dtype=float)
    if (x.ndim!=2 or y.ndim!=2 or len(x)!=len(y) or alphas.shape!=(y.shape[1],)
            or not all(np.isfinite(a).all() for a in [x,y,alphas]) or (alphas<=0).any()):
        raise ValueError('Require finite aligned probe inputs and one positive alpha per target')
    head=Pipeline([('scale',StandardScaler()),('ridge',Ridge(alpha=alphas,solver='svd',fit_intercept=True))])
    return head.fit(x,y)


def select_probe(contexts,penalties,targets):
    if not contexts or not penalties or min(penalties)<=0 or not np.isfinite(penalties).all():
        raise ValueError('Require inner contexts and positive finite ridge candidates')
    for c in contexts:
        if set(c['train_ids'])&set(c['val_ids']):raise ValueError('Probe tuning patient overlap')
        if (len(c['train_ids'])!=len(c['train_ct']) or len(c['val_ids'])!=len(c['val_ct'])
                or len(set(c['train_ids']))!=len(c['train_ids']) or len(set(c['val_ids']))!=len(c['val_ids'])):
            raise ValueError('Probe patient identity/row mismatch')
    candidates=[]
    for alpha in penalties:
        errors=[]
        for c in contexts:
            head=fit_probe(c['train_ct'],c['train_targets'],[alpha]*len(targets))
            pred=head.predict(c['val_ct']).reshape(len(c['val_ct']),len(targets))
            error=np.mean((pred-np.asarray(c['val_targets'],dtype=float))**2,axis=0)
            if error.shape!=(len(targets),) or not np.isfinite(error).all():raise ValueError('Invalid inner probe error')
            errors.append(error.tolist())
        candidates.append(dict(ridge=float(alpha),inner_mse=errors,mean_inner_mse=np.mean(errors,axis=0).tolist()))
    chosen=[min(candidates,key=lambda c:(c['mean_inner_mse'][i],-c['ridge'])) for i in range(len(targets))]
    return dict(targets=targets,alphas=[c['ridge'] for c in chosen],
                mean_inner_mse=[c['mean_inner_mse'][i] for i,c in enumerate(chosen)],candidates=candidates,
                selection='target-wise mean inner MSE; exact ties favor stronger ridge; no DFS-label scoring')


def evaluate_probe(train_targets,truth,prediction,targets):
    train,truth,prediction=[np.asarray(x,dtype=np.float64) for x in [train_targets,truth,prediction]]
    if (truth.shape!=prediction.shape or truth.ndim!=2 or train.ndim!=2
            or truth.shape[1]!=len(targets) or train.shape[1]!=len(targets)
            or not all(np.isfinite(x).all() for x in [train,truth,prediction])):
        raise ValueError('Invalid probe evaluation arrays')
    rows=[]
    for i,name in enumerate(targets):
        baseline=np.full(len(truth),train[:,i].mean())
        rows.append(dict(target=name,r2=float(r2_score(truth[:,i],prediction[:,i],force_finite=False)),
            mae=float(mean_absolute_error(truth[:,i],prediction[:,i])),
            rmse=float(np.sqrt(mean_squared_error(truth[:,i],prediction[:,i]))),
            baseline_r2=float(r2_score(truth[:,i],baseline,force_finite=False)),
            baseline_mae=float(mean_absolute_error(truth[:,i],baseline)),
            baseline_rmse=float(np.sqrt(mean_squared_error(truth[:,i],baseline))),
            n_train=len(train),n_val=len(truth)))
    if not all(np.isfinite(list({k:v for k,v in r.items() if k!='target'}.values())).all() for r in rows):
        raise ValueError('Nonfinite probe metric; inspect target variance')
    return rows
