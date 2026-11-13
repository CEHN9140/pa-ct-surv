"""Strictly nested Shape decoding for a FIXED PUBLIC feature representation."""

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from phenotype_probe import fit_probe,select_probe


def fit_shape_probe(features,targets,train_ids,penalties,seed):
    if len(train_ids)<6:raise ValueError('Need at least six patients for nested probe tuning')
    contexts=[];roles=[]
    ids=np.asarray(train_ids,dtype=str)
    for fit,held in KFold(n_splits=3,shuffle=True,random_state=seed).split(ids):
        training,validation=ids[fit].tolist(),ids[held].tolist()
        roles.append(dict(train_ids=training,val_ids=validation))
        contexts.append(dict(train_ids=training,val_ids=validation,
            train_ct=features.loc[training].to_numpy(),val_ct=features.loc[validation].to_numpy(),
            train_targets=targets.loc[training].to_numpy(),val_targets=targets.loc[validation].to_numpy()))
    choice=select_probe(contexts,penalties,targets.columns.tolist())
    head=fit_probe(features.loc[train_ids].to_numpy(),targets.loc[train_ids].to_numpy(),choice['alphas'])
    return head,choice,roles


def crossfit_shape(features,targets,train_ids,val_ids,splits,penalties,seed):
    train_ids,val_ids=list(train_ids),list(val_ids)
    train,val=set(train_ids),set(val_ids)
    if train&val:raise ValueError('Shape cross-fitting patient overlap')
    if (len(train)!=len(train_ids) or len(val)!=len(val_ids)
            or not features.index.is_unique or not targets.index.is_unique):
        raise ValueError('Duplicate patient identity')
    if not(train|val)<=set(features.index) or not train<=set(targets.index):raise ValueError('Missing public features/Shape targets')
    held=[]
    for split in splits:
        a,b=list(split['train_ids']),list(split['val_ids'])
        if set(a)&set(b) or set(a)|set(b)!=train or len(set(a))+len(set(b))!=len(a)+len(b):
            raise ValueError('Invalid Shape OOF partition')
        held.extend(b)
    if len(splits)<2 or len(held)!=len(train) or set(held)!=train:raise ValueError('Incomplete Shape OOF partition')
    prediction=pd.DataFrame(np.nan,index=train_ids,columns=targets.columns)
    records=[]
    for number,split in enumerate(splits):
        a,b=list(split['train_ids']),list(split['val_ids'])
        head,selected,roles=fit_shape_probe(features,targets,a,penalties,seed+number)
        prediction.loc[b]=head.predict(features.loc[b].to_numpy())
        records.append(dict(train_ids=a,val_ids=b,tuning_roles=roles,selection=selected))
    if not np.isfinite(prediction).all().all():raise ValueError('Nonfinite or missing cross-fitted Shape')
    head,choice,roles=fit_shape_probe(features,targets,train_ids,penalties,seed+100)
    validation=pd.DataFrame(head.predict(features.loc[val_ids].to_numpy()),index=val_ids,columns=targets.columns)
    return dict(train_pred=prediction,val_pred=validation,head=head,selection=choice,
        roles=dict(oof=records,final=dict(train_ids=train_ids,val_ids=val_ids,tuning_roles=roles,selection=choice)))
