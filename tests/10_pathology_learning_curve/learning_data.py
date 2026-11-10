"""Freeze training-only nested subsets while retaining the original five folds."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from dataset import Path_Dataset


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def balanced_order(samples,rng):
    frame=samples.copy();frame['time_bin']=0
    for event,group in frame.groupby('event'):
        cuts=np.unique(np.quantile(group.time,[.25,.5,.75]))
        frame.loc[group.index,'time_bin']=np.searchsorted(cuts,group.time,side='right')
    groups=[[rng.permutation(group.index.to_numpy())for _,group in event_frame.groupby('time_bin')]
            for _,event_frame in frame.groupby('event')]
    counts=[np.array([len(group)for group in event_groups])for event_groups in groups]
    used=[np.zeros(len(values),dtype=int)for values in counts]
    event_counts=np.array([values.sum()for values in counts]);event_used=np.zeros(len(groups),dtype=int);order=[]
    for step in range(1,len(frame)+1):
        event_deficit=step*event_counts/len(frame)-event_used
        event_deficit[event_used==event_counts]=-np.inf
        event=int(np.argmax(event_deficit))
        deficit=(event_used[event]+1)*counts[event]/event_counts[event]-used[event]
        deficit[used[event]==counts[event]]=-np.inf
        group=int(np.argmax(deficit))
        order.append(int(groups[event][group][used[event][group]]))
        used[event][group]+=1;event_used[event]+=1
    return order


def check_subsets(samples,selected,validation,train_sizes,repeats):
    expected={(fold,f'n{size}',repeat)for fold in range(5)for size in train_sizes for repeat in range(repeats)}
    expected|={(fold,'full',0)for fold in range(5)}
    if set(selected.groupby(['fold','size_label','repeat']).groups)!=expected:
        raise ValueError('Missing or duplicate learning-curve conditions')
    for fold in range(5):
        val=validation.loc[validation.fold.eq(fold)]
        if set(val.row_index)!=set(samples.index[samples.split.eq(fold)])or val.row_index.duplicated().any():
            raise ValueError('Validation patients differ from original fold')
        if val.pa_id.tolist()!=samples.loc[val.row_index,'pa_id'].tolist():
            raise ValueError('Validation patient identities do not match row indices')
        for (label,repeat),group in selected.loc[selected.fold.eq(fold)].groupby(['size_label','repeat']):
            count=int((samples.split!=fold).sum())if label=='full'else int(label[1:])
            if len(group)!=count or group.row_index.duplicated().any() or set(group.row_index)&set(val.row_index):
                raise ValueError('Invalid subset size, repeated patient or validation leakage')
            if group.pa_id.tolist()!=samples.loc[group.row_index,'pa_id'].tolist():
                raise ValueError('Training patient identities do not match row indices')
        for repeat in range(repeats):
            smaller=set()
            for size in train_sizes:
                group=selected.loc[selected.fold.eq(fold)&selected['repeat'].eq(repeat)&selected.size_label.eq(f'n{size}')]
                current=set(group.row_index)
                if not smaller<=current:raise ValueError('Training subsets must be nested within each repeat')
                smaller=current


def prepare_subsets(label_csv,data_dir,train_sizes,repeats):
    source=Path(label_csv).resolve();root=Path(data_dir)
    if source.name!='all_label_roi64.csv':raise ValueError('Experiment10 expects all_label_roi64.csv')
    if train_sizes!=sorted(set(train_sizes))or min(train_sizes)<2 or repeats<1:
        raise ValueError('Use increasing unique subset sizes and a positive repeat count')
    dataset=Path_Dataset(source.parent,64);samples=dataset.samples
    if samples.pa_id.duplicated().any()or set(samples.split)!=set(range(5)):
        raise ValueError('Expected unique patients and the five fixed folds 0..4')
    if max(train_sizes)>=min(int((samples.split!=fold).sum())for fold in range(5)):
        raise ValueError('Requested subset sizes must be smaller than every full training fold')
    feature_state=[(p,Path(p).stat().st_size,Path(p).stat().st_mtime_ns)for p in samples.pa_path]
    settings=dict(source_csv=str(source),source_sha256=digest(source),train_sizes=train_sizes,repeats=repeats,
                  feature_signature=hashlib.sha256(json.dumps(feature_state).encode()).hexdigest())
    metadata_path=root/'manifest.json'
    if metadata_path.exists():
        metadata=json.loads(metadata_path.read_text())
        if metadata['settings']!=settings:raise ValueError('Source, features or sampling settings changed; use a new output directory')
        for name,expected_hash in metadata['file_hashes'].items():
            if digest(root/name)!=expected_hash:raise ValueError(f'Frozen learning-curve data changed: {name}')
    else:
        if root.exists()and any(root.iterdir()):raise ValueError('Incomplete data preparation; refusing to resample silently')
        root.mkdir(parents=True,exist_ok=True)
        (root/source.name).write_bytes(source.read_bytes())
        rng=np.random.default_rng();records=[];val_records=[]
        for fold in range(5):
            training=samples.loc[samples.split!=fold]
            for index in samples.index[samples.split.eq(fold)]:
                val_records.append(dict(fold=fold,row_index=int(index),pa_id=samples.loc[index,'pa_id']))
            for repeat in range(repeats):
                order=balanced_order(training,rng)
                for size in train_sizes:
                    for index in order[:size]:
                        records.append(dict(fold=fold,size_label=f'n{size}',repeat=repeat,
                                            row_index=index,pa_id=samples.loc[index,'pa_id']))
            for index in training.index:
                records.append(dict(fold=fold,size_label='full',repeat=0,row_index=int(index),pa_id=samples.loc[index,'pa_id']))
        selected=pd.DataFrame(records);validation=pd.DataFrame(val_records)
        check_subsets(samples,selected,validation,train_sizes,repeats)
        selected.to_csv(root/'subsets.csv',index=False);validation.to_csv(root/'validation.csv',index=False)
        metadata=dict(settings=settings,sampling='Unseeded once, then frozen; nested event/time-quartile stratification within training only',
                      file_hashes={name:digest(root/name)for name in [source.name,'subsets.csv','validation.csv']})
        metadata_path.write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+'\n')
    selected=pd.read_csv(root/'subsets.csv',dtype={'pa_id':str})
    validation=pd.read_csv(root/'validation.csv',dtype={'pa_id':str})
    check_subsets(samples,selected,validation,train_sizes,repeats)
    return Path_Dataset(root,64),selected,validation,metadata
