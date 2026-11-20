"""Main pathology five-fold epoch training with experiment-local patch sampling."""
import json
import os
import sys
from pathlib import Path
from unittest.mock import patch

import torch
from torch.utils.data import Dataset

BASE=Path(__file__).resolve().parents[1]
ROOT=BASE.parents[2]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(BASE))
import path_train as mainflow
from cluster import load_cached_clusters,sha256
from sampler import select_indices


class PreparedDataset(mainflow.Path_Dataset):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.seed=123;self.bags=[];self.labels=[]
        for index,row in self.samples.iterrows():
            raw,event,time,patient=super().__getitem__(index)
            labels,_=load_cached_clusters(row.pa_path,patient,raw.numpy(),BASE/'cache')
            self.bags.append(raw);self.labels.append(labels)
            if index%100==0:print(f'Validated cluster cache: {index+1}/{len(self)}',flush=True)
    def __getitem__(self,index):
        row=self.samples.iloc[index]
        return self.bags[index],torch.tensor(row.event,dtype=torch.long),torch.tensor(row.time,dtype=torch.float32),row.pa_id


class ClusterDataset(Dataset):
    def __init__(self,source,indices,training):
        self.source=source;self.indices=list(indices);self.training=training;self.epoch=0
        self.samples=source.samples.iloc[self.indices]
    def __len__(self):return len(self.indices)
    def __getitem__(self,position):
        index=self.indices[position]
        raw,event,time,patient=self.source[index]
        if self.training:
            chosen=select_indices(self.source.labels[index],128,self.source.seed,'epoch_train',self.epoch,patient)
            raw=raw[chosen]
        return raw,event,time,patient


class EpochLoader:
    def __init__(self,loader):self.loader=loader;self.dataset=loader.dataset
    def __len__(self):return len(self.loader)
    def __iter__(self):
        if self.dataset.training:self.dataset.epoch+=1
        return iter(self.loader)


def build_loader(dataset,indices=None,batch_size=1,shuffle=False,num_workers=0,pin_memory=True):
    subset=ClusterDataset(dataset,range(len(dataset)) if indices is None else indices,shuffle)
    loader=mainflow_loader(subset,batch_size=batch_size,shuffle=shuffle,num_workers=num_workers,pin_memory=pin_memory)
    return EpochLoader(loader)


mainflow_loader=mainflow.build_dataloader


def main():
    torch.set_num_threads(4)
    if not torch.cuda.is_available():raise RuntimeError('CUDA required for the full experiment')
    if (BASE/'results').exists() or (BASE/'checkpoints').exists():raise FileExistsError('Refusing to overwrite outputs')
    original_train=mainflow.train_path;original_eval=mainflow.evaluate_survival;original_config=mainflow.save_run_config
    status=dict(status='preparing',pid=os.getpid(),protocol='mainflow_epoch_cv5',completed_folds=0)
    def save_config(args,*a,**kw):
        original_config(args,*a,**kw,extra=dict(train_patch_sampling='cluster_balanced128',cluster_input_l2=True,
            model_input_l2=False,cluster_cache=str(BASE/'cache'),validation_patches='Full WSI',protocol='mainflow_epoch_cv5'))
        files=[Path(__file__),ROOT/'path_train.py',ROOT/'dataset.py',ROOT/'model/build.py',ROOT/'model/abmil_cox.py',
               ROOT/'cox_utils.py',ROOT/'utils.py',ROOT/'loss/survival_loss.py',BASE/'cluster.py',BASE/'sampler.py',Path(args.data_dir)/'all_label_roi64.csv']
        (Path(args.results_root)/'source_sha256.json').write_text(json.dumps({str(p):sha256(p) for p in files},indent=2))
    def train(*a,**kw):
        status.update(status='training',active_fold=int(a[6]))
        (BASE/'status.json').write_text(json.dumps(status,indent=2))
        return original_train(*a,**kw)
    def evaluate(model,train_loader,val_loader,*a,**kw):
        train_loader.dataset.training=False
        result=original_eval(model,train_loader,val_loader,*a,**kw)
        status['completed_folds']+=1
        (BASE/'status.json').write_text(json.dumps(status,indent=2))
        return result
    try:
        (BASE/'status.json').write_text(json.dumps(status,indent=2))
        with patch.object(mainflow,'Path_Dataset',PreparedDataset),patch.object(mainflow,'build_dataloader',build_loader),\
             patch.object(mainflow,'train_path',train),patch.object(mainflow,'evaluate_survival',evaluate),\
             patch.object(mainflow,'save_run_config',save_config):
            mainflow.main()
        status['status']='completed'
    except Exception as exc:
        status.update(status='failed',error=f'{type(exc).__name__}: {exc}')
        raise
    finally:(BASE/'status.json').write_text(json.dumps(status,indent=2))


if __name__=='__main__':main()
