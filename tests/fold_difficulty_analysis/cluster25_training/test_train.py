import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
sys.path.insert(0,str(Path(__file__).parent))
from train25 import build_loader

class Source:
    seed=123
    samples=pd.DataFrame({'pa_id':['a','b']})
    bags=[torch.arange(256*1024,dtype=torch.float32).reshape(256,1024),torch.arange(10*1024,dtype=torch.float32).reshape(10,1024)]
    labels=[np.repeat(np.arange(8),32),np.arange(10)%8]
    def __getitem__(self,i):return self.bags[i],torch.tensor(i),torch.tensor(float(i+1)),self.samples.pa_id.iloc[i]

def test_dynamic25_balanced_raw_and_full_inference():
    source=Source();original=source.bags[0].clone()
    loader=build_loader(source,[0],shuffle=True,pin_memory=False)
    a=next(iter(loader))[0];b=next(iter(loader))[0]
    assert a.shape==(1,25,1024) and not torch.equal(a,b)
    ix=(a[0,:,0]/1024).numpy().astype(int)
    counts=np.bincount(source.labels[0][ix],minlength=8)
    assert counts.sum()==25 and counts.min()==3 and counts.max()==4 and len(set(ix))==25
    replay=build_loader(source,[0],shuffle=True,pin_memory=False)
    assert torch.equal(a,next(iter(replay))[0]) and torch.equal(original,source.bags[0])
    loader.dataset.training=False
    assert next(iter(loader))[0].shape==(1,256,1024)
    val=build_loader(source,[1],shuffle=False,pin_memory=False)
    assert next(iter(val))[0].shape==(1,10,1024) and val.dataset.epoch==0
    small=build_loader(source,[1],shuffle=True,pin_memory=False)
    assert next(iter(small))[0].shape==(1,10,1024)


def test_main_training_checkpoint_and_full_evaluation(tmp_path):
    from types import SimpleNamespace
    import train25 as entry
    from model.build import Pa_Model
    class SmallSource:
        samples=pd.DataFrame({'pa_id':[str(i) for i in range(8)]})
        bags=[torch.randn(50,1024) for _ in range(8)]
        labels=[np.arange(50)%8]*8
        seed=123
        def __getitem__(self,i):return self.bags[i],torch.tensor(i%2),torch.tensor(float(i+1)),str(i)
    torch.set_num_threads(1)
    source=SmallSource();train=build_loader(source,[0,1,2,3],shuffle=True,pin_memory=False)
    val=build_loader(source,[4,5,6,7],pin_memory=False)
    model=Pa_Model(abmil_dropout=.25)
    args=SimpleNamespace(survival_loss='cox',cox_batch_size=3,num_epochs=2,patience=10,rank_weight=1.)
    entry.mainflow.train_path(model,train,val,torch.optim.Adam(model.parameters(),lr=1e-4,weight_decay=5e-4),args,torch.device('cpu'),0,tmp_path)
    assert (tmp_path/'best_model.pth').exists() and (tmp_path/'last_model.pth').exists()
    assert train.dataset.epoch==2 and val.dataset.epoch==0
    train.dataset.training=False
    assert next(iter(train))[0].shape[1]==50
    metrics=entry.mainflow.evaluate_survival(model,train,val,torch.device('cpu'))
    assert all(np.isfinite(metrics[k]) for k in ['train_cindex','val_cindex','train_loss','val_loss'])


def test_report_uses_only_five_real_folds(tmp_path):
    from report25 import five_folds
    import pytest
    p=tmp_path/'metrics.csv'
    pd.DataFrame(dict(fold=['0','1','2','3','4','mean','std'],val_cindex=[.7]*7)).to_csv(p,index=False)
    assert len(five_folds(p))==5
    pd.DataFrame(dict(fold=[0,1,2,3,3],val_cindex=[.7]*5)).to_csv(p,index=False)
    with pytest.raises(ValueError):five_folds(p)
