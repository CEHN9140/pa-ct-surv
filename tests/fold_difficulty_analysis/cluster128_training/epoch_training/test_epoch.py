import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
sys.path.insert(0,str(Path(__file__).parent))
from train_epoch import ClusterDataset, EpochLoader, build_loader


def test_epoch_samples_and_full_eval():
    class Source:
        samples=pd.DataFrame({'pa_id':['a','b','c','d']})
        bags=[torch.arange(256*1024,dtype=torch.float32).reshape(256,1024)]*4
        labels=[np.repeat(np.arange(8),32)]*4
        seed=123
        def __getitem__(self,i):return self.bags[i],torch.tensor(i%2),torch.tensor(float(i+1)),str(i)
    source=Source()
    train=build_loader(source,indices=[0,1],shuffle=True,num_workers=0,pin_memory=False)
    first={b[3][0]:b[0] for b in train}
    second={b[3][0]:b[0] for b in train}
    assert all(x.shape==(1,128,1024) for x in first.values())
    assert all(not torch.equal(first[k],second[k]) for k in first)
    assert train.dataset.epoch==2
    replay=build_loader(source,indices=[0,1],shuffle=True,num_workers=0,pin_memory=False)
    assert all(torch.equal(first[b[3][0]],b[0]) for b in replay)
    train.dataset.training=False
    assert all(b[0].shape[1]==256 for b in train)
    val=build_loader(source,indices=[2,3],shuffle=False,num_workers=0,pin_memory=False)
    assert all(b[0].shape[1]==256 for b in val)
    assert val.dataset.epoch==0


def test_main_epoch_training_checkpoint(tmp_path):
    from types import SimpleNamespace
    import train_epoch as entry
    from model.build import Pa_Model
    class Source:
        samples=pd.DataFrame({'pa_id':[str(i) for i in range(8)]})
        bags=[torch.randn(150,1024) for i in range(8)]
        labels=[np.arange(150)%8]*8
        seed=123
        def __getitem__(self,i):return self.bags[i],torch.tensor(i%2),torch.tensor(float(i+1)),str(i)
    torch.set_num_threads(1)
    source=Source();train=build_loader(source,[0,1,2,3],shuffle=True,pin_memory=False)
    val=build_loader(source,[4,5,6,7],pin_memory=False)
    model=Pa_Model(abmil_dropout=.25)
    args=SimpleNamespace(survival_loss='cox',cox_batch_size=3,num_epochs=2,patience=10,rank_weight=1.)
    entry.mainflow.train_path(model,train,val,torch.optim.Adam(model.parameters(),lr=1e-4,weight_decay=5e-4),args,torch.device('cpu'),0,tmp_path)
    assert (tmp_path/'best_model.pth').exists() and (tmp_path/'last_model.pth').exists()
    assert train.dataset.epoch==2 and val.dataset.epoch==0
    train.dataset.training=False
    metrics=entry.mainflow.evaluate_survival(model,train,val,torch.device('cpu'))
    assert all(np.isfinite(metrics[k]) for k in ('train_cindex','val_cindex','train_loss','val_loss'))
