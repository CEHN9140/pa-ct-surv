import sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).parent))
from model_local import MeanGlobalLocal


def test_forward_mean_attention_and_invariance():
    torch.set_num_threads(1);torch.manual_seed(123)
    model=MeanGlobalLocal(dropout=.25).eval()
    x=torch.randn(2,131,1024);raw=x.clone()
    risk,z,attention=model(x)
    assert risk.shape==(2,) and z.shape==(2,1088) and attention.shape==(2,131,1)
    torch.testing.assert_close(z[:,:1024],x.mean(1))
    torch.testing.assert_close(attention.sum(1),torch.ones(2,1))
    torch.testing.assert_close(model(x[:,torch.randperm(131)])[0],risk,atol=1e-6,rtol=1e-5)
    assert torch.equal(x,raw)
    assert model(x[0])[0].shape==(1,)
    assert sum(p.numel() for p in model.parameters())==213378


def test_cox_gradients_reach_both_paths():
    from loss.survival_loss import cox_loss
    model=MeanGlobalLocal(dropout=.25).train()
    x=torch.randn(4,150,1024)
    risk,_,_=model(x)
    cox_loss(risk,torch.tensor([1.,2.,3.,4.]),torch.tensor([1.,0.,1.,0.])).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    # Cox and softmax are invariant to a common offset; bias gradients may be zero.
    assert all(p.grad.abs().sum()>0 for n,p in model.named_parameters() if n.endswith('weight'))
    assert model.risk_head[0].weight.grad[:,:1024].abs().sum()>0
    assert model.risk_head[0].weight.grad[:,1024:].abs().sum()>0


def test_mainflow_training_checkpoint_and_eval(tmp_path):
    from types import SimpleNamespace
    from torch.utils.data import DataLoader,TensorDataset
    sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
    import path_train
    torch.manual_seed(123)
    model=MeanGlobalLocal()
    def data():
        return TensorDataset(torch.randn(8,64,1024),torch.tensor([1,0,1,0,1,0,1,0]),torch.arange(1,9).float(),torch.arange(8))
    train=DataLoader(data(),batch_size=1,shuffle=True);val=DataLoader(data(),batch_size=1)
    args=SimpleNamespace(survival_loss='cox',cox_batch_size=3,num_epochs=2,patience=10,rank_weight=1.)
    trained=path_train.train_path(model,train,val,torch.optim.Adam(model.parameters(),lr=1e-4,weight_decay=5e-4),args,torch.device('cpu'),0,tmp_path)
    assert (tmp_path/'best_model.pth').exists() and (tmp_path/'last_model.pth').exists()
    restored=MeanGlobalLocal().eval()
    restored.load_state_dict(torch.load(tmp_path/'best_model.pth',weights_only=True))
    trained.eval();x=next(iter(val))[0]
    torch.testing.assert_close(trained(x)[0],restored(x)[0],rtol=0,atol=0)
