"""Fixed warmup and paired branches using original training loop; Outer Val stays outside."""
from pathlib import Path
import builtins,hashlib,json,re,sys
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch

BASE=Path(__file__).resolve().parent;ROOT=BASE.parents[2]
sys.path[:0]=[str(ROOT),str(BASE.parent/'attention_regularization'),str(BASE.parent/'uniform128_epoch_training')]
import path_train
from dataset import build_dataloader
from model.build import Pa_Model
from utils import seed_everything
from protocol import partition,evaluate_group
from train_uniform import PreparedDataset
from freeze import configure,weight_hash,capture_rng,restore_rng,rng_hash,modules as module_groups,choose_budget

# Imported legacy helpers alter sys.path; keep this isolated experiment first.
sys.path.insert(0,str(BASE))
ARMS=['baseline','freeze_ab','freeze_c']


class OrderedLoader:
    def __init__(self,loader):self.loader=loader;self.dataset=loader.dataset;self.orders=[]
    def __len__(self):return len(self.loader)
    def __iter__(self):
        h=hashlib.sha256()
        for batch in self.loader:
            for patient in batch[3]:h.update(str(patient).encode()+b'\0')
            yield batch
        self.orders.append(h.hexdigest())


def fit(model,source,indices,stop,directory,fold,epochs,arm,start_rng=None):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True);device=next(model.parameters()).device
    configure(model,arm);frozen='AB' if arm=='freeze_ab' else 'C' if arm=='freeze_c' else ''
    before=weight_hash(model,frozen) if frozen else None
    loader=OrderedLoader(build_dataloader(source,indices,batch_size=1,shuffle=True,num_workers=0,pin_memory=True))
    reference=build_dataloader(source,indices if stop is None else stop,batch_size=1,shuffle=False,num_workers=0,pin_memory=True)
    optimizer=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=1e-4,weight_decay=5e-4)
    if optimizer.state:raise ValueError('Adam did not start empty')
    if start_rng is not None:restore_rng(start_rng)
    initial=weight_hash(model);start_state=rng_hash(capture_rng());curves=[]
    def printed(*values,**kw):
        text=' '.join(map(str,values));match=re.match(r'Epoch (\d+)/',text)
        if stop is None:text=text.replace('Val Loss:','TrainReference Loss:').replace('Val C-index:','TrainReference C-index:')
        if text.startswith(f'Fold {fold} best C-index:'):text='Segment complete; original loop best checkpoint is not used for fixed-budget fitting'
        if not match:builtins.print(text,flush=True)
        if match:
            epoch=int(match[1])
            for role,group in [('train',indices)]+([] if stop is None else [('stop',stop)]):curves.append(dict(epoch=epoch,split=role,**evaluate_group(model,source,group)))
            pd.DataFrame(curves).to_csv(directory/'eval_curves.csv',index=False)
            recent=[row for row in curves if row['epoch']==epoch];tr=recent[0]
            message=f"Epoch {epoch}/{epochs} | Train Eval Loss: {tr['cox_loss']:.4f} | Train Eval C-index: {tr['cindex']:.4f}"
            if stop is not None:
                va=recent[1];message+=f" | STOP Loss: {va['cox_loss']:.4f} | STOP C-index: {va['cindex']:.4f}"
            builtins.print(message,flush=True)
            if frozen and weight_hash(model,frozen)!=before:raise ValueError('Frozen parameters changed')
            if frozen and any(p.grad is not None for key,layer in module_groups(model).items() if key in frozen for m in layer for p in m.parameters()):raise ValueError('Frozen gradient unexpectedly present')
    args=SimpleNamespace(survival_loss='cox',cox_batch_size=32,rank_weight=1.,num_epochs=epochs,patience=0)
    if epochs:
        with patch.object(path_train,'print',printed,create=True):path_train.train_path(model,loader,reference,optimizer,args,device,fold,directory)
        model.load_state_dict(torch.load(directory/'last_model.pth',map_location=device,weights_only=True))
    else:torch.save(model.state_dict(),directory/'last_model.pth')
    audit=dict(arm=arm,epochs=epochs,initial_hash=initial,start_rng_hash=start_state,frozen_scope=frozen,frozen_unchanged=(not frozen or weight_hash(model,frozen)==before),
        frozen_initial_hash=before,frozen_final_hash=weight_hash(model,frozen) if frozen else None,optimizer_reset=True,optimizer_initial_state_entries=0,
        fit_patients=len(indices),stop_patients=0 if stop is None else len(stop),outer_val_used=False,order_hashes=loader.orders,
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad))
    (directory/'audit.json').write_text(json.dumps(audit,indent=2));return model,pd.DataFrame(curves),audit


def common_start(source,indices,stop,directory,fold,device='cuda:0'):
    seed_everything(123);model=Pa_Model(feature_dim=1024,abmil_dropout=.25,l2_normalize=False).to(device)
    model,_,_=fit(model,source,indices,stop,directory,fold,3,'baseline')
    checkpoint={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};state=capture_rng()
    return checkpoint,state


def branch(checkpoint,state,source,indices,stop,directory,fold,arm,epochs,device='cuda:0'):
    model=Pa_Model(feature_dim=1024,abmil_dropout=.25,l2_normalize=False).to(device);model.load_state_dict(checkpoint)
    return fit(model,source,indices,stop,directory,fold,epochs,arm,state)


def run_fold(fold):
    from cox_utils import evaluate_survival
    directory=BASE/f'fold_{fold}';directory.mkdir(exist_ok=True)
    if set(p.name for p in directory.iterdir())-{'training.log'}:raise FileExistsError('Existing fold')
    torch.set_num_threads(4);source=PreparedDataset(ROOT/'data/trident/features_uni_v1',roi_size=64,feature_dim=1024)
    if len(source)!=721 or source.samples.event.sum()!=251:raise ValueError('Original cohort differs')
    fit_ids,stop,train,val=partition(source.samples,fold);selections=[]
    private=directory/'private';private.mkdir()
    pd.DataFrame(dict(pa_id=source.samples.iloc[np.r_[fit_ids,stop]].pa_id.to_numpy(),role=['fit']*len(fit_ids)+['stop']*len(stop))).to_csv(private/'inner_roles.csv',index=False)
    checkpoint,state=common_start(source,fit_ids,stop,private/'inner/warmup',fold)
    selected={};branch_audits=[]
    for arm in ARMS:
        model=Pa_Model(feature_dim=1024,abmil_dropout=.25).cuda();model.load_state_dict(checkpoint)
        score0=evaluate_group(model,source,stop)['cindex'];del model
        print(f'Fold {fold} inner {arm}: common warmup3 +20 continuation Epochs',flush=True)
        model,curve,audit=branch(checkpoint,state,source,fit_ids,stop,private/'inner'/arm,fold,arm,20);branch_audits.append(audit)
        candidates=[(score0,0)]+[(float(r.cindex),int(r.epoch)) for r in curve[curve.split.eq('stop')].itertuples()]
        score,budget=choose_budget(candidates);selected[arm]=budget
        selections.append(dict(arm=arm,continuation_epochs=budget,inner_stop_cindex=score,score_at_fork=score0,warmup_epochs=3));del model
    if len({a['initial_hash'] for a in branch_audits})!=1 or len({a['start_rng_hash'] for a in branch_audits})!=1 or len({tuple(a['order_hashes']) for a in branch_audits})!=1:raise ValueError('Inner branch pairing failed')
    pd.DataFrame(selections).to_csv(directory/'selection.csv',index=False)
    checkpoint,state=common_start(source,train,None,private/'outer/warmup',fold);audits=[];metrics=[]
    for arm in ARMS:
        budget=selected[arm];print(f'Fold {fold} outer {arm}: fixed3+{budget} Epochs',flush=True)
        model,_,audit=branch(checkpoint,state,source,train,None,private/'outer'/arm,fold,arm,budget);audits.append(audit)
        train_loader=build_dataloader(source,train,batch_size=1,shuffle=False,num_workers=0,pin_memory=True)
        val_loader=build_dataloader(source,val,batch_size=1,shuffle=False,num_workers=0,pin_memory=True)
        metric=evaluate_survival(model,train_loader,val_loader,torch.device('cuda:0'),save_dir=private/'outer'/arm/'results')
        metrics.append(dict(fold=fold,arm=arm,continuation_epochs=budget,**metric));pd.DataFrame(metrics).to_csv(directory/'metrics.csv',index=False);del model
    for a in audits:
        if a['initial_hash']!=audits[0]['initial_hash'] or a['start_rng_hash']!=audits[0]['start_rng_hash']:raise ValueError('Outer branch init mismatch')
        for b in audits:
            n=min(a['epochs'],b['epochs'])
            if a['order_hashes'][:n]!=b['order_hashes'][:n]:raise ValueError('Outer patient order mismatch')
    (directory/'completed.json').write_text(json.dumps(dict(passed=True,fold=fold,outer_val_passes_per_arm=1,shared_inner_start=True,shared_outer_start=True),indent=2))


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--fold',type=int,choices=range(5),required=True);run_fold(p.parse_args().fold)
