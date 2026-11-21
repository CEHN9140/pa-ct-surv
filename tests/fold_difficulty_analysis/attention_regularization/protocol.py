"""Inner STOP selects settings/epochs; outer refits never select on Outer Val."""
import builtins
import hashlib
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.model_selection import train_test_split
from sksurv.metrics import concordance_index_censored

ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
import path_train as mainflow
from dataset import build_dataloader
from model.build import Pa_Model
from utils import seed_everything
from loss.survival_loss import cox_loss
from regularization import AttentionObjective

CANDIDATES=[dict(key='baseline',mode='baseline',weight=0.),
    *[dict(key=f'entropy_w{weight:g}',mode='entropy',weight=weight) for weight in [.01,.1]],
    *[dict(key=f'neff_q{q:g}_w{weight:g}',mode='neff',weight=weight,q_min=q) for q in [.05,.1] for weight in [1.,10.]]]


def partition(samples,fold):
    train=np.flatnonzero(samples.split.ne(fold));outer=np.flatnonzero(samples.split.eq(fold))
    fit,stop=train_test_split(train,test_size=.2,random_state=123,stratify=samples.iloc[train].event)
    if set(fit)&set(stop) or set(fit)&set(outer) or set(stop)&set(outer):raise ValueError('Patient-role overlap')
    if set(fit)|set(stop)!=set(train):raise ValueError('Inner partition differs')
    return np.asarray(fit),np.asarray(stop),train,outer


def choose(candidates,records,family):
    eligible=[item for item in candidates if item['mode'] in ['baseline',family]]
    def order(item):
        result=records[item['key']]
        return (-result['selection_cindex'],item['weight'],item.get('q_min',0.),result['selected_epoch'])
    selected=sorted(eligible,key=order)[0]
    return dict(selected),records[selected['key']]


def evaluate_group(model,source,indices):
    device=next(model.parameters()).device;was_training=model.training
    risk=[];entropy=[];ratios=[];effective=[];mass=[]
    try:
        with torch.random.fork_rng(devices=[device.index] if device.type=='cuda' else []),torch.no_grad():
            model.eval()
            for index in indices:
                output=model(source.bags[index].to(device));risk.append(float(output[0]))
                p=output[2].reshape(-1).double();n=p.numel();neff=1/p.square().sum()
                entropy.append(float(-(p*p.clamp_min(1e-300).log()).sum()/np.log(n)) if n>1 else 0.)
                effective.append(float(neff));ratios.append(float(neff/n));mass.append(float(p.topk(min(128,n)).values.sum()))
        frame=source.samples.iloc[indices];r=np.asarray(risk,np.float32);times=frame.time.to_numpy(np.float32);events=frame.event.to_numpy(bool)
        return dict(cox_loss=float(cox_loss(torch.from_numpy(r),torch.from_numpy(times),torch.from_numpy(events).float())),
            cindex=float(concordance_index_censored(events,times,r)[0]),mean_entropy=float(np.mean(entropy)),mean_neff=float(np.mean(effective)),
            mean_q=float(np.mean(ratios)),mean_top128_mass=float(np.mean(mass)),patients=len(indices),events=int(events.sum()))
    finally:model.train(was_training)


def fit_phase(source,fit,stop,config,directory,fold,device='cuda:0',epochs=30,patience=10):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    seed_everything(123)
    training=build_dataloader(source,indices=fit,batch_size=1,shuffle=True,num_workers=getattr(source,'workers',8),pin_memory=True)
    reference=fit if stop is None else stop
    validation=build_dataloader(source,indices=reference,batch_size=1,shuffle=False,num_workers=getattr(source,'workers',8),pin_memory=True)
    model=Pa_Model(feature_dim=1024,model_name='abmil',abmil_dropout=.25,l2_normalize=False).to(device)
    initial=hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for v in model.state_dict().values())).hexdigest()
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-4,weight_decay=5e-4)
    args=SimpleNamespace(survival_loss='cox',rank_weight=1.,cox_batch_size=32,num_epochs=epochs,patience=patience if stop is not None else 0)
    objective=AttentionObjective(model,config['mode'],config['weight'],config.get('q_min',.1))
    curves=[];updates=[0]
    def stepped(*unused):updates[0]+=1
    handle=optimizer.register_step_post_hook(stepped)
    def printed(*values,**kw):
        text=' '.join(map(str,values));match=re.match(r'Epoch (\d+)/',text)
        if stop is None:text=text.replace('Val Loss:','Train Eval Loss:').replace('Val C-index:','Train Eval C-index:')
        if stop is None and text.startswith(f'Fold {fold} best C-index:'):text=f'Outer fixed-budget fit complete: Epoch {epochs}; training-reference selection ignored'
        builtins.print(text,flush=True)
        if match:
            epoch=int(match[1]);parts=objective.components;objective.components=[]
            totals={k:float(np.mean([r[k] for r in parts])) for k in ['cox','penalty','total']}
            if sum(r['patients'] for r in parts)!=len(fit):raise RuntimeError('Epoch did not consume every training patient')
            active_fraction=sum(r['active_patients'] for r in parts)/len(fit)
            for split,indices in [('train',fit)]+([] if stop is None else [('stop',stop)]):
                row=dict(epoch=epoch,split=split,optimizer_updates=updates[0],**evaluate_group(model,source,indices),
                    train_batch_cox=totals['cox'],train_batch_penalty=totals['penalty'],train_batch_total=totals['total'],regularization_weight=config['weight'],active_training_fraction=active_fraction)
                curves.append(row)
            pd.DataFrame(curves).to_csv(directory/'epoch_metrics.csv',index=False)
            builtins.print(f"  Train Cox: {totals['cox']:.6f} | Attention penalty: {totals['penalty']:.6f} | Weight: {config['weight']:g} | Active bags: {active_fraction:.2%}",flush=True)
    try:
        with patch.object(mainflow,'print',printed,create=True):
            mainflow.train_path(model,training,validation,optimizer,args,torch.device(device),fold,directory,criterion=objective)
    finally:
        handle.remove();objective.close()
    metadata=yaml.safe_load((directory/'checkpoint_metadata.yaml').read_text())
    selected_epoch=int(metadata['best_epoch']) if stop is not None else epochs
    selection_cindex=float(metadata['best_val_cindex']) if stop is not None else None
    if stop is None:
        model.load_state_dict(torch.load(directory/'last_model.pth',weights_only=True,map_location=device))
        torch.save(model.state_dict(),directory/'best_model.pth')
        metadata=dict(fold=fold,best_epoch=epochs,last_epoch=epochs,selection_source='inner_STOP_only',outer_validation_used=False)
        (directory/'checkpoint_metadata.yaml').write_text(yaml.safe_dump(metadata))
    audit=dict(config=config,phase='inner' if stop is not None else 'outer',fold=fold,selected_epoch=selected_epoch,
        selection_cindex=selection_cindex,last_epoch=int(metadata['last_epoch']),optimizer_updates=updates[0],initial_state_sha256=initial,
        regularizer_active_patient_exposures=objective.active_exposures,patient_exposures=objective.patient_exposures,
        fit_patients=len(fit),stop_patients=len(stop) if stop is not None else 0,outer_validation_used_for_selection=False)
    (directory/'audit.json').write_text(json.dumps(audit,indent=2))
    return model,audit
