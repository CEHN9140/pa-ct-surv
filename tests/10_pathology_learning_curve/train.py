"""Experiment10: fixed-validation, fixed-update UNI pathology learning curves."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('MPLCONFIGDIR','/tmp/pact_matplotlib')
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import torch
import yaml

from analysis import plot_learning_curve
from cox_utils import evaluate_survival
from dataset import build_dataloader,cox_batch_sizes
from learning_data import prepare_subsets,digest
from loss import cox_loss
from model.build import Pa_Model
from utils import seed_everything,save_run_config


def parse_args():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--label_csv',default=str(Path(__file__).resolve().parents[2]/'data/trident/features_uni_v1/all_label_roi64.csv'))
    parser.add_argument('--output_dir',default=str(Path(__file__).resolve().parent))
    parser.add_argument('--train_sizes',type=int,nargs='+',default=[128,256,384])
    parser.add_argument('--repeats',type=int,default=3)
    parser.add_argument('--max_updates',type=int,default=180)
    parser.add_argument('--eval_every',type=int,default=9)
    parser.add_argument('--cox_batch_size',type=int,default=64)
    parser.add_argument('--dropout',type=float,default=.25)
    parser.add_argument('--lr',type=float,default=1e-4)
    parser.add_argument('--weight_decay',type=float,default=5e-4)
    parser.add_argument('--num_workers',type=int,default=8)
    parser.add_argument('--seed',type=int,default=123)
    parser.add_argument('--prepare_only',action='store_true')
    args=parser.parse_args()
    if (args.cox_batch_size<2 or args.max_updates<1 or args.eval_every<1
            or args.max_updates%args.eval_every or args.num_workers<0 or args.repeats<1
            or any(n<args.cox_batch_size or n%args.cox_batch_size for n in args.train_sizes)):
        parser.error('Use positive fixed budgets, eval_every dividing max_updates, and subset sizes divisible by Cox batch')
    if not 0<=args.dropout<1 or not np.isfinite(args.lr)or args.lr<=0 or not np.isfinite(args.weight_decay)or args.weight_decay<0:
        parser.error('Invalid dropout, learning rate or weight decay')
    return args


def train_fixed_updates(model,train_loader,train_eval_loader,val_loader,optimizer,args,device,directory):
    directory.mkdir(parents=True,exist_ok=True)
    ends=set(np.cumsum(cox_batch_sizes(len(train_loader),args.cox_batch_size)).tolist())
    step=0;best=-np.inf;best_step=None;history=[];patients_seen=0;events_seen=0;checks=0
    gpu_devices=[device.index if device.index is not None else torch.cuda.current_device()]if device.type=='cuda'else[]
    optimizer.zero_grad();model.train()
    while step<args.max_updates:
        risks=[];times=[];events=[]
        for index,(features,event,time,case)in enumerate(train_loader,start=1):
            risks.append(model(features.to(device))[0]);events.append(event.to(device));times.append(time.to(device))
            if index not in ends:continue
            risk=torch.cat(risks);event=torch.cat(events);time=torch.cat(times)
            loss=cox_loss(risk,time,event);loss.backward();optimizer.step();optimizer.zero_grad()
            step+=1;patients_seen+=risk.numel();events_seen+=int(event.sum().item())
            risks=[];times=[];events=[]
            if step%args.eval_every==0:
                # Validation DataLoader iterations must not perturb training/dropout RNGs.
                with torch.random.fork_rng(devices=gpu_devices):
                    metrics=evaluate_survival(model,train_eval_loader,val_loader,device)
                checks+=1;history.append(dict(step=step,**metrics))
                print(f"Update {step}/{args.max_updates} | Train Loss: {metrics['train_loss']:.4f} | "
                      f"Train C-index: {metrics['train_cindex']:.4f} | Val Loss: {metrics['val_loss']:.4f} | "
                      f"Val C-index: {metrics['val_cindex']:.4f}",flush=True)
                if metrics['val_cindex']>best:
                    best=metrics['val_cindex'];best_step=step
                    torch.save(model.state_dict(),directory/'best_model.pth')
                model.train()
            if step==args.max_updates:break
    torch.save(model.state_dict(),directory/'last_model.pth')
    metadata=dict(optimizer_steps=step,validation_checks=checks,best_step=best_step,
                  training_patient_exposures=patients_seen,training_event_exposures=events_seen)
    (directory/'checkpoint_metadata.yaml').write_text(yaml.safe_dump(metadata))
    return pd.DataFrame(history),metadata


def main():
    args=parse_args();args.output_dir=str(Path(args.output_dir).resolve());args.label_csv=str(Path(args.label_csv).resolve())
    root=Path(args.output_dir);root.mkdir(parents=True,exist_ok=True)
    with (root/'.run.lock').open('a')as run_lock:
        try:fcntl.flock(run_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('Experiment10 is already running in this output directory')from None
        dataset,selected,validation,metadata=prepare_subsets(args.label_csv,root/'data',args.train_sizes,args.repeats)
        conditions=list(selected.groupby(['size_label','repeat','fold'],sort=False))
        print(f'Experiment10 | {len(dataset)} patients | {int(dataset.samples.event.sum())} events | {len(conditions)} fits',flush=True)
        if args.prepare_only:
            print(f'Frozen training subsets and validation patients: {root/"data"}',flush=True);return
        tag=(f'uni_abmil_cox_dropout{args.dropout}_updates{args.max_updates}_eval{args.eval_every}'
             f'_coxbs{args.cox_batch_size}_lr{args.lr:g}_wd{args.weight_decay:g}_seed{args.seed}')
        args.results_root=str(root/'results'/tag);args.checkpoint_root=str(root/'checkpoints'/tag)
        results=Path(args.results_root)
        if (results/'run_config.yaml').exists()or Path(args.checkpoint_root).exists():
            raise FileExistsError('This run already exists, possibly incomplete; use a new --output_dir to avoid overwriting')
        save_run_config(args,'pathology',tag,excluded=['prepare_only'],
                        extra=dict(experiment_id='10',pa_model='abmil',feature_encoder='uni_v1',feature_dim=1024,
                                   survival_loss='cox',cox_reduction='events',early_stopping=False,
                                   cohort_sha256=metadata['settings']['source_sha256'],
                                   subset_manifest_sha256=digest(root/'data/manifest.json'),
                                   evaluation_protocol='Internal CV: fixed validation patients; fixed updates and validation opportunities'))
        device=torch.device('cuda'if torch.cuda.is_available()else'cpu');rows=[]
        for (label,repeat,fold),group in conditions:
            indices=group.row_index.to_numpy(dtype=int)
            val_indices=validation.loc[validation.fold.eq(fold),'row_index'].to_numpy(dtype=int)
            name=f'{label}_repeat{repeat}';destination=results/name/f'fold_{fold}'
            checkpoint=Path(args.checkpoint_root)/name/f'fold_{fold}'
            print(f'\n{name} | Fold {fold} | Train {len(indices)} / Val {len(val_indices)} | '
                  f'Train events {int(dataset.samples.loc[indices,"event"].sum())}',flush=True)
            seed_everything(args.seed)
            model=Pa_Model(model_name='abmil',feature_dim=1024,abmil_dropout=args.dropout)
            fingerprint=hashlib.sha256()
            for key,value in model.state_dict().items():fingerprint.update(key.encode()+value.numpy().tobytes())
            initial_hash=fingerprint.hexdigest();model=model.to(device)
            optimizer=torch.optim.Adam(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
            train_loader=build_dataloader(dataset,indices,batch_size=1,shuffle=True,num_workers=args.num_workers)
            train_eval_loader=build_dataloader(dataset,indices,batch_size=1,num_workers=args.num_workers)
            val_loader=build_dataloader(dataset,val_indices,batch_size=1,num_workers=args.num_workers)
            history,training=train_fixed_updates(model,train_loader,train_eval_loader,val_loader,optimizer,args,device,checkpoint)
            destination.mkdir(parents=True,exist_ok=True);history.to_csv(destination/'update_metrics.csv',index=False)
            last=evaluate_survival(model,train_eval_loader,val_loader,device,save_dir=destination/'last')
            model.load_state_dict(torch.load(checkpoint/'best_model.pth',map_location=device,weights_only=True))
            best=evaluate_survival(model,train_eval_loader,val_loader,device,save_dir=destination)
            row={**best,**training,'fold':int(fold),'size_label':label,'repeat':int(repeat),'run_name':name,
                 'n_train':len(indices),'n_events_train':int(dataset.samples.loc[indices,'event'].sum()),
                 'n_val':len(val_indices),'initial_state_sha256':initial_hash}
            row.update({'last_'+key:value for key,value in last.items()if key in
                        ['train_loss','train_cindex','val_loss','val_cindex','gap','auc_36m','auc_60m']})
            rows.append(row)
            temporary=results/'metrics.csv.tmp';pd.DataFrame(rows).to_csv(temporary,index=False);temporary.replace(results/'metrics.csv')
            print(f"Completed {name} fold {fold} | Best Val: {best['val_cindex']:.4f} | "
                  f"Fixed-budget Val: {last['val_cindex']:.4f}",flush=True)
            del model,optimizer,train_loader,train_eval_loader,val_loader
        plot_learning_curve(pd.DataFrame(rows),results)
        print(f'Completed all {len(rows)} learning-curve fits.',flush=True)


if __name__=='__main__':main()
