"""One outer fold: seven inner candidates, three freshly initialized outer refits."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

BASE=Path(__file__).resolve().parent;ROOT=BASE.parents[2]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(BASE.parent/'uniform128_epoch_training'))
from train_uniform import PreparedDataset,sha256
from dataset import build_dataloader
from cox_utils import evaluate_survival
from protocol import CANDIDATES,partition,choose,fit_phase


def final_evaluation(model,source,train,val,directory,fold):
    records=[]
    def capture(module,inputs,output):
        p=output[2].reshape(-1).detach().double();n=p.numel();neff=1/p.square().sum()
        records.append(dict(mean_entropy=float(-(p*p.clamp_min(1e-300).log()).sum()/np.log(n)) if n>1 else 0.,
            mean_neff=float(neff),mean_q=float(neff/n),mean_top128_mass=float(p.topk(min(128,n)).values.sum())))
    train_loader=build_dataloader(source,train,batch_size=1,shuffle=False,num_workers=8,pin_memory=True)
    val_loader=build_dataloader(source,val,batch_size=1,shuffle=False,num_workers=8,pin_memory=True)
    model.eval();handle=model.register_forward_hook(capture)
    try:metrics=evaluate_survival(model,train_loader,val_loader,torch.device('cuda:0'),save_dir=directory)
    finally:handle.remove()
    if len(records)!=len(train)+len(val):raise RuntimeError('Final evaluation patient count differs')
    for split,values in [('train',records[:len(train)]),('val',records[len(train):])]:
        for key in values[0]:metrics[split+'_'+key]=float(np.mean([r[key] for r in values]))
    return dict(fold=fold,**metrics)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--fold',type=int,choices=range(5),required=True)
    fold=parser.parse_args().fold;destination=BASE/f'fold_{fold}'
    if destination.exists() and {p.name for p in destination.iterdir()}-{'training.log'}:raise FileExistsError('Refusing to overwrite fold')
    destination.mkdir(exist_ok=True);torch.set_num_threads(4)
    source=PreparedDataset(ROOT/'data/trident/features_uni_v1',roi_size=64,feature_dim=1024)
    if len(source)!=721 or source.samples.event.sum()!=251:raise ValueError('Original cohort differs')
    fit,stop,train,val=partition(source.samples,fold)
    private=destination/'private';private.mkdir()
    pd.DataFrame(dict(pa_id=source.samples.iloc[np.r_[fit,stop]].pa_id.to_numpy(),role=['fit']*len(fit)+['stop']*len(stop))).to_csv(private/'inner_roles.csv',index=False)
    pd.DataFrame([dict(fold=fold,fit=len(fit),stop=len(stop),outer_train=len(train),outer_val=len(val),
        fit_events=int(source.samples.iloc[fit].event.sum()),stop_events=int(source.samples.iloc[stop].event.sum()),
        outer_train_events=int(source.samples.iloc[train].event.sum()),outer_val_events=int(source.samples.iloc[val].event.sum()))]).to_csv(destination/'role_counts.csv',index=False)
    status=dict(status='running',completed_inner=0,completed_outer=0);status_path=destination/'status.json'
    results={};selected=[]
    try:
        status_path.write_text(json.dumps(status,indent=2))
        for config in CANDIDATES:
            print(f"\nFold {fold} | Inner FIT/STOP | {config['key']} | FIT={len(fit)} STOP={len(stop)}",flush=True)
            model,audit=fit_phase(source,fit,stop,config,destination/'inner'/config['key'],fold)
            results[config['key']]=audit;status['completed_inner']+=1
            status_path.write_text(json.dumps(status,indent=2));del model
        for family in ['baseline','entropy','neff']:
            config,inner=choose(CANDIDATES,results,family)
            chosen_epoch=inner['selected_epoch'];directory=destination/'outer'/family
            print(f"\nFold {fold} | Outer {family} | selected {config['key']}, Epoch={chosen_epoch}; Outer Val unused until final",flush=True)
            model,audit=fit_phase(source,train,None,config,directory,fold,epochs=chosen_epoch,patience=0)
            configuration=dict(family=family,selected=config,selected_epoch=chosen_epoch,selection_source='fixed_inner80_20_STOP',
                seed=123,feature_dim=1024,dropout=.25,cox_batch_size=32,lr=1e-4,weight_decay=5e-4,model_input_l2=False,
                sampling='Full WSI',outer_patience=0,max_inner_epochs=30,inner_patience=10,cosine_annealing=False)
            (directory/'run_config.yaml').write_text(yaml.safe_dump(configuration,sort_keys=True))
            metrics=final_evaluation(model,source,train,val,directory/'results',fold)
            metrics.update(family=family,selected_key=config['key'],selected_weight=config['weight'],selected_epoch=chosen_epoch)
            if config['mode']=='neff':metrics['selected_q_min']=config['q_min']
            pd.DataFrame([metrics]).to_csv(directory/'metrics.csv',index=False)
            selected.append(dict(fold=fold,family=family,**config,selected_epoch=chosen_epoch,inner_cindex=inner['selection_cindex']))
            pd.DataFrame(selected).to_csv(destination/'selection.csv',index=False)
            # Check own checkpoint on a training patient, without a second Outer Val forward.
            restored=type(model)(model_name='abmil',feature_dim=1024,abmil_dropout=.25).cuda().eval()
            restored.load_state_dict(torch.load(directory/'best_model.pth',weights_only=True,map_location='cuda'))
            with torch.no_grad():
                x=source.bags[train[0]].cuda()
                if not torch.equal(model(x)[0],restored(x)[0]):raise RuntimeError('Checkpoint reload mismatch')
            (directory/'completed.json').write_text(json.dumps(dict(completed=True,outer_val_passes=1,checkpoint_reload_exact=True),indent=2))
            status['completed_outer']+=1;status_path.write_text(json.dumps(status,indent=2));del model,restored
        status['status']='completed'
    except Exception as exc:status.update(status='failed',error=f'{type(exc).__name__}: {exc}');raise
    finally:status_path.write_text(json.dumps(status,indent=2))


if __name__=='__main__':main()
