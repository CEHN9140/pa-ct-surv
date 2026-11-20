"""Original main-flow Epoch training plus RNG-neutral checkpoint diagnostics."""
import argparse
import builtins
import hashlib
import json
import re
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
import yaml

BASE=Path(__file__).resolve().parent;ROOT=BASE.parents[2]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(BASE.parent/'uniform128_epoch_training'))
import path_train as mainflow
from train_uniform import PreparedDataset,sha256
from diagnostics import evaluate_checkpoint


def diagnostic_train(model,train_loader,val_loader,optimizer,args,device,fold,checkpoint_dir,criterion=None):
    source=train_loader.dataset
    while hasattr(source,'dataset'):source=source.dataset
    directory=Path(args.results_root).parent/'diagnostics'/f'fold_{fold}'
    directory.mkdir(parents=True,exist_ok=True)
    source.samples[['pa_id','time','event','split']].to_csv(directory/'private_cohort.csv',index=False)
    initial=hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for v in model.state_dict().values())).hexdigest()
    state=dict(updates=0,last_epoch=0);curves=[]
    fixed={0,1,3,5,7,10}
    def snapshot(epoch,capture=False):
        rows=evaluate_checkpoint(model,source,fold,epoch,state['updates'],directory,capture)
        curves.extend(rows);pd.DataFrame(curves).to_csv(directory/'epoch_metrics.csv',index=False)
        train,val=rows
        builtins.print(f"Eval Epoch {epoch}/{args.num_epochs} | Train Loss: {train['cox_loss']:.4f} | Train C-index: {train['cindex']:.4f} | Val Loss: {val['cox_loss']:.4f} | Val C-index: {val['cindex']:.4f}",flush=True)
    def printed(*values,**kw):
        text=' '.join(map(str,values));kw.setdefault('flush',True);builtins.print(*values,**kw)
        match=re.match(r'Epoch (\d+)/',text)
        if match:
            epoch=int(match[1]);state['last_epoch']=epoch;snapshot(epoch,epoch in fixed)
    def updated(*unused):state['updates']+=1
    snapshot(0,True)
    handle=optimizer.register_step_post_hook(updated)
    try:
        with patch.object(mainflow,'print',printed,create=True):
            result=ORIGINAL_TRAIN(model,train_loader,val_loader,optimizer,args,device,fold,checkpoint_dir,criterion)
    finally:handle.remove()
    metadata=yaml.safe_load((Path(checkpoint_dir)/'checkpoint_metadata.yaml').read_text())
    best_epoch=int(metadata['best_epoch']);last=int(metadata['last_epoch'])
    for epoch,filename in [(last,'last_model.pth'),(best_epoch,'best_model.pth')]:
        snapshot_path=directory/f'private/epoch_{epoch:03d}.npz'
        with np.load(snapshot_path,allow_pickle=False) as saved:
            previous=saved['risk'].copy();has_features='pool512' in saved
        if not has_features:
            model.load_state_dict(torch.load(Path(checkpoint_dir)/filename,map_location=device,weights_only=True))
            updates=int(pd.DataFrame(curves).query('epoch==@epoch').optimizer_updates.iloc[0])
            evaluate_checkpoint(model,source,fold,epoch,updates,directory,True)
            with np.load(snapshot_path,allow_pickle=False) as saved:
                if not np.array_equal(previous,saved['risk']):raise RuntimeError('Reloaded epoch prediction differs')
    model.load_state_dict(torch.load(Path(checkpoint_dir)/'best_model.pth',map_location=device,weights_only=True))
    (directory/'audit.json').write_text(json.dumps(dict(initial_state_sha256=initial,optimizer_updates=state['updates'],last_epoch=last,best_epoch=best_epoch,
        saved_feature_epochs=sorted((fixed&{r['epoch'] for r in curves})|{best_epoch,last}),outer_val_used_for_early_stopping=True,train_patients=len(train_loader),val_patients=len(val_loader)),indent=2))
    return result


ORIGINAL_TRAIN=mainflow.train_path


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--batch',type=int,choices=[32,64,128],required=True)
    batch=parser.parse_args().batch;destination=BASE/f'bs{batch}'
    if (destination/'results').exists():raise FileExistsError('Refusing to overwrite a diagnostic run')
    destination.mkdir(exist_ok=True);torch.set_num_threads(4)
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    status=dict(status='running',batch=batch,completed_folds=0)
    original_eval=mainflow.evaluate_survival;original_config=mainflow.save_run_config
    def evaluated(*a,**kw):
        result=original_eval(*a,**kw);status['completed_folds']+=1
        (destination/'status.json').write_text(json.dumps(status,indent=2));return result
    def configured(args,*a,**kw):
        reference=BASE.parent/'full_patch_epoch_controls/validation/source_sha256.json'
        for filename,digest in json.loads(reference.read_text()).items():
            if sha256(filename)!=digest:raise ValueError(f'Original control source changed: {filename}')
        original_config(args,*a,**kw,extra=dict(diagnostics='fixed-model eval every epoch; feature snapshots0,1,3,5,7,10,last,best',
            training_budget='maximum30 patient passes; early-stop10; optimizer updates recorded, not matched by design',train_patch_sampling='none'))
    sys.argv=['path_train.py','--data_dir',str(ROOT/'data/trident/features_uni_v1'),'--ct_roi_size','64','--feature_dim','1024',
        '--pa_model','abmil','--survival_loss','cox','--dropout','0.25','--num_epochs','30','--patience','10',
        '--cox_batch_size',str(batch),'--lr','1e-4','--weight_decay','5e-4','--num_workers','8',
        '--checkpoint_root',str(destination/'checkpoints'),'--results_root',str(destination/'results')]
    try:
        (destination/'status.json').write_text(json.dumps(status,indent=2))
        with patch.object(mainflow,'Path_Dataset',PreparedDataset),patch.object(mainflow,'train_path',diagnostic_train),\
             patch.object(mainflow,'evaluate_survival',evaluated),patch.object(mainflow,'save_run_config',configured):
            mainflow.main()
        status['status']='completed'
    except Exception as exc:status.update(status='failed',error=f'{type(exc).__name__}: {exc}');raise
    finally:(destination/'status.json').write_text(json.dumps(status,indent=2))


if __name__=='__main__':main()
