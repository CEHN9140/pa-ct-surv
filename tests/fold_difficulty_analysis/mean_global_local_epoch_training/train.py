"""Independent main-flow Epoch comparison; only the pathology model changes."""
import hashlib
import json
import os
import sys
from pathlib import Path
from unittest.mock import patch

import torch
import yaml

BASE=Path(__file__).resolve().parent
ROOT=BASE.parents[2]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(BASE))
sys.path.insert(0,str(BASE.parent/'uniform128_epoch_training'))
import path_train as mainflow
from train_uniform import PreparedDataset,sha256
from model_local import MeanGlobalLocal


def make_model(model_name='abmil',feature_dim=1024,k=None,abmil_dropout=.25,attention_branches=1,time_bins=None,l2_normalize=False):
    if model_name!='abmil' or k is not None or attention_branches!=1 or time_bins is not None or l2_normalize:
        raise ValueError('This control uses raw UNI, one attention branch and scalar Cox risk')
    return MeanGlobalLocal(feature_dim,abmil_dropout)


def main():
    torch.set_num_threads(4)
    if not torch.cuda.is_available():raise RuntimeError('CUDA required for full training')
    if (BASE/'results').exists() or (BASE/'checkpoints').exists():raise FileExistsError('Refusing to overwrite outputs')
    original_train=mainflow.train_path;original_eval=mainflow.evaluate_survival;original_config=mainflow.save_run_config
    status=dict(status='preparing',pid=os.getpid(),completed_folds=0,protocol='mainflow_epoch_cv5')
    def configure(args,*a,**kw):
        ref=BASE.parent/'cluster128_training/results'
        settings=yaml.safe_load((ref/'run_config.yaml').read_text())
        keys=['data_dir','feature_dim','ct_roi_size','survival_loss','dropout','num_epochs','patience','cox_batch_size','lr','weight_decay','seed','num_workers','l2_normalize']
        if any(getattr(args,k)!=settings[k] for k in keys):raise ValueError('Control training settings differ')
        for filename,digest in json.loads((ref/'source_sha256.json').read_text()).items():
            if not Path(filename).is_relative_to(ROOT/'tests') and sha256(filename)!=digest:raise ValueError(f'Control source changed: {filename}')
        original_config(args,*a,**kw,extra=dict(pa_model='mean_global_local',global_branch='Full raw UNI mean1024',
            local_branch='Full raw UNI ->128; attention32; pool128 ->64',risk_head='concat1088 ->64 ->1',
            train_patch_sampling='none',model_input_l2=False,validation_patches='Full WSI',parameters=sum(p.numel() for p in MeanGlobalLocal(args.feature_dim,args.dropout).parameters())))
        files=[Path(__file__),BASE/'model_local.py',BASE.parent/'uniform128_epoch_training/train_uniform.py',ROOT/'path_train.py',ROOT/'dataset.py',ROOT/'model/abmil_cox.py',ROOT/'cox_utils.py',ROOT/'utils.py',ROOT/'loss/survival_loss.py',Path(args.data_dir)/'all_label_roi64.csv']
        (Path(args.results_root)/'source_sha256.json').write_text(json.dumps({str(p):sha256(p) for p in files},indent=2))
        print('Actual model: Mean global1024 + local MIL128/attention32/post64 -> concat1088 ->64 ->1 | Full WSI both branches',flush=True)
    def train(*a,**kw):
        status.update(status='training',active_fold=int(a[6]));(BASE/'status.json').write_text(json.dumps(status,indent=2))
        return original_train(*a,**kw)
    def evaluate(*a,**kw):
        result=original_eval(*a,**kw);status['completed_folds']+=1
        (BASE/'status.json').write_text(json.dumps(status,indent=2));return result
    try:
        (BASE/'status.json').write_text(json.dumps(status,indent=2))
        with patch.object(mainflow,'Path_Dataset',PreparedDataset),patch.object(mainflow,'Pa_Model',make_model),\
             patch.object(mainflow,'train_path',train),patch.object(mainflow,'evaluate_survival',evaluate),patch.object(mainflow,'save_run_config',configure):
            mainflow.main()
        status['status']='completed'
    except Exception as exc:
        status.update(status='failed',error=f'{type(exc).__name__}: {exc}');raise
    finally:(BASE/'status.json').write_text(json.dumps(status,indent=2))


if __name__=='__main__':main()
