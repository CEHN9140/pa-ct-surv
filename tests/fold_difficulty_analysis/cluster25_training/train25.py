"""Cluster128 main-flow adapter with only the train sampling budget changed to25."""
import importlib.util,json,os,sys
from pathlib import Path
from unittest.mock import patch
import torch
import yaml

BASE=Path(__file__).resolve().parent
ROOT=BASE.parents[2]
CONTROL=BASE.parent/'cluster128_training'
spec=importlib.util.spec_from_file_location('cluster128_epoch_reference',CONTROL/'epoch_training/train_epoch.py')
reference=importlib.util.module_from_spec(spec);spec.loader.exec_module(reference)
mainflow=reference.mainflow

class Cluster25Dataset(reference.ClusterDataset):
    def __getitem__(self,position):
        index=self.indices[position]
        raw,event,time,patient=self.source[index]
        if self.training:
            chosen=reference.select_indices(self.source.labels[index],25,self.source.seed,'epoch_train',self.epoch,patient)
            raw=raw[chosen]
        return raw,event,time,patient

def build_loader(dataset,indices=None,batch_size=1,shuffle=False,num_workers=0,pin_memory=True):
    subset=Cluster25Dataset(dataset,range(len(dataset)) if indices is None else indices,shuffle)
    return reference.EpochLoader(reference.mainflow_loader(subset,batch_size=batch_size,shuffle=shuffle,num_workers=num_workers,pin_memory=pin_memory))

def main():
    torch.set_num_threads(4)
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    if (BASE/'results').exists() or (BASE/'checkpoints').exists():raise FileExistsError('Refusing overwrite')
    status=dict(status='preparing',pid=os.getpid(),completed_folds=0,protocol='mainflow_epoch_cv5')
    def write_status():(BASE/'status.json').write_text(json.dumps(status,indent=2))
    original_train=mainflow.train_path;original_eval=mainflow.evaluate_survival;original_config=mainflow.save_run_config
    original_cache=reference.load_cached_clusters
    hashes={}
    def readonly_cache(source,patient,raw,cache):
        if not (Path(cache)/f'{patient}.npz').exists():raise FileNotFoundError('Required existing cluster cache: '+str(patient))
        return original_cache(source,patient,raw,cache)
    def save_config(args,*a,**kw):
        settings=yaml.safe_load((CONTROL/'results/run_config.yaml').read_text())
        keys=['data_dir','feature_dim','ct_roi_size','pa_model','survival_loss','dropout','num_epochs','patience','cox_batch_size','lr','weight_decay','seed','num_workers','l2_normalize','attention_branches']
        assert all(getattr(args,k)==settings[k] for k in keys),'Settings differ from Cluster128'
        pinned=json.loads((CONTROL/'results/source_sha256.json').read_text())
        for filename,digest in pinned.items():
            if '/tests/' not in filename:assert reference.sha256(filename)==digest,filename
        original_config(args,*a,**kw,extra=dict(train_patch_sampling='cluster_balanced25',cluster_input_l2=True,model_input_l2=False,cluster_cache=str(CONTROL/'cache'),validation_patches='Full WSI',protocol='mainflow_epoch_cv5'))
        files=[Path(__file__),BASE/'report25.py',CONTROL/'epoch_training/train_epoch.py',CONTROL/'sampler.py',CONTROL/'cluster.py']+[Path(p) for p in pinned if '/tests/' not in p]
        hashes.update({str(p):reference.sha256(p) for p in files})
        (Path(args.results_root)/'source_sha256.json').write_text(json.dumps(hashes,indent=2))
    def train(*a,**kw):
        status.update(status='training',active_fold=int(a[6]));write_status();return original_train(*a,**kw)
    def evaluate(model,train_loader,val_loader,*a,**kw):
        train_loader.dataset.training=False
        result=original_eval(model,train_loader,val_loader,*a,**kw)
        status['completed_folds']+=1;write_status();return result
    try:
        write_status()
        with patch.object(reference,'load_cached_clusters',readonly_cache),patch.object(mainflow,'Path_Dataset',reference.PreparedDataset),patch.object(mainflow,'build_dataloader',build_loader),patch.object(mainflow,'train_path',train),patch.object(mainflow,'evaluate_survival',evaluate),patch.object(mainflow,'save_run_config',save_config):
            mainflow.main()
        assert status['completed_folds']==5
        assert all(reference.sha256(p)==h for p,h in hashes.items()),'Sources changed'
        from report25 import build
        build();status['status']='completed'
    except Exception as exc:
        status.update(status='failed',error=f'{type(exc).__name__}: {exc}');raise
    finally:write_status()

if __name__=='__main__':main()
