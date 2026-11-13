"""PA-only controls for residual teachers; matched initialization and training."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import torch
import yaml
from torch.nn import functional as F
from sksurv.metrics import concordance_index_censored

from dataset import Pa_CT_Dataset,cv_splits
from residual_teacher import ResidualTeacher,pair_changes
from residual_train import fit_teacher,ordered_teacher,frame_objective,source_partitions
from risk_calibration import paired_comparison
from utils import save_run_config,save_cv_results
from visualization.survival_plots import analyze_fold


class MatchedPathologyControl(ResidualTeacher):
    def __init__(self,ct_pretrained_path=None,dropout=.25):
        # Construct identically before removing CT: preserve BOTH PA weights and
        # post-initialization RNG state, which controls shuffle/dropout sequences.
        super().__init__(ct_pretrained_path=ct_pretrained_path,dropout=dropout)
        self.ct_backbone=None
        self.ct_delta=None
        self.initialization_description='same new PA weights/RNG as residual teacher; CT removed; no downstream checkpoint'

    def forward(self,ct,pa):
        if pa.ndim!=3 or pa.shape[1:]!=(25,1280):raise ValueError('Expected pathology [B,25,1280]')
        path=self.pa_branch(F.normalize(pa,p=2,dim=-1,eps=1e-12))[0]
        return path,torch.zeros_like(path),path


def export_control(name,fold,frames,teacher,epochs,args):
    destination=Path(args.results_root)/name/f'fold_{fold}'
    destination.mkdir(parents=True)
    frames['train'].to_csv(destination/'train_predictions.csv',index=False)
    metrics=analyze_fold(frames['train'],frames['val'],destination)
    for split,frame in frames.items():
        values=frame_objective(frame,args,0.)
        metrics.update({split+'_'+key:value for key,value in values.items()})
        metrics[split+'_loss']=values['total']
    val=frames['val'];reference=teacher.set_index('case_id').loc[val.case_id]
    np.testing.assert_allclose(reference['dfs.month'],val['dfs.month'])
    np.testing.assert_array_equal(reference['dfs.status'],val['dfs.status'])
    pairs=pair_changes(val['dfs.month'],val['dfs.status'],val.risk_score,reference.risk_score)
    pairs['comparison']='fixed inner-selected residual teacher minus freshly trained matched PA-only control'
    pairs['control_epochs']=int(epochs)
    pairs['interpretation']='Different independently optimized models; not a pure inference removal or causal CT estimate.'
    (destination/'pair_changes.json').write_text(json.dumps(pairs,indent=2))
    combined=val[['case_id','dfs.month','dfs.status']].copy()
    combined['risk_score']=reference.risk_score.to_numpy()
    combined['matched_pathology']=val.risk_score.to_numpy()
    print(f'{name} fold{fold} | PA Val C-index={metrics["val_cindex"]:.4f} | '
          f'Teacher={pairs["teacher_cindex"]:.4f} | Delta={pairs["delta_cindex"]:+.4f}',flush=True)
    return dict(fold=fold,**metrics),combined,pairs


def run(args):
    teacher=Path(args.teacher_run)
    config=yaml.safe_load((teacher/'run_config.yaml').read_text())
    if not (teacher/'completed.json').is_file() or config['fusion_type']!='residual_teacher_nested_cv':
        raise ValueError('Require completed residual teacher')
    args.data_dir=config['data_dir'];args.ct_pretrained_path=config['ct_pretrained_path']
    for key in ['seed','lr','ct_backbone_lr','weight_decay','dropout','cox_batch_size','num_epochs','patience','lambda_pa']:
        setattr(args,key,config[key])
    args.cohort_sha256=hashlib.sha256((Path(args.data_dir)/'all_label_roi64.csv').read_bytes()).hexdigest()
    args.public_pretrained_sha256=hashlib.sha256(Path(args.ct_pretrained_path).read_bytes()).hexdigest()
    if args.cohort_sha256!=config['cohort_sha256'] or args.public_pretrained_sha256!=config['public_pretrained_sha256']:
        raise ValueError('Cohort or configured public CT initialization weights changed')
    args.teacher_config_sha256=hashlib.sha256((teacher/'run_config.yaml').read_bytes()).hexdigest()
    nested=Path(config['source_run'])
    if hashlib.sha256((nested/'run_config.yaml').read_bytes()).hexdigest()!=config['source_config_sha256']:
        raise ValueError('Original nested split source changed')
    for p in [Path(args.results_root),Path(args.checkpoint_root)]:
        if p.exists():raise FileExistsError(f'Refusing to overwrite {p}')
    dataset=Pa_CT_Dataset(args.data_dir,roi_size=64,feature_dim=1280)
    if len(dataset)!=721 or not dataset.samples.pa_id.is_unique or not dataset.samples.ct_id.is_unique:
        raise ValueError('Expected original721 unique PA/CT pairs')
    device=torch.device(args.device)
    if device.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA requested but unavailable')
    torch.set_num_threads(4)
    args.fusion_type='matched_pathology_control';args.pathology_patch_l2=True
    args.initialization='teacher-matched new PA weights and RNG; CT removed before optimization; no downstream checkpoint'
    args.public_ct_loading_role='initialization RNG parity only; CT modules then removed'
    args.objective='(1+lambda_pa)*Cox(pathology); CT absent'
    save_run_config(args,'pathology','residual_pathology_control',excluded=['ct_backbone_lr'])
    names=['protocol_matched','teacher_epoch_matched']
    metrics={name:[] for name in names};comparisons={name:[] for name in names};pairs={name:[] for name in names}
    print('Matched PA-only control: same initialization/RNG, splits, dropout, Cox32 and1.3*Cox; no CT forward.',flush=True)
    for fold,(training,validation) in enumerate(cv_splits(dataset.samples)):
        if fold not in args.outer_folds:continue
        partitions=source_partitions(dataset.samples,training,validation,nested/f'fold_{fold}/inner_split_roles.csv')
        inner=[]
        for number,fit,stop,held in partitions:
            print(f'\nPA control | Outer{fold} | Inner{number} | fit={len(fit)}, stop={len(stop)}, held={len(held)}',flush=True)
            model,metadata=fit_teacher(dataset,fit,stop,args,device,
                Path(args.checkpoint_root)/f'fold_{fold}/inner_{number}',0.,model_factory=MatchedPathologyControl)
            frame=ordered_teacher(model,dataset,held,args,device)
            score=concordance_index_censored(frame['dfs.status'].astype(bool),frame['dfs.month'],frame.risk_score)[0]
            inner.append(dict(inner_fold=number,selected_epoch=metadata['selected_epoch'],held_cindex=float(score)))
            destination=Path(args.results_root)/f'inner_oof/fold_{fold}/inner_{number}'
            destination.mkdir(parents=True);frame.to_csv(destination/'val_predictions.csv',index=False)
            del model
            if device.type=='cuda':torch.cuda.empty_cache()
        selection=json.loads((teacher/f'fold_{fold}/selection.json').read_text())['primary']
        epochs=dict(protocol_matched=int(np.median([x['selected_epoch'] for x in inner])),
                    teacher_epoch_matched=int(selection['epochs']))
        provenance=Path(args.results_root)/f'fold_{fold}';provenance.mkdir()
        schedules=dict(epochs=epochs,inner_models=inner,teacher_inner_selection=selection,
            source_teacher_checkpoint=str(Path(config['checkpoint_root'])/selection['name']/f'fold_{fold}/outer/best_model.pth'),
            policy='Schedules fixed from inner stop or original inner-selected teacher before any outer control predictions; '
                   'never load a teacher/unimodal checkpoint into the control.')
        (provenance/'selection.json').write_text(json.dumps(schedules,indent=2))
        fixed_teacher=pd.read_csv(teacher/f'inner_selected/fold_{fold}/val_predictions.csv',dtype={'case_id':str})
        cached={}
        for name,epoch in epochs.items():
            if epoch not in cached:
                model,_=fit_teacher(dataset,training,None,args,device,
                    Path(args.checkpoint_root)/f'fold_{fold}/outer_epochs{epoch}',0.,fixed_epochs=epoch,
                    model_factory=MatchedPathologyControl)
                frames={split:ordered_teacher(model,dataset,indices,args,device)
                        for split,indices in [('train',training),('val',validation)]}
                cached[epoch]=frames
                del model
                if device.type=='cuda':torch.cuda.empty_cache()
            row,combined,changes=export_control(name,fold,cached[epoch],fixed_teacher,epoch,args)
            metrics[name].append(row);comparisons[name].append(combined);pairs[name].append(dict(fold=fold,**changes))
        (provenance/'completed.json').write_text(json.dumps(schedules,indent=2))
        for name in names:pd.DataFrame(metrics[name]).to_csv(Path(args.results_root)/name/'partial_metrics.csv',index=False)
    full=sorted(args.outer_folds)==list(range(5))
    if full:
        for name in names:
            root=Path(args.results_root)/name
            save_cv_results(metrics[name],root)
            comparison=paired_comparison(comparisons[name],seed=args.seed,reference_columns={'matched_pathology':'matched_pathology'})
            (root/'teacher_minus_control.json').write_text(json.dumps(comparison,indent=2))
            pd.DataFrame(pairs[name]).to_csv(root/'pair_changes_summary.csv',index=False)
    (Path(args.results_root)/'completed.json').write_text(json.dumps(dict(models=names,folds=args.outer_folds,complete_fivefold=full,
        no_source_model_weights_loaded=True,source_teacher_unchanged=True),indent=2))
    print('Matched pathology controls complete.',flush=True)


if __name__=='__main__':
    experiment=Path(__file__).resolve().parent
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--teacher_run',default=str(experiment/'results/residual_teacher'))
    parser.add_argument('--results_root',default=str(experiment/'results/residual_pathology_control'))
    parser.add_argument('--checkpoint_root',default=str(experiment/'checkpoints/residual_pathology_control'))
    parser.add_argument('--num_workers',type=int,default=8)
    parser.add_argument('--outer_folds',type=int,nargs='+',default=list(range(5)))
    parser.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    args=parser.parse_args()
    if args.num_workers<0 or not args.outer_folds or len(set(args.outer_folds))!=len(args.outer_folds) or not set(args.outer_folds).issubset(range(5)):
        parser.error('Invalid workers or outer folds')
    run(args)
