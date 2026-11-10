"""Experiment09: fixed single-WSI CPTAC OS, ABMIL/MeanPool+Cox, internal five-fold CV."""
import argparse
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
from lifelines import KaplanMeierFitter
from lifelines.plotting import add_at_risk_counts
from sksurv.metrics import concordance_index_censored

from cptac_dataset import CPTACDataset,CPTACEagleDataset,file_hash
from meanpool_linear import MeanPoolLinear
from patient_cox_loss import cox_patient_loss
from dataset import build_dataloader,cox_batch_sizes,cv_splits
from loss import cox_loss
from model.build import Pa_Model
from path_train import train_path
from utils import save_cv_results,save_run_config,seed_everything
from visualization.survival_plots import fold_metrics


def parse_args():
    here=Path(__file__).resolve().parent
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data_dir',default=str(here/'data'))
    parser.add_argument('--output_dir',default=str(here))
    parser.add_argument('--feature_source',choices=['uni','uni_l2','keep','eagle_top25'],default='uni')
    parser.add_argument('--pa_model',choices=['abmil','meanpool','meanpool_linear'],default='abmil')
    parser.add_argument('--eagle_root',default='/home/gly001/cqj/EAGLE/test/cptac_luad')
    parser.add_argument('--dropout',type=float,default=.25)
    parser.add_argument('--num_epochs',type=int,default=30)
    parser.add_argument('--cox_batch_size',type=int,default=16)
    parser.add_argument('--cox_reduction',choices=['events','patients'],default='events')
    parser.add_argument('--lr',type=float,default=1e-4)
    parser.add_argument('--weight_decay',type=float,default=5e-4)
    parser.add_argument('--patience',type=int,default=10)
    parser.add_argument('--num_workers',type=int,default=8)
    parser.add_argument('--seed',type=int,default=123)
    parser.add_argument('--eval_only',action='store_true')
    args=parser.parse_args()
    if args.pa_model=='meanpool_linear'and args.feature_source!='eagle_top25':
        parser.error('meanpool_linear is the EAGLE top25 control; use --feature_source eagle_top25')
    if args.num_epochs<1 or args.cox_batch_size<2 or args.num_workers<0 or not 0<=args.dropout<1:
        parser.error('Invalid epochs, Cox batch, num_workers or dropout')
    if not np.isfinite(args.lr) or args.lr<=0 or not np.isfinite(args.weight_decay) or args.weight_decay<0:
        parser.error('lr must be finite and positive; weight_decay finite and nonnegative')
    return args


def run_tag(args):
    tag=f'cptac_luad_single_{args.feature_source}_{args.pa_model}_cox'
    if args.cox_reduction=='patients':tag+='_patientmean'
    if args.pa_model!='meanpool_linear':tag+=f'_dropout{args.dropout}'
    tag+=(f'_epochs{args.num_epochs}_coxbs{args.cox_batch_size}_lr{args.lr:g}'
         f'_wd{args.weight_decay:g}_seed{args.seed}')
    # Preserve existing patience=10 run paths, while isolating new early-stop settings.
    return tag if args.patience==10 else f'{tag}_patience{args.patience}'


def plot_os_km(val,metrics,path):
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(7,7));fitters=[]
    for group,name,color in [(1,'High risk','crimson'),(0,'Low risk','steelblue')]:
        part=val[val.risk_group_binary==group]
        if part.empty:continue
        fitter=KaplanMeierFitter().fit(part.OS_days,part.OS_event,label=f'{name} (n={len(part)})')
        fitter.plot_survival_function(ax=ax,ci_show=True,show_censors=True,color=color)
        fitters.append(fitter)
    ax.set(xlabel='Overall survival time (days)',ylabel='Overall survival probability',
           ylim=(0,1.02),title=f"log-rank p = {metrics['logrank_p']:.3g}")
    ax.grid(alpha=.2)
    if fitters:add_at_risk_counts(*fitters,ax=ax,rows_to_show=['At risk'])
    fig.subplots_adjust(bottom=.3,top=.86)
    fig.savefig(path,dpi=200,bbox_inches='tight');plt.close(fig)


def evaluate_fold(model,train_loader,val_loader,device,destination,criterion=cox_loss):
    model.eval();destination.mkdir(parents=True,exist_ok=True)
    frames,metrics=[],{}
    with torch.no_grad():
        for part,loader in [('train',train_loader),('val',val_loader)]:
            risks,times,events,ids=[],[],[],[]
            for features,event,time,case in loader:
                risks.append(model(features.to(device))[0].detach().cpu().reshape(-1))
                times.append(time.reshape(-1));events.append(event.reshape(-1));ids.extend(case)
            risk,time,event=torch.cat(risks),torch.cat(times),torch.cat(events)
            metrics[f'{part}_loss']=float(criterion(risk.to(device),time.to(device),event.to(device)))
            metrics[f'{part}_cindex']=float(concordance_index_censored(event.numpy().astype(bool),time.numpy(),risk.numpy())[0])
            frames.append(pd.DataFrame(dict(case_id=ids,OS_days=time.numpy(),OS_event=event.numpy(),risk_score=risk.numpy())))
    compatible=[]
    for frame in frames:
        internal=frame.rename(columns={'OS_days':'dfs.month','OS_event':'dfs.status'}).copy()
        internal['dfs.month']=internal['dfs.month'].astype(float)/(365.25/12)
        compatible.append(internal)
    grouped,extra=fold_metrics(*compatible,eval_times=[24.,36.])
    frames[1]['risk_group_binary']=grouped.risk_group_binary.to_numpy()
    for part,frame in zip(['train','val'],frames):frame.to_csv(destination/f'{part}_predictions.csv',index=False)
    plot_os_km(frames[1],extra,destination/'km_high_vs_low.png')
    notes={k:v for k,v in extra.items()if k.endswith('_status')and v!='ok'}
    notes['evaluation_protocol']='Internal validation used for early stopping; not independent test results'
    (destination/'evaluation_notes.json').write_text(json.dumps(notes,ensure_ascii=False,indent=2)+'\n')
    metrics.update({k:v for k,v in extra.items()if not k.endswith('_status')})
    print('\nExtra survival metrics:')
    for key in ['n_val','n_event_val','n_high_risk','n_low_risk','auc_24m','auc_36m',
                'logrank_p','HR_high_vs_low','HR_95CI_lower','HR_95CI_upper','cox_p']:
        value=metrics[key]
        print(f'{key}: {value:.6f}'if isinstance(value,float)else f'{key}: {value}')
    return metrics


def main():
    args=parse_args()
    is_eagle=args.feature_source=='eagle_top25'
    is_linear=args.pa_model=='meanpool_linear'
    patient_mean=args.cox_reduction=='patients'
    criterion=cox_patient_loss if patient_mean else cox_loss
    if is_eagle:
        args.eagle_root=str(Path(args.eagle_root).resolve())
        dataset=CPTACEagleDataset(args.data_dir,args.eagle_root)
    else:dataset=CPTACDataset(args.data_dir,args.feature_source)
    args.data_dir=str(Path(args.data_dir).resolve());args.output_dir=str(Path(args.output_dir).resolve())
    args.survival_loss,args.feature_dim='cox',dataset.feature_dim
    tag=run_tag(args)
    args.checkpoint_root=str(Path(args.output_dir)/'checkpoints'/tag)
    args.results_root=str(Path(args.output_dir)/'results'/tag)
    normalization={'uni':'none','uni_l2':'per_patch_l2','keep':'encoder_native','eagle_top25':'none'}[args.feature_source]
    signature_key='eagle_top25'if is_eagle else 'keep'if args.feature_source=='keep'else'uni'
    encoder='Virchow2-CLS'if is_eagle else 'keep'if args.feature_source=='keep'else'uni_v1'
    extra=dict(experiment_id='09',endpoint='OS',time_unit='days',feature_encoder=encoder,
               normalization=normalization,selection_method=dataset.metadata['selection_method'],
               evaluation_protocol=dataset.metadata['evaluation_protocol'],
               cohort_sha256=dataset.metadata['cohort_sha256'],
               feature_signature=dataset.metadata['feature_signatures'][signature_key],
               source_hashes=dataset.metadata['source_hashes'],auc_months=[24.,36.],auc_days_per_month=365.25/12)
    if args.feature_source=='uni_l2':extra['normalization_eps']=1e-12
    snapshot=(dataset.samples.to_csv(index=False).encode()if is_eagle else
              (Path(args.data_dir)/'all_label.csv').read_bytes())
    if is_eagle:
        extra.update(patch_count=25,patch_selection='Frozen EAGLE CHIEF Top25 per selected WSI',
                     batch_mode='patient_batch',patient_batch_size=args.cox_batch_size,merge_tail=True,
                     data_snapshot_sha256=hashlib.sha256(snapshot).hexdigest())
    if is_linear:extra.update(risk_head='linear',trainable_parameters=args.feature_dim+1)
    if patient_mean:
        extra.update(training_loss_denominator='actual_risk_set_patient_count',
                     evaluation_loss_denominator='evaluated_cohort_patient_count')
    results=Path(args.results_root);config_path=results/'run_config.yaml'
    if config_path.is_file():
        saved=yaml.safe_load(config_path.read_text())
        for key in ['data_dir','feature_source','feature_dim','pa_model','survival_loss','dropout','cox_batch_size',
                    'seed','patience','lr','weight_decay','cox_reduction']:
            if is_linear and key=='dropout':continue
            previous=saved.get(key,'events'if key=='cox_reduction'else None)
            if previous!=getattr(args,key):raise ValueError(f'Existing run setting differs: {key}')
        for key,value in extra.items():
            if saved.get(key)!=value:raise ValueError(f'Existing run provenance differs: {key}')
    if args.eval_only:
        if not config_path.is_file()or not(results/'data.csv').is_file():
            raise FileNotFoundError('Original run configuration or data snapshot missing')
        if file_hash(results/'data.csv')!=extra.get('data_snapshot_sha256',extra['cohort_sha256']):
            raise ValueError('Original selection/folds snapshot differs')
        for fold in range(5):
            p=Path(args.checkpoint_root)/f'fold_{fold}/best_model.pth'
            if not p.is_file():raise FileNotFoundError(p)
    else:
        excluded=['eval_only']+([]if is_eagle else['eagle_root'])+(['dropout']if is_linear else[])
        if not patient_mean:excluded.append('cox_reduction')
        save_run_config(args,'pathology',tag,excluded=excluded,extra=extra)
        (results/'data.csv').write_bytes(snapshot)
    seed_everything(args.seed)
    device=torch.device('cuda'if torch.cuda.is_available()else'cpu')
    print(f'Experiment09 | {args.feature_source}: {args.feature_dim}-D | {args.pa_model}+Cox | {device}',flush=True)
    print(f'Patients: {len(dataset)} | OS events: {int(dataset.samples.event.sum())} | One fixed WSI per patient',flush=True)
    print('Protocol: internal five-fold validation with early stopping; NOT an independent benchmark test',flush=True)
    print(f'Normalization: {normalization} | Training time units: days',flush=True)
    if patient_mean:print('Cox loss denominator: actual patient count (including the merged tail)',flush=True)
    patient_batch=args.cox_batch_size if is_eagle else 1
    train_args=argparse.Namespace(**vars(args))if is_eagle else args
    if is_eagle:
        # One loader iteration is already one joint patient Cox batch.
        train_args.cox_batch_size=1
        print(f'Patient batch size: {patient_batch} | No cross-batch gradient accumulation',flush=True)
    fold_results=[]
    for fold,(train_indices,val_indices)in enumerate(cv_splits(dataset.samples)):
        print(f'\n{"="*50}\nFold {fold+1}/5\n{"="*50}',flush=True)
        seed_everything(args.seed)
        train_loader=build_dataloader(dataset,train_indices,batch_size=patient_batch,shuffle=True,
                                     merge_tail=is_eagle,num_workers=args.num_workers)
        val_loader=build_dataloader(dataset,val_indices,batch_size=patient_batch,num_workers=args.num_workers)
        print(f'Cox risk-set sizes (tail merged): {cox_batch_sizes(len(train_indices),args.cox_batch_size)}',flush=True)
        directory=Path(args.checkpoint_root)/f'fold_{fold}';directory.mkdir(parents=True,exist_ok=True)
        model=(MeanPoolLinear(args.feature_dim)if is_linear else
               Pa_Model(model_name=args.pa_model,feature_dim=args.feature_dim,abmil_dropout=args.dropout)).to(device)
        if args.eval_only:model.load_state_dict(torch.load(directory/'best_model.pth',map_location=device,weights_only=True))
        else:
            optimizer=torch.optim.Adam(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
            model=train_path(model,train_loader,val_loader,optimizer,train_args,device,fold,directory,
                             criterion=criterion if patient_mean else None)
        metrics=evaluate_fold(model,train_loader,val_loader,device,results/f'fold_{fold}',criterion=criterion)
        print(f"Fold {fold} | Train Loss: {metrics['train_loss']:.4f} | Train C-index: {metrics['train_cindex']:.4f} | "
              f"Val Loss: {metrics['val_loss']:.4f} | Val C-index: {metrics['val_cindex']:.4f}")
        fold_results.append(dict(fold=fold,**metrics))
    save_cv_results(fold_results,results)


if __name__=='__main__':main()
