"""Frozen Stage6 ABMIL: Full vs random/top/bottom K, without training."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sksurv.metrics import concordance_index_censored
import torch

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from dataset import load_path_feature
from model.abmil_cox import ABMIL
from utils import seed_everything


def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1<<20),b''): digest.update(chunk)
    return digest.hexdigest()


def select_indices(logits,strategy,k,seed,case_id,repeat):
    if k<1 or strategy not in ('top','bottom','random'): raise ValueError('Invalid strategy or K')
    n=logits.numel()
    if n<1 or not torch.isfinite(logits).all(): raise ValueError('Invalid patch scores')
    if n<=k: return torch.arange(n,device=logits.device)
    if strategy=='random':
        words=np.frombuffer(hashlib.sha256(str(case_id).encode()).digest()[:16],dtype='<u4').tolist()
        rng=np.random.default_rng(np.random.SeedSequence([seed,repeat,*words]))
        indices=torch.as_tensor(rng.permutation(n)[:k].copy(),device=logits.device)
    else:
        indices=torch.argsort(logits,descending=strategy=='top',stable=True)[:k]
    return indices.sort().values


@torch.inference_mode()
def subset_risk(model,h,logits,indices):
    if model.training: raise ValueError('Subset inference requires model.eval()')
    pooled,_=model.pool_attention(h[:,indices],logits[:,indices])
    return model.risk_head(model.post_compression(pooled)).item()


def attention_stats(logits,ks):
    values=logits.detach().double().flatten()
    loga=torch.log_softmax(values,dim=0);a=loga.exp();n=len(a)
    entropy=float(-(a*loga).sum())
    neff=float(a.square().sum().reciprocal())
    row=dict(patch_count=n,normalized_entropy=entropy/np.log(n) if n>1 else 0.,
             effective_patches=neff,effective_ratio=neff/n)
    ranked=a.sort(descending=True).values
    for k in ks: row[f'top_mass_{k}']=float(ranked[:k].sum())
    return row


def fold_metrics(predictions):
    full=predictions.loc[predictions.strategy.eq('full')].set_index('case_id')
    rows=[]
    for (strategy,k,repeat),group in predictions.groupby(['strategy','k','repeat'],sort=False):
        original=full.loc[group.case_id,'risk'].to_numpy(np.float32)
        risks=group.risk.to_numpy(np.float32)
        ci=concordance_index_censored(group.event.astype(bool),group.time,risks)[0]
        constant=np.ptp(original)==0 or np.ptp(risks)==0
        rho=np.nan if constant else float(spearmanr(original,risks).statistic)
        rows.append(dict(strategy=strategy,k=int(k),repeat=int(repeat),cindex=float(ci),spearman=rho,
                         spearman_status='constant_risk' if constant else 'ok',n_val=len(group),
                         mean_kept=float(group.n_kept.mean()),full_retained_fraction=float(group.n_kept.eq(group.n_patches).mean())))
    return pd.DataFrame(rows)


@torch.inference_mode()
def run_fold(samples,fold,reference,output,device,ks,repeats,seed,feature_hashes):
    private=Path(output)/'private';private.mkdir(parents=True,exist_ok=True)
    source=Path(reference)/f'fold_{fold}/private/outer'
    checkpoint=source/'model.pth'
    checkpoint_hash=sha256(checkpoint)
    model=ABMIL(in_dim=1024,dropout=.25).to(device)
    state=torch.load(checkpoint,map_location=device,weights_only=True)
    model.load_state_dict(state['model'],strict=True);model.eval();model.requires_grad_(False)
    before={name:value.clone() for name,value in model.state_dict().items()}
    patients=samples.loc[samples.split.eq(fold)]
    saved=pd.read_csv(source/'val_predictions.csv',dtype={'case_id':str})
    if saved.case_id.duplicated().any() or set(saved.case_id)!=set(patients.pa_id): raise ValueError('Saved Full cohort differs')
    saved=saved.set_index('case_id').loc[patients.pa_id]
    if not np.array_equal(saved['dfs.month'].to_numpy(np.float32),patients.time.to_numpy(np.float32)) \
            or not np.array_equal(saved['dfs.status'].to_numpy(),patients.event.to_numpy()): raise ValueError('Saved Full labels differ')
    records,concentration=[],[]
    max_direct_error=0.
    for index,patient in enumerate(patients.itertuples()):
        if feature_hashes.get(patient.pa_path)!=sha256(patient.pa_path): raise ValueError('UNI feature checksum differs from Stage6')
        x=load_path_feature(patient.pa_path,1024).to(device)
        risk,_,_,h=model(x,return_patch_features=True)
        logits=model.attention(h);full=risk.item();n=len(x)
        if not np.isclose(full,np.float32(saved.loc[patient.pa_id,'risk_score']),atol=2e-6,rtol=2e-6):
            raise RuntimeError('Full risk replay mismatch before subset diagnosis')
        concentration.append(dict(case_id=patient.pa_id,fold=fold,**attention_stats(logits,ks)))
        base=dict(case_id=patient.pa_id,fold=fold,time=patient.time,event=patient.event,n_patches=n)
        records.append(dict(**base,strategy='full',k=0,repeat=-1,n_kept=n,risk=full))
        for strategy in ('top','bottom','random'):
            for repeat in (range(repeats) if strategy=='random' else [-1]):
                for k in ks:
                    chosen=select_indices(logits[0,:,0],strategy,k,seed,patient.pa_id,repeat)
                    result=full if k>=n else subset_risk(model,h,logits,chosen)
                    # Real first-patient check in each fold, in addition to synthetic CPU/GPU tests.
                    if index==0 and repeat in (-1,0):
                        direct=model(x[chosen])[0].item()
                        max_direct_error=max(max_direct_error,abs(direct-result))
                        if not np.isclose(direct,result,atol=2e-6,rtol=2e-6): raise RuntimeError('Cached subset differs from direct forward')
                    records.append(dict(**base,strategy=strategy,k=k,repeat=repeat,n_kept=len(chosen),risk=result))
        if (index+1)%25==0: print(f'Fold {fold}: {index+1}/{len(patients)} validation patients',flush=True)
    predictions=pd.DataFrame(records)
    actual=predictions.loc[predictions.strategy.eq('full'),'risk'].to_numpy(np.float32)
    expected=saved.risk_score.to_numpy(np.float32)
    if not np.allclose(actual,expected,atol=2e-6,rtol=2e-6): raise RuntimeError('Full risk replay mismatch')
    if not all(torch.equal(before[name],value) for name,value in model.state_dict().items()) \
            or any(p.grad is not None or p.requires_grad for p in model.parameters()): raise RuntimeError('Frozen model changed')
    if checkpoint_hash!=sha256(checkpoint): raise RuntimeError('Checkpoint changed during diagnosis')
    metrics=fold_metrics(predictions).assign(fold=fold)
    full_ci=metrics.loc[metrics.strategy.eq('full'),'cindex'].item()
    old=pd.read_csv(Path(reference)/'metrics.csv')
    if not np.isclose(full_ci,old.loc[old.fold.eq(fold),'val_cindex'].item(),atol=1e-12,rtol=0): raise RuntimeError('Full C-index replay mismatch')
    predictions.to_csv(private/f'fold_{fold}_predictions.csv',index=False)
    attention=pd.DataFrame(concentration);attention.to_csv(private/f'fold_{fold}_attention.csv',index=False)
    audit=dict(fold=fold,n_val=len(patients),n_patches=int(attention.patch_count.sum()),
               full_risk_max_abs_error=float(np.max(np.abs(actual-expected))),full_cindex=full_ci,
               direct_subset_max_abs_error=max_direct_error,model_frozen=True,checkpoint_sha256=checkpoint_hash,
               checkpoint_step=int(state['step']))
    print(f'Fold {fold} complete | Full C-index={full_ci:.6f} | Full replay max error={audit["full_risk_max_abs_error"]:.3g}',flush=True)
    return metrics,attention,audit


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv',type=Path,default=ROOT/'data/trident/features_uni_v1/all_label_roi64.csv')
    parser.add_argument('--reference',type=Path,default=Path(__file__).parent.parent/'pathology_stage6/results')
    parser.add_argument('--output',type=Path,default=Path(__file__).parent/'results')
    parser.add_argument('--seed',type=int,default=123)
    parser.add_argument('--device',default='cuda:0')
    args=parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()): raise FileExistsError('Use a fresh output directory; no overwriting')
    args.output.mkdir(parents=True)
    status=dict(status='running',pid=os.getpid(),completed_folds=0,training_started=False)
    sources={}
    def track(path):
        path=Path(path).resolve();value=sha256(path)
        if str(path) in sources and sources[str(path)]!=value: raise RuntimeError('Reference source changed between folds')
        sources[str(path)]=value
        return path
    code_paths=[Path(__file__),Path(__file__).with_name('report.py'),ROOT/'model/abmil_cox.py',ROOT/'dataset.py',ROOT/'utils.py']
    code_hashes={str(path.relative_to(ROOT)):sha256(path) for path in code_paths}
    try:
        (args.output/'status.json').write_text(json.dumps(status,indent=2))
        torch.set_num_threads(4);seed_everything(args.seed)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        reference=json.loads(track(args.reference/'validation.json').read_text())
        if reference['status']!='completed' or reference['feature_dim']!=1024: raise ValueError('Completed raw UNI Stage6 reference required')
        for name,value in code_hashes.items():
            if name in reference['code_sha256'] and reference['code_sha256'][name]!=value: raise ValueError(f'Shared source changed: {name}')
        if reference['source_sha256'][str(args.csv.resolve())]!=sha256(args.csv): raise ValueError('Cohort changed since Stage6')
        samples=pd.read_csv(track(args.csv),dtype={'pa_id':str,'ct_id':str})
        if len(samples)!=721 or samples.event.sum()!=251 or samples.pa_id.duplicated().any() or set(samples.split)!=set(range(5)):
            raise ValueError('Expected unchanged 721 patients, 251 events, saved five folds')
        if not samples.event.isin([0,1]).all() or not np.isfinite(samples.time).all() or (samples.time<0).any(): raise ValueError('Invalid survival labels')
        samples['time']=samples.time.to_numpy(np.float32).astype(float)
        provenance=Path(__file__).parent.parent/'pathology_stage5/results/private/feature_provenance.json'
        if reference['source_sha256'][str(provenance.resolve())]!=sha256(provenance): raise ValueError('Feature provenance changed')
        features=json.loads(track(provenance).read_text())
        config=dict(ks=[128,512,1024],random_repeats=30,seed=args.seed,device=args.device,
                    raw_uni=True,l2=False,training_started=False,torch_version=str(torch.__version__),tf32=False,
                    protocol='Frozen Stage6 final outer models; each patient evaluated by their held-out fold model')
        (args.output/'run_config.json').write_text(json.dumps(config,indent=2))
        tables,attention,audits=[],[],[]
        for fold in range(5):
            status['active_fold']=fold
            (args.output/'status.json').write_text(json.dumps(status,indent=2))
            for file in ('model.pth','val_predictions.csv'): track(args.reference/f'fold_{fold}/private/outer'/file)
            track(args.reference/'metrics.csv')
            table,part,audit=run_fold(samples,fold,args.reference,args.output,args.device,config['ks'],30,args.seed,features)
            tables.append(table);attention.append(part);audits.append(audit)
            pd.concat(tables,ignore_index=True).to_csv(args.output/'fold_metrics.csv',index=False)
            pd.DataFrame(audits).to_csv(args.output/'replay_audit.csv',index=False)
            status['completed_folds']=fold+1
        from report import summarize,attention_summary
        summarize(pd.concat(tables,ignore_index=True),args.output,config)
        attention_summary(pd.concat(attention,ignore_index=True),args.output,config['ks'])
        if any(sha256(path)!=value for path,value in sources.items()): raise RuntimeError('Reference source changed')
        if any(sha256(path)!=features[path] for path in samples.pa_path): raise RuntimeError('Raw UNI feature changed')
        if any(sha256(ROOT/path)!=value for path,value in code_hashes.items()): raise RuntimeError('Diagnostic code changed')
        validation=dict(status='completed',n_patients=721,n_events=251,n_patches=sum(x['n_patches'] for x in audits),
                        n_folds=5,neural_fits=0,full_replay_verified=True,weights_unchanged=True,source_hashes_unchanged=True,
                        source_sha256=sources,code_sha256=code_hashes,**config)
        (args.output/'validation.json').write_text(json.dumps(validation,indent=2))
        status['status']='completed'
        print('Completed five-fold frozen patch diagnosis; no training was performed.',flush=True)
    except Exception as exc:
        status.update(status='failed',error=f'{type(exc).__name__}: {exc}');raise
    finally:
        (args.output/'status.json').write_text(json.dumps(status,indent=2))


if __name__=='__main__': main()
