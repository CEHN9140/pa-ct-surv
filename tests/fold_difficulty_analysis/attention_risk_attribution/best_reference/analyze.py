"""Best-attention frozen intervention for full eval Train and Val; no fitting."""
from pathlib import Path
import argparse,json,sys
import h5py
import numpy as np
import pandas as pd
import torch
from sksurv.metrics import concordance_index_censored

BASE=Path(__file__).resolve().parent;PARENT=BASE.parent;ROOT=PARENT.parents[2]
sys.path[:0]=[str(PARENT),str(ROOT)]
from attribution import attention_path,pair_changes
from model.build import Pa_Model
from run import sha256
from best_tools import contextual_counts,context_scores


def main(smoke=False):
    torch.set_num_threads(4);out=BASE/('validation/smoke' if smoke else 'private')
    if out.exists():raise FileExistsError('Refusing overwrite')
    out.mkdir(parents=True);source=pd.read_csv(ROOT/'data/trident/features_uni_v1/all_label_roi64.csv',dtype={'pa_id':str})
    if len(source)!=721 or source.event.sum()!=251:raise ValueError('Wrong cohort')
    original=pd.read_csv(PARENT/'private/patients.csv',dtype={'pa_id':str}).set_index(['fold','pa_id'])
    old_audit=pd.read_csv(PARENT/'private/input_audit.csv',dtype={'pa_id':str}).set_index('pa_id')
    patients=[];pairs_all=[];metrics=[];verified={};hashes={}
    for fold in range(1 if smoke else 5):
        saved=PARENT.parent/f'cox_epoch_diagnostics/bs32/diagnostics/fold_{fold}';audit=json.loads((saved/'audit.json').read_text())
        frame=pd.read_csv(saved/'private_cohort.csv',dtype={'pa_id':str});assert frame.pa_id.tolist()==source.pa_id.tolist()
        np.testing.assert_array_equal(frame.split,source.split);np.testing.assert_array_equal(frame.event,source.event)
        np.testing.assert_array_equal(frame.time.to_numpy(np.float32),source.time.to_numpy(np.float32))
        best_path=saved/f"private/epoch_{audit['best_epoch']:03d}.npz";late_path=saved/f"private/epoch_{audit['last_epoch']:03d}.npz"
        checkpoint=PARENT.parent/f'cox_epoch_diagnostics/bs32/checkpoints/fold_{fold}/last_model.pth'
        for path in [best_path,late_path,checkpoint]:hashes[str(path)]=sha256(path)
        with np.load(best_path,allow_pickle=False) as f:best={k:f[k] for k in ['offsets','attention','risk']}
        with np.load(late_path,allow_pickle=False) as f:late={k:f[k] for k in ['offsets','attention','risk']}
        np.testing.assert_array_equal(best['offsets'],late['offsets'])
        model=Pa_Model(feature_dim=1024,abmil_dropout=.25,l2_normalize=False).cuda().eval()
        model.load_state_dict(torch.load(checkpoint,weights_only=True,map_location='cuda'))
        indices=np.arange(len(source))
        if smoke:
            chosen=[]
            for validation in [False,True]:
                mask=source.split.eq(fold) if validation else source.split.ne(fold)
                for event in [0,1]:chosen.extend(source.index[mask&source.event.eq(event)][:16])
            indices=np.asarray(sorted(chosen))
        observations=[]
        for number,index in enumerate(indices,1):
            row=source.iloc[index];case=out/f'fold_{fold}'/row.pa_id;case.mkdir(parents=True)
            for filename,expected in [(row.pa_path,old_audit.loc[row.pa_id].feature_sha256),(row.coord_path,old_audit.loc[row.pa_id].coord_sha256)]:
                stat=Path(filename).stat();signature=(stat.st_size,stat.st_mtime_ns)
                if filename not in verified:
                    if sha256(filename)!=expected:raise ValueError('Feature/coord changed since original diagnostic')
                    verified[filename]=signature
                elif verified[filename]!=signature:raise ValueError('Input stat changed')
            with h5py.File(row.pa_path) as f,h5py.File(row.coord_path) as c:
                features=f['features'][:];np.testing.assert_array_equal(f['coords'][:],c['coords'][:])
            start,end=map(int,best['offsets'][index:index+2])
            if features.shape!=(end-start,1024) or not np.isfinite(features).all():raise ValueError('Patch alignment differs')
            with torch.no_grad():
                native,_,attention,h=model.mil(torch.from_numpy(features).cuda(),return_patch_features=True)
                if abs(float(native)-float(late['risk'][index]))>2e-5:raise ValueError('Native late risk mismatch')
                np.testing.assert_allclose(attention.cpu().numpy().reshape(-1),late['attention'][start:end],rtol=1e-5,atol=1e-7)
                result=attention_path(h[0],torch.from_numpy(best['attention'][start:end]),attention,model.mil.post_compression,model.mil.risk_head)
            c=result['contributions'].cpu().numpy();error=abs(float(c.sum())-result['risk_delta'])
            if error>1e-8*max(1.,abs(result['risk_delta'])):raise ValueError('IG completeness failed')
            reused=False
            if row.split==fold:
                with np.load(PARENT/f'private/fold_{fold}/{row.pa_id}/attribution.npz',allow_pickle=False) as f:np.testing.assert_allclose(c,f['best_contributions'],rtol=1e-8,atol=1e-9)
                np.testing.assert_allclose(result['risk_early_attention'],original.loc[(fold,row.pa_id)].best_attention_risk,rtol=0,atol=1e-8);reused=True
            np.savez_compressed(case/'best_attribution.npz',contributions=c,best_attention=best['attention'][start:end],late_attention=late['attention'][start:end])
            observations.append(dict(fold=fold,split='val' if row.split==fold else 'train',pa_id=row.pa_id,time=float(np.float32(row.time)),event=int(row.event),
                best_epoch=audit['best_epoch'],last_epoch=audit['last_epoch'],n_patches=len(c),actual_best_risk=float(best['risk'][index]),actual_late_risk=float(late['risk'][index]),
                frozen_best_attention_risk=result['risk_early_attention'],frozen_late_attention_risk=result['risk_late_attention'],attention_risk_delta=result['risk_delta'],
                completeness_error=error,native_replay_error=abs(float(native)-float(late['risk'][index])),prior_val_ig_verified=reused))
            if number%100==0 or number==len(indices):print(f'Fold {fold}: eval/IG {number}/{len(indices)}',flush=True)
        fold_frame=pd.DataFrame(observations)
        for split,group in fold_frame.groupby('split',sort=True):
            group=group.reset_index(drop=True)
            pairs,roles=pair_changes(group.frozen_best_attention_risk,group.frozen_late_attention_risk,group.time,group.event)
            actual,actual_roles=pair_changes(group.actual_best_risk,group.actual_late_risk,group.time,group.event)
            np.testing.assert_array_equal(pairs[['event_index','other_index']],actual[['event_index','other_index']])
            counts=contextual_counts(pairs,len(group))
            for key in counts:group[key]=counts[key]
            for key in ['comparable_degree','pair_credit_change','mean_pair_credit_change','broken_pairs','corrected_pairs']:
                group[key]=roles[key];group['actual_'+key]=actual_roles[key]
            group['actual_changed_pair_fraction']=np.divide(actual_roles.broken_pairs+actual_roles.corrected_pairs,actual_roles.comparable_degree,out=np.full(len(group),np.nan),where=actual_roles.comparable_degree>0)
            pairs['actual_credit_change']=actual.credit_change;pairs['fold']=fold;pairs['split']=split
            pairs['event_pa_id']=group.pa_id.to_numpy()[pairs.event_index];pairs['other_pa_id']=group.pa_id.to_numpy()[pairs.other_index]
            np.testing.assert_allclose(group.attention_risk_delta.to_numpy()[pairs.event_index]-group.attention_risk_delta.to_numpy()[pairs.other_index],pairs.margin_delta,rtol=1e-9,atol=1e-9)
            for row in group.itertuples():
                path=out/f'fold_{fold}/{row.pa_id}/best_attribution.npz'
                with np.load(path,allow_pickle=False) as f:c=f['contributions']
                scores=context_scores(c,row)
                group.loc[group.pa_id.eq(row.pa_id),'harm_support_mass']=scores['harm_support'].sum()
                group.loc[group.pa_id.eq(row.pa_id),'repair_support_mass']=scores['repair_support'].sum()
            scores={name:float(concordance_index_censored(group.event.to_numpy(bool),group.time,group[name])[0]) for name in ['actual_best_risk','actual_late_risk','frozen_best_attention_risk','frozen_late_attention_risk']}
            if abs(scores['frozen_late_attention_risk']-scores['frozen_best_attention_risk']-pairs.credit_change.mean())>1e-12:raise ValueError('C-index/pairs differ')
            metrics.append(dict(fold=fold,split=split,patients=len(group),best_epoch=audit['best_epoch'],last_epoch=audit['last_epoch'],**scores,
                attention_delta_c=float(pairs.credit_change.mean()),actual_delta_c=float(actual.credit_change.mean()),corrected_pairs=int(pairs.credit_change.gt(0).sum()),broken_pairs=int(pairs.credit_change.lt(0).sum())))
            patients.extend(group.to_dict('records'));pairs_all.append(pairs)
        pd.DataFrame(patients).to_csv(out/'patients.csv',index=False);pd.concat(pairs_all,ignore_index=True).to_csv(out/'pairs.csv',index=False)
        pd.DataFrame(metrics).to_csv(out/'fold_metrics.csv',index=False)
    if not smoke and len(patients)!=3605:raise ValueError('Missing fold-patient observations')
    for filename,signature in verified.items():
        stat=Path(filename).stat()
        if (stat.st_size,stat.st_mtime_ns)!=signature:raise ValueError('Source file changed')
    for filename,digest in hashes.items():
        if sha256(filename)!=digest:raise ValueError('Checkpoint/snapshot changed')
    (out/'source_sha256.json').write_text(json.dumps(hashes,indent=2))
    (out/'attribution_completed.json').write_text(json.dumps(dict(passed=True,fold_patient_observations=len(patients),training=False,best_reference_only=True,prior_validation_attributions_verified=True),indent=2))
    print('Train/Val best-reference attribution complete',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');main(p.parse_args().smoke)
