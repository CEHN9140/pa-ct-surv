"""Frozen bs32 final checkpoints and Epoch5 attention; all private outputs stay local."""
from pathlib import Path
import argparse,hashlib,json,sys,time
import h5py
import numpy as np
import pandas as pd
import torch
from sksurv.metrics import concordance_index_censored

BASE=Path(__file__).resolve().parent;ROOT=BASE.parents[2];SOURCE=BASE.parent/'cox_epoch_diagnostics'
sys.path.insert(0,str(ROOT))
from model.build import Pa_Model
from attribution import attention_path,pair_changes,select_controls


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda:f.read(8*1024*1024),b''):h.update(part)
    return h.hexdigest()


def run(smoke=False):
    torch.set_num_threads(4);out=BASE/('validation/smoke' if smoke else 'private')
    if out.exists():raise FileExistsError('Refusing overwrite of attribution outputs')
    out.mkdir(parents=True);source=pd.read_csv(ROOT/'data/trident/features_uni_v1/all_label_roi64.csv',dtype={'pa_id':str})
    if len(source)!=721 or int(source.event.sum())!=251 or source.pa_id.duplicated().any():raise ValueError('Original cohort differs')
    audit=[];patient_rows=[];patch_rows=[];pairs_all=[];summary=[];manifest={}
    for fold in range(1 if smoke else 5):
        directory=SOURCE/f'bs32/diagnostics/fold_{fold}';saved=pd.read_csv(directory/'private_cohort.csv',dtype={'pa_id':str})
        if saved.pa_id.tolist()!=source.pa_id.tolist():raise ValueError('Snapshot patient order changed')
        for field in ['split','event']:np.testing.assert_array_equal(saved[field],source[field])
        np.testing.assert_array_equal(saved.time.to_numpy(np.float32),source.time.to_numpy(np.float32))
        meta=json.loads((directory/'audit.json').read_text());last=meta['last_epoch'];best_epoch=meta['best_epoch']
        checkpoint=SOURCE/f'bs32/checkpoints/fold_{fold}/last_model.pth'
        early_path=directory/'private/epoch_005.npz';late_path=directory/f'private/epoch_{last:03d}.npz'
        best_path=directory/f'private/epoch_{best_epoch:03d}.npz';epoch10_path=directory/'private/epoch_010.npz'
        for p in [checkpoint,early_path,late_path,best_path,epoch10_path]:manifest[str(p)]=sha256(p)
        with np.load(early_path,allow_pickle=False) as f:early={k:f[k] for k in ['offsets','attention','risk']}
        with np.load(late_path,allow_pickle=False) as f:late={k:f[k] for k in ['offsets','attention','risk']}
        with np.load(best_path,allow_pickle=False) as f:best={k:f[k] for k in ['offsets','attention','risk']}
        with np.load(epoch10_path,allow_pickle=False) as f:epoch10_risk=f['risk']
        np.testing.assert_array_equal(early['offsets'],late['offsets']);np.testing.assert_array_equal(best['offsets'],late['offsets'])
        model=Pa_Model(feature_dim=1024,abmil_dropout=.25,l2_normalize=False).cuda().eval()
        model.load_state_dict(torch.load(checkpoint,weights_only=True,map_location='cuda'))
        if any(m.training for m in model.modules()):raise ValueError('Dropout must be disabled')
        indices=np.flatnonzero(source.split.eq(fold))
        if smoke:indices=np.r_[indices[source.iloc[indices].event.to_numpy()==1][:2],indices[source.iloc[indices].event.to_numpy()==0][:2]]
        per_fold=[]
        for number,index in enumerate(indices):
            row=source.iloc[index];case=out/f'fold_{fold}'/row.pa_id;case.mkdir(parents=True)
            with h5py.File(row.pa_path) as f,h5py.File(row.coord_path) as c:
                features=f['features'][:];coords=f['coords'][:];attrs=dict(c['coords'].attrs)
                np.testing.assert_array_equal(coords,c['coords'][:])
                for field in ['name','patch_size','patch_size_level0','level0_width','level0_height','target_magnification']:
                    if attrs[field]!=f['coords'].attrs[field]:raise ValueError('Feature coordinate geometry mismatch')
            if str(attrs['name'])!=row.pa_id or features.shape!=(len(coords),1024) or not np.isfinite(features).all():raise ValueError('Feature/identity mismatch')
            start,end=map(int,late['offsets'][index:index+2])
            if end-start!=len(features):raise ValueError('Snapshot patch count differs')
            with torch.no_grad():
                native,_,weights,h=model.mil(torch.from_numpy(features).cuda(),return_patch_features=True)
                np.testing.assert_allclose(weights.cpu().numpy().reshape(-1),late['attention'][start:end],rtol=1e-5,atol=1e-7)
                if abs(float(native)-float(late['risk'][index]))>2e-5:raise ValueError('Native risk replay differs')
                result=attention_path(h[0],torch.from_numpy(early['attention'][start:end]),weights,model.mil.post_compression,model.mil.risk_head)
            best_result=attention_path(h[0],torch.from_numpy(best['attention'][start:end]),weights,model.mil.post_compression,model.mil.risk_head)
            best_ig=best_result.pop('contributions').cpu().numpy()
            if best_result['completeness_error']>1e-8*max(1.,abs(best_result['risk_delta'])):raise ValueError('Best-reference IG completeness failed')
            ig=result.pop('contributions').cpu().numpy();a=early['attention'][start:end];b=late['attention'][start:end]
            if result['completeness_error']>1e-8*max(1.,abs(result['risk_delta'])):raise ValueError('IG completeness failed')
            if abs(result['risk_late_attention']-float(native))>1e-4:raise ValueError('Float64 head risk replay differs')
            norms=np.linalg.norm(features,axis=1)
            np.savez_compressed(case/'attribution.npz',contributions=ig,early_attention=a,late_attention=b,delta_attention=b-a,uni_norm=norms,best_contributions=best_ig,best_attention=best['attention'][start:end])
            patient=dict(fold=fold,pa_id=row.pa_id,time=float(np.float32(row.time)),event=int(row.event),last_epoch=last,n_patches=len(coords),
                best_epoch=best_epoch,actual_best_risk=float(best['risk'][index]),actual_epoch10_risk=float(epoch10_risk[index]),
                best_attention_risk=best_result['risk_early_attention'],best_to_late_delta=best_result['risk_delta'],best_completeness_error=best_result['completeness_error'],
                actual_epoch5_risk=float(early['risk'][index]),actual_late_risk=float(late['risk'][index]),**result,
                sum_abs_ig=float(np.abs(ig).sum()),max_abs_ig=float(np.abs(ig).max()),native_replay_error=abs(float(native)-float(late['risk'][index])))
            per_fold.append(patient)
            high,low=select_controls(ig)
            for group,chosen in [('high_abs_ig',high),('low_abs_ig',low)]:
                for rank,k in enumerate(chosen):patch_rows.append(dict(fold=fold,pa_id=row.pa_id,group=group,rank=rank,patch_index=int(k),
                    contribution=float(ig[k]),best_reference_contribution=float(best_ig[k]),abs_contribution=float(abs(ig[k])),delta_attention=float(b[k]-a[k]),early_attention=float(a[k]),late_attention=float(b[k]),
                    uni_norm=float(norms[k]),x=int(coords[k,0]),y=int(coords[k,1]),side=float(attrs['patch_size_level0']),target=int(attrs['patch_size'])))
            audit.append(dict(fold=fold,pa_id=row.pa_id,feature_sha256=sha256(row.pa_path),coord_sha256=sha256(row.coord_path),
                coordinate_order_exact=True,completeness_error=result['completeness_error']))
            if (number+1)%20==0 or number==len(indices)-1:print(f'Fold {fold}: attribution {number+1}/{len(indices)}',flush=True)
        frame=pd.DataFrame(per_fold)
        counter,patients=pair_changes(frame.risk_early_attention,frame.risk_late_attention,frame.time,frame.event)
        temporal,temporal_patients=pair_changes(frame.actual_epoch5_risk,frame.actual_late_risk,frame.time,frame.event)
        np.testing.assert_array_equal(counter[['event_index','other_index']],temporal[['event_index','other_index']])
        counter['temporal_credit_change']=temporal.credit_change;counter['fold']=fold
        best_temporal,_=pair_changes(frame.actual_best_risk,frame.actual_late_risk,frame.time,frame.event)
        counter['best_to_late_credit_change']=best_temporal.credit_change
        counter['event_pa_id']=frame.pa_id.to_numpy()[counter.event_index];counter['other_pa_id']=frame.pa_id.to_numpy()[counter.other_index]
        counter['event_ig_sum']=frame.risk_delta.to_numpy()[counter.event_index];counter['other_ig_sum']=frame.risk_delta.to_numpy()[counter.other_index]
        np.testing.assert_allclose(counter.event_ig_sum-counter.other_ig_sum,counter.margin_delta,atol=1e-9,rtol=1e-9)
        for name in ['comparable_degree','pair_credit_change','mean_pair_credit_change','broken_pairs','corrected_pairs']:frame[name]=patients[name].to_numpy()
        for name in ['pair_credit_change','mean_pair_credit_change','broken_pairs','corrected_pairs']:frame['temporal_'+name]=temporal_patients[name].to_numpy()
        frame['temporal_changed_pair_fraction']=np.divide(temporal_patients.broken_pairs+temporal_patients.corrected_pairs,temporal_patients.comparable_degree,out=np.full(len(frame),np.nan),where=temporal_patients.comparable_degree>0)
        patient_rows.extend(frame.to_dict('records'));pairs_all.append(counter)
        events=frame.event.to_numpy(bool);times=frame.time.to_numpy(float)
        if len(indices)>2:
            scores={name:float(concordance_index_censored(events,times,frame[name])[0]) for name in ['actual_epoch5_risk','actual_late_risk','risk_early_attention','risk_late_attention']}
            expected=scores['risk_late_attention']-scores['risk_early_attention']
            if abs(expected-counter.credit_change.mean())>1e-12:raise ValueError('Pair credit/C-index mismatch')
            summary.append(dict(fold=fold,last_epoch=last,patients=len(frame),**scores,attention_delta_c=expected,
                corrected_pairs=int((counter.credit_change>0).sum()),broken_pairs=int((counter.credit_change<0).sum()),
                temporal_broken_pairs=int((counter.temporal_credit_change<0).sum()),
                temporal_and_attention_broken=int(((counter.temporal_credit_change<0)&(counter.credit_change<0)).sum()),
                max_completeness_error=float(frame.completeness_error.max()),max_native_replay_error=float(frame.native_replay_error.max())))
        pd.DataFrame(patient_rows).to_csv(out/'patients.csv',index=False);pd.DataFrame(patch_rows).to_csv(out/'selected_patches.csv',index=False)
        pd.DataFrame(audit).to_csv(out/'input_audit.csv',index=False);pd.concat(pairs_all).to_csv(out/'pairs.csv',index=False)
    if not smoke and len(patient_rows)!=721:raise ValueError('Missing outer patients')
    for p,digest in manifest.items():
        if sha256(p)!=digest:raise ValueError('Input checkpoint/snapshot changed')
    (out/'source_sha256.json').write_text(json.dumps(manifest,indent=2))
    pd.DataFrame(summary).to_csv(out/'fold_summary.csv',index=False)
    (out/'attribution_completed.json').write_text(json.dumps(dict(passed=True,patients=len(patient_rows),folds=1 if smoke else 5,neural_training=False,
        integration='exact straight-line IG of frozen eval Linear/ReLU/Linear head',all_patch_completeness_checked=True,selected_patch_count=len(patch_rows)),indent=2))
    print('Frozen attribution complete',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--smoke',action='store_true');run(parser.parse_args().smoke)
