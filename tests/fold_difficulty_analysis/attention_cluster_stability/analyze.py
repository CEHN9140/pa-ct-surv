"""Patient-balanced Train-only dictionaries and complete Best-to-Late class accounting."""
from pathlib import Path
import argparse,hashlib,json,sys
import h5py,joblib
import numpy as np
import pandas as pd
from PIL import Image

BASE=Path(__file__).resolve().parent;ROOT=BASE.parents[2];ATTR=BASE.parent/'attention_risk_attribution';PREVIOUS=ATTR/'best_reference'
sys.path[:0]=[str(ATTR),str(PREVIOUS)]
from run import sha256
from best_tools import white_fraction
from cluster_tools import fit_dictionary,aggregate_bag,sample_indices,patient_contexts


def main(smoke=False):
    out=BASE/('validation/smoke' if smoke else 'private')
    if out.exists():raise FileExistsError('Refusing overwrite')
    out.mkdir(parents=True);technical=out/'technical';technical.mkdir()
    if not json.loads((PREVIOUS/'results/validation.json').read_text())['passed']:raise ValueError('Prior attribution incomplete')
    source=pd.read_csv(ROOT/'data/trident/features_uni_v1/all_label_roi64.csv',dtype={'pa_id':str})
    old=pd.read_csv(PREVIOUS/'private/patients.csv',dtype={'pa_id':str});audit=pd.read_csv(ATTR/'private/input_audit.csv',dtype={'pa_id':str}).set_index('pa_id')
    if len(source)!=721 or source.event.sum()!=251 or source.pa_id.duplicated().any() or len(old)!=3605:raise ValueError('Wrong original cohort')
    old=old.set_index(['fold','pa_id']);verified={};provenance=[];bag_rows=[]
    def load(row):
        for filename,expected in [(row.pa_path,audit.loc[row.pa_id].feature_sha256),(row.coord_path,audit.loc[row.pa_id].coord_sha256)]:
            stat=Path(filename).stat();signature=(stat.st_size,stat.st_mtime_ns)
            if filename not in verified:
                if sha256(filename)!=expected:raise ValueError('UNI/coordinate source changed')
                verified[filename]=signature
            elif verified[filename]!=signature:raise ValueError('Source stat changed')
        with h5py.File(row.pa_path) as f,h5py.File(row.coord_path) as c:
            raw=f['features'][:];np.testing.assert_array_equal(f['coords'][:],c['coords'][:])
            if raw.shape!=(len(f['coords']),1024) or str(c['coords'].attrs['name'])!=row.pa_id or not np.isfinite(raw).all():raise ValueError('Raw feature/geometry mismatch')
        return raw
    for fold in range(1 if smoke else 5):
        frame=source
        if smoke:
            ids=[]
            for validation in [False,True]:
                mask=source.split.eq(fold) if validation else source.split.ne(fold)
                for event in [0,1]:ids.extend(source.index[mask&source.event.eq(event)][:16])
            frame=source.loc[sorted(ids)]
        directory=out/f'fold_{fold}';directory.mkdir()
        k=4 if smoke else 20;cap=32 if smoke else 128
        print(f'Fold {fold}: fit{k} classes on Train only',flush=True)
        model,fit,prototypes=fit_dictionary(frame,fold,load,n_clusters=k,cap=cap)
        fit.to_csv(directory/'fit_patients.csv',index=False);prototypes.to_csv(directory/'prototypes.csv',index=False)
        joblib.dump(model,directory/'dictionary.joblib');centers_hash=hashlib.sha256(model.cluster_centers_.tobytes()).hexdigest()
        np.savez_compressed(directory/'centers.npz',centers=model.cluster_centers_,training_radius=model.training_radius_)
        contexts={}
        for role in ['train','val']:
            ids=[r.pa_id for r in frame.itertuples() if (r.split==fold)==(role=='val')]
            observed=old.loc[[(fold,p) for p in ids]].copy();observed=patient_contexts(observed)
            for patient in ids:contexts[patient]=observed.loc[(fold,patient)]
        for number,row in enumerate(frame.itertuples(),1):
            raw=load(row);labels=model.predict(raw);case=directory/row.pa_id;case.mkdir()
            distance=np.linalg.norm(raw-model.cluster_centers_[labels],axis=1)
            np.savez_compressed(case/'assignment.npz',labels=labels.astype(np.int16),distance=distance.astype(np.float32))
            previous=PREVIOUS/f'private/fold_{fold}/{row.pa_id}/best_attribution.npz';signature=Path(previous).stat()
            with np.load(previous,allow_pickle=False) as f:c=f['contributions'];best=f['best_attention'];late=f['late_attention']
            record=contexts[row.pa_id]
            if len(c)!=len(raw) or abs(c.sum()-record.attention_risk_delta)>1e-8*max(1.,abs(record.attention_risk_delta)):raise ValueError('Prior IG alignment/completeness differs')
            if abs(best.sum()-1)>2e-5 or abs(late.sum()-1)>2e-5:raise ValueError('Attention mass invalid')
            with h5py.File(row.coord_path) as f:coords=f['coords'][:];attrs=dict(f['coords'].attrs)
            cached=technical/(row.pa_id+'.npz')
            if not cached.exists():
                thumb=Path(row.coord_path).parents[2]/'thumbnails'/f'{row.pa_id}.jpg'
                with Image.open(thumb) as image:rgb=np.asarray(image.convert('RGB'))
                dimensions=(int(attrs['level0_width']),int(attrs['level0_height']))
                if abs(rgb.shape[1]/rgb.shape[0]-dimensions[0]/dimensions[1])>.01:raise ValueError('Thumbnail geometry')
                white=white_fraction(rgb,coords,float(attrs['patch_size_level0']),dimensions)
                np.savez_compressed(cached,white=white,qc_indices=sample_indices(len(raw),8,row.pa_id))
            with np.load(cached,allow_pickle=False) as f:white=f['white']
            aggregate=aggregate_bag(labels,best,late,c,np.linalg.norm(raw,axis=1),white,record,k)
            aggregate['mean_assignment_distance']=np.divide(np.bincount(labels,weights=distance,minlength=k),aggregate.patch_count,out=np.full(k,np.nan),where=aggregate.patch_count>0)
            aggregate.insert(0,'pa_id',row.pa_id);aggregate.insert(0,'split','val' if row.split==fold else 'train');aggregate.insert(0,'fold',fold)
            if int(aggregate.patch_count.sum())!=len(raw) or abs(aggregate.ig_sum.sum()-record.attention_risk_delta)>1e-8*max(1.,abs(record.attention_risk_delta)):raise ValueError('Class accounting lost patches/IG')
            bag_rows.append(aggregate)
            provenance.append(dict(fold=fold,pa_id=row.pa_id,ig_sha256=sha256(previous),ig_size=signature.st_size,ig_mtime_ns=signature.st_mtime_ns))
            if number%100==0 or number==len(frame):print(f'Fold {fold}: assigned{number}/{len(frame)}',flush=True)
        if hashlib.sha256(model.cluster_centers_.tobytes()).hexdigest()!=centers_hash:raise ValueError('Dictionary updated on validation')
        (directory/'fit_audit.json').write_text(json.dumps(dict(fold=fold,clusters=k,train_patients=len(fit),validation_used_for_fit=False,patient_weight_equal=True,
            centers_sha256=centers_hash,kmeans_steps=int(model.n_steps_),raw_uni=True,model_input_l2=False),indent=2))
        pd.concat(bag_rows,ignore_index=True).to_csv(out/'patient_clusters.csv',index=False)
    for filename,signature in verified.items():
        stat=Path(filename).stat()
        if (stat.st_size,stat.st_mtime_ns)!=signature:raise ValueError('Input changed during assignment')
    for row in provenance:
        path=PREVIOUS/f"private/fold_{row['fold']}/{row['pa_id']}/best_attribution.npz";stat=path.stat()
        if (stat.st_size,stat.st_mtime_ns)!=(row['ig_size'],row['ig_mtime_ns']) or sha256(path)!=row['ig_sha256']:raise ValueError('IG source changed during assignment')
    pd.DataFrame(provenance).to_csv(out/'provenance.csv',index=False)
    (out/'assigned.json').write_text(json.dumps(dict(passed=True,folds=1 if smoke else 5,classes_per_fold=4 if smoke else 20,
        fold_patient_observations=len(bag_rows),neural_training=False,dict_fit_train_only=True),indent=2))
    print('Train-only dictionaries and all class attributions complete',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--smoke',action='store_true');main(parser.parse_args().smoke)
