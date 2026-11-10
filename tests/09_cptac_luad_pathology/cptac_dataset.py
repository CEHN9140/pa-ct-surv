"""One frozen randomly selected WSI per CPTAC patient, shared by three inputs."""
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from torch.nn.functional import normalize
from torch.utils.data import Dataset


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def feature_signature(paths):
    values=[(str(Path(p).resolve()),Path(p).stat().st_size,Path(p).stat().st_mtime_ns) for p in paths]
    return hashlib.sha256(json.dumps(values).encode()).hexdigest()


def validate_preparation(data_dir):
    root=Path(data_dir)
    metadata=json.loads((root/'preparation.json').read_text())
    for path,expected in metadata['source_hashes'].items():
        if file_hash(path)!=expected:raise ValueError(f'Source table changed after preparation: {path}')
    if file_hash(root/'all_label.csv')!=metadata['cohort_sha256']:
        raise ValueError('Frozen slide selection, labels or folds changed')
    return metadata


def prepare_cohort(source_root,output_dir):
    source_root,output_dir=Path(source_root).resolve(),Path(output_dir).resolve()
    if (output_dir/'all_label.csv').exists():
        metadata=validate_preparation(output_dir)
        if metadata['source_root']!=str(source_root):raise ValueError('Existing cohort uses another source')
        data=CPTACDataset(output_dir,'uni').samples
        CPTACDataset(output_dir,'keep')
        print(f'Reusing frozen single-WSI selection and five-fold assignments: {output_dir}',flush=True)
        return data
    source_paths=[source_root/'survival'/name for name in
                  ['patients.csv','slides.csv','PathoBench_CPTAC_LUAD_OS.tsv']]
    patients,slides,benchmark=[pd.read_csv(p,sep='\t' if p.suffix=='.tsv' else ',',
                                         dtype={'case_id':str,'slide_id':str}) for p in source_paths]
    if patients.case_id.duplicated().any() or patients.case_id.isna().any():
        raise ValueError('Patient IDs must be unique and nonmissing')
    if set(patients.time_unit.str.lower())!={'days'}:
        raise ValueError('Expected OS_days in days')
    if not patients.OS_event.isin([0,1]).all() or not np.isfinite(patients.OS_days).all() or (patients.OS_days<0).any():
        raise ValueError('Invalid OS event/time labels')
    reference=patients.set_index('case_id')[['OS_event','OS_days']]
    for name,table in [('slides',slides),('benchmark',benchmark)]:
        if (table[['case_id','slide_id','OS_event','OS_days']].isna().any().any()
                or table.slide_id.duplicated().any() or set(table.case_id)!=set(reference.index)):
            raise ValueError(f'Invalid patient/slide mapping in {name}')
        expected=reference.loc[table.case_id].reset_index(drop=True)
        if not np.array_equal(table[['OS_event','OS_days']].to_numpy(),expected.to_numpy()):
            raise ValueError(f'Conflicting patient labels in {name}')
    left=slides.set_index('slide_id').case_id.sort_index()
    right=benchmark.set_index('slide_id').case_id.sort_index()
    if not left.equals(right):raise ValueError('Slide IDs or patients differ between source tables')
    fold_columns=[f'fold_{i}'for i in range(5)]
    for column in fold_columns:
        if not benchmark[column].isin(['train','test']).all() or (benchmark.groupby('case_id')[column].nunique()!=1).any():
            raise ValueError(f'Patient crosses partitions or has invalid assignment: {column}')
    by_case=benchmark.drop_duplicates('case_id').set_index('case_id').loc[patients.case_id]
    heldout=(by_case[fold_columns]=='test').to_numpy()
    if not (heldout.sum(1)==1).all():raise ValueError('Each patient must be held out exactly once')
    folds=pd.Series(heldout.argmax(1),index=patients.case_id)
    base=source_root/'pathology/trident/20x_256px_0px_overlap'
    candidate=slides[['case_id','slide_id']].copy()
    for column,folder in [('uni_path','features_uni_v1'),('keep_path','features_keep'),('coord_path','patches')]:
        suffix='_patches.h5' if column=='coord_path' else '.h5'
        candidate[column]=candidate.slide_id.map(lambda s:str(base/folder/f'{s}{suffix}'))
    patch_counts=[]
    for row in candidate.itertuples(index=False):
        with h5py.File(row.coord_path,'r') as c:
            coordinates=c['coords'][:]
        if coordinates.ndim!=2 or coordinates.shape[1]!=2 or not len(coordinates):
            raise ValueError(f'Invalid coordinates: {row.coord_path}')
        for path,dim in [(row.uni_path,1024),(row.keep_path,768)]:
            with h5py.File(path,'r') as f:
                if f['features'].shape!=(len(coordinates),dim) or not np.array_equal(f['coords'][:],coordinates):
                    raise ValueError(f'Feature dimension or coordinate order mismatch: {path}')
        patch_counts.append(len(coordinates))
    candidate['n_patches']=patch_counts
    # Entropy-seeded, label-independent choice ONCE; the saved manifest supplies reproducibility.
    rng=np.random.default_rng()
    selected=[rng.choice(group.index.to_numpy()) for _,group in candidate.groupby('case_id',sort=True)]
    data=candidate.loc[selected].set_index('case_id').loc[patients.case_id].reset_index()
    data['event']=reference.loc[data.case_id,'OS_event'].to_numpy(dtype=int)
    data['time']=reference.loc[data.case_id,'OS_days'].to_numpy(dtype=float)
    data['split']=folds.loc[data.case_id].to_numpy(dtype=int)
    if set(data.split)!=set(range(5)):raise ValueError('All five patient folds are required')
    output_dir.mkdir(parents=True,exist_ok=True)
    data.to_csv(output_dir/'all_label.csv',index=False)
    candidate['selected']=candidate.slide_id.isin(data.slide_id)
    candidate['split']=candidate.case_id.map(folds)
    candidate.to_csv(output_dir/'slide_selection_audit.csv',index=False)
    dimensions={'uni':1024,'uni_l2':1024,'keep':768}
    signatures={source:feature_signature(data[column])for source,column in [('uni','uni_path'),('keep','keep_path')]}
    metadata=dict(source_root=str(source_root),source_hashes={str(p):file_hash(p)for p in source_paths},
                  cohort_sha256=file_hash(output_dir/'all_label.csv'),feature_signatures=signatures,
                  feature_dimensions=dimensions,n_patients=len(data),n_source_slides=len(candidate),
                  n_events=int(data.event.sum()),endpoint='OS',time_unit='days',
                  selection_method='One random WSI per patient, generated once without a fixed seed; frozen manifest',
                  split_method='Original PathoBench patient folds; test membership used as internal validation',
                  evaluation_protocol='internal_cv_with_validation_early_stopping')
    (output_dir/'preparation.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+'\n')
    data.groupby('split').agg(n=('case_id','size'),events=('event','sum')).to_csv(output_dir/'fold_counts.csv')
    print(f'Prepared {len(data)} patients / {len(candidate)} candidate slides / {int(data.event.sum())} OS events',flush=True)
    print(data.groupby('split').agg(n=('case_id','size'),events=('event','sum')).to_string(),flush=True)
    return data


class CPTACDataset(Dataset):
    def __init__(self,data_dir,feature_source='uni'):
        if feature_source not in {'uni','uni_l2','keep'}:raise ValueError('Unknown feature source')
        self.metadata=validate_preparation(data_dir)
        self.samples=pd.read_csv(Path(data_dir)/'all_label.csv',dtype={'case_id':str,'slide_id':str})
        self.feature_source=feature_source
        self.feature_dim=int(self.metadata['feature_dimensions'][feature_source])
        if self.samples.case_id.duplicated().any() or self.samples.slide_id.duplicated().any():
            raise ValueError('Exactly one unique WSI per patient is required')
        self.feature_column='keep_path' if feature_source=='keep' else 'uni_path'
        name='keep' if feature_source=='keep' else 'uni'
        if feature_signature(self.samples[self.feature_column])!=self.metadata['feature_signatures'][name]:
            raise ValueError('Selected feature files changed after preparation')
        for row in self.samples.itertuples(index=False):
            path=getattr(row,self.feature_column)
            with h5py.File(path,'r') as f:
                if f['features'].shape!=(row.n_patches,self.feature_dim):
                    raise ValueError(f'Selected feature shape changed: {path}')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self,index):
        row=self.samples.iloc[index]
        with h5py.File(row[self.feature_column],'r') as f:
            features=np.asarray(f['features'],dtype=np.float32)
        if not np.isfinite(features).all():raise ValueError(f'Non-finite features: {row.slide_id}')
        features=torch.from_numpy(features)
        if self.feature_source=='uni_l2':features=normalize(features,p=2,dim=-1,eps=1e-12)
        return features,torch.tensor(row.event,dtype=torch.long),torch.tensor(row.time,dtype=torch.float32),row.case_id


class CPTACEagleDataset(Dataset):
    """EAGLE top25 for exactly the previously frozen single WSI per patient."""
    def __init__(self,data_dir,eagle_root):
        self.metadata=validate_preparation(data_dir)
        self.samples=pd.read_csv(Path(data_dir)/'all_label.csv',dtype={'case_id':str,'slide_id':str})
        self.feature_dim,self.patch_count=1280,25
        root=Path(eagle_root).resolve()
        source=root/'inputs/slides.csv';manifest=root/'eagle/selection_manifest.csv'
        slides=pd.read_csv(source,dtype={'case_id':str,'slide_id':str})
        selections=pd.read_csv(manifest,dtype={'pa_id':str})
        if (self.samples.case_id.duplicated().any() or self.samples.slide_id.duplicated().any()
                or slides.slide_id.duplicated().any() or selections.pa_id.duplicated().any()):
            raise ValueError('Duplicate patient/slide identities in frozen selection or EAGLE tables')
        if not set(self.samples.slide_id)<=set(slides.slide_id)&set(selections.pa_id):
            raise ValueError('Frozen selected slides missing from EAGLE tables')
        reference=slides.set_index('slide_id').loc[self.samples.slide_id]
        if (not np.array_equal(reference.case_id.to_numpy(),self.samples.case_id.to_numpy())
                or not np.array_equal(reference.OS_event.to_numpy(),self.samples.event.to_numpy())
                or not np.array_equal(reference.OS_days.to_numpy(),self.samples.time.to_numpy())):
            raise ValueError('EAGLE patient identities or OS labels disagree with frozen cohort')
        roles=reference[[f'fold_{f}'for f in range(5)]]
        heldout=roles.eq('test').to_numpy()
        if (not roles.isin(['train','test']).all().all() or not (heldout.sum(1)==1).all()
                or not np.array_equal(heldout.argmax(1),self.samples.split.to_numpy())):
            raise ValueError('EAGLE patient folds disagree with frozen cohort')
        selected=selections.set_index('pa_id')
        paths=[]
        for row in self.samples.itertuples(index=False):
            path=root/'eagle/features'/f'{row.slide_id}.h5'
            entry=selected.loc[row.slide_id]
            if entry.selected!=25 or Path(entry.feature_path).resolve()!=path:
                raise ValueError(f'EAGLE Top25 manifest mismatch: {row.slide_id}')
            with h5py.File(path,'r') as handle:
                features=np.asarray(handle['features'],dtype=np.float32)
                if (features.shape!=(25,1280) or handle.attrs.get('top_k')!=25
                        or handle.attrs.get('downstream_encoder')!='Virchow2-CLS'):
                    raise ValueError(f'Expected EAGLE Virchow2-CLS features [25,1280]: {path}')
                if not np.isfinite(features).all():raise ValueError(f'Non-finite EAGLE features: {path}')
            paths.append(str(path))
        self.samples['eagle_path']=paths
        self.samples['n_patches']=25
        self.metadata['feature_signatures']['eagle_top25']=feature_signature(paths)
        self.metadata['source_hashes'].update({str(p):file_hash(p)for p in [source,manifest]})

    def __len__(self):
        return len(self.samples)

    def __getitem__(self,index):
        row=self.samples.iloc[index]
        with h5py.File(row.eagle_path,'r') as handle:features=np.asarray(handle['features'],dtype=np.float32)
        if features.shape!=(25,1280) or not np.isfinite(features).all():
            raise ValueError(f'Expected finite EAGLE features [25,1280]: {row.slide_id}')
        return torch.from_numpy(features),torch.tensor(row.event,dtype=torch.long),torch.tensor(row.time,dtype=torch.float32),row.case_id
