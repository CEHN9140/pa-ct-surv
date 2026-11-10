"""Derive KEEP inputs from the frozen GL UNI cohort without resampling patients."""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from gl_dataset import GLDataset,file_hash,feature_signature


def prepare_keep(reference_dir,feature_dir,output_dir):
    reference_dir,feature_dir,output_dir=map(lambda p:Path(p).resolve(),[reference_dir,feature_dir,output_dir])
    reference=GLDataset(reference_dir)
    if feature_dir.name!='features_keep':raise ValueError('Expected the TRIDENT features_keep directory')
    reference_csv=reference_dir/'all_label.csv';destination=output_dir/'all_label.csv'
    if destination.exists():
        existing=GLDataset(output_dir)
        if (existing.metadata.get('reference_cohort_csv')!=str(reference_csv)
                or existing.metadata['feature_dir']!=str(feature_dir)):
            raise ValueError('Existing KEEP data uses another reference cohort or feature directory')
        pd.testing.assert_frame_equal(existing.samples[['pa_id','event','time','split']],
                                      reference.samples[['pa_id','event','time','split']])
        print(f'Reusing frozen KEEP inputs: {destination}',flush=True)
        return existing.samples
    if output_dir.exists()and any(output_dir.iterdir()):
        raise ValueError('Incomplete KEEP preparation; refusing to overwrite existing files')
    samples=reference.samples.copy();paths=[]
    for row in samples.itertuples(index=False):
        path=feature_dir/f'{row.pa_id}.h5'
        with h5py.File(row.pa_path,'r')as original,h5py.File(path,'r')as keep:
            if 'features'not in keep or keep['features'].shape!=(row.n_patches,768):
                raise ValueError(f'Expected KEEP features [{row.n_patches},768]: {path}')
            if ('coords'not in original or 'coords'not in keep or
                    not np.array_equal(original['coords'][:],keep['coords'][:])):
                raise ValueError(f'UNI/KEEP patch coordinate order differs: {row.pa_id}')
        paths.append(str(path))
    samples['pa_path']=paths;samples['feature_dim']=768
    output_dir.mkdir(parents=True,exist_ok=True);samples.to_csv(destination,index=False)
    metadata=dict(reference.metadata)
    metadata.update(feature_dir=str(feature_dir),feature_encoder='keep',feature_dim=768,
                    feature_signature=feature_signature(paths),cohort_sha256=file_hash(destination),
                    reference_cohort_csv=str(reference_csv),reference_cohort_sha256=file_hash(reference_csv),
                    split_method='Copied original patient IDs, labels and saved folds; no resampling')
    (output_dir/'preparation.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+'\n')
    print(f'Prepared KEEP: {len(samples)} patients | {int(samples.event.sum())} events | 768-D | Same saved folds',flush=True)
    print(samples.groupby('split').agg(n=('pa_id','size'),events=('event','sum')).to_string(),flush=True)
    return samples


def main():
    here=Path(__file__).resolve().parent
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference_dir',default=str(here/'data_154_eventtime'))
    parser.add_argument('--feature_dir',default='/home/gly001/cqj/data/yixianai/GL/pathology/trident/20x_256px_0px_overlap/features_keep')
    parser.add_argument('--output_dir',default=str(here/'data_154_eventtime_keep'))
    args=parser.parse_args();prepare_keep(args.reference_dir,args.feature_dir,args.output_dir)


if __name__=='__main__':main()
