"""Coarse background proxy on original thumbnails; not histological annotation."""
from pathlib import Path
import json
import h5py
import numpy as np
import pandas as pd
from PIL import Image

BASE=Path(__file__).resolve().parent;ROOT=BASE.parents[3]


def box_fraction(mask,coords,side,canvas):
    height,width=mask.shape;cw,ch=canvas;sx,sy=width/cw,height/ch
    x0=np.clip(np.floor(coords[:,0]*sx).astype(int),0,width);x1=np.clip(np.ceil((coords[:,0]+side)*sx).astype(int),0,width)
    y0=np.clip(np.floor(coords[:,1]*sy).astype(int),0,height);y1=np.clip(np.ceil((coords[:,1]+side)*sy).astype(int),0,height)
    integral=np.pad(mask.astype(np.int64).cumsum(0).cumsum(1),((1,0),(1,0)))
    count=integral[y1,x1]-integral[y0,x1]-integral[y1,x0]+integral[y0,x0]
    area=(x1-x0)*(y1-y0)
    if (area<=0).any():raise ValueError('Coordinate outside thumbnail')
    return count/area


def run():
    cohort=pd.read_csv(ROOT/'data/trident/features_uni_v1/all_label_roi64.csv',dtype={'pa_id':str})
    qc={};rows=[]
    for n,row in cohort.iterrows():
        with h5py.File(row.coord_path) as f:coords=f['coords'][:];attrs=dict(f['coords'].attrs)
        with h5py.File(row.pa_path) as f:
            if not np.array_equal(coords,f['coords'][:]):raise ValueError('Feature/coordinate order differs')
        thumb=Path(row.coord_path).parents[2]/'thumbnails'/f'{row.pa_id}.jpg'
        with Image.open(thumb) as image:rgb=np.asarray(image.convert('RGB'))
        width,height=int(attrs['level0_width']),int(attrs['level0_height'])
        if abs(rgb.shape[1]/rgb.shape[0]-width/height)>.01:raise ValueError('Thumbnail geometry mismatch')
        maximum=rgb.max(2);minimum=rgb.min(2)
        white=(minimum>=220)&((maximum.astype(float)-minimum)/np.maximum(maximum,1)<=.1)
        black=maximum<=30
        blank=box_fraction(white,coords,float(attrs['patch_size_level0']),(width,height))
        dark=box_fraction(black,coords,float(attrs['patch_size_level0']),(width,height))
        qc[row.pa_id]=(blank,dark)
        rows.append(dict(pa_id=row.pa_id,fold=int(row.split),n_patches=len(coords),uniform_white_fraction=float(blank.mean()),
            uniform_bg70=float((blank>=.7).mean()),uniform_bg80=float((blank>=.8).mean()),uniform_bg90=float((blank>=.9).mean()),uniform_dark=float(dark.mean()),
            thumb_width=rgb.shape[1],thumb_height=rgb.shape[0]))
        if (n+1)%100==0:print('Thumbnail background QC',n+1,'/721',flush=True)
    frame=pd.DataFrame(rows);frame.to_csv(BASE/'private/thumbnail_qc.csv',index=False)
    weighted=[]
    for batch in [32,64,128]:
        for fold in range(5):
            directory=BASE.parent/f'bs{batch}/diagnostics/fold_{fold}'
            cohort_fold=pd.read_csv(directory/'private_cohort.csv',dtype={'pa_id':str})
            last=json.loads((directory/'audit.json').read_text())['last_epoch']
            for epoch in [5,10,last]:
                with np.load(directory/f'private/epoch_{epoch:03d}.npz',allow_pickle=False) as f:
                    offsets=f['offsets'];attention=f['attention']
                for index in np.flatnonzero(cohort_fold.split.eq(fold)):
                    patient=cohort_fold.iloc[index].pa_id;blank,dark=qc[patient];start,end=offsets[index:index+2]
                    if end-start!=len(blank):raise ValueError('Attention/patch coordinate count differs')
                    weights=attention[start:end].astype(float);weights/=weights.sum();k=min(128,len(weights));top=np.argpartition(weights,-k)[-k:]
                    weighted.append(dict(batch=batch,fold=fold,epoch=epoch,phase_point='last' if epoch==last else str(epoch),pa_id=patient,
                        weighted_white_fraction=float(weights@blank),weighted_bg70=float(weights@(blank>=.7)),weighted_bg80=float(weights@(blank>=.8)),weighted_bg90=float(weights@(blank>=.9)),
                        weighted_dark=float(weights@dark),top128_bg80=float((blank[top]>=.8).mean())))
    pd.DataFrame(weighted).to_csv(BASE/'private/attention_qc.csv',index=False)
    pd.DataFrame(weighted).groupby(['batch','phase_point'])[['weighted_white_fraction','weighted_bg70','weighted_bg80','weighted_bg90','weighted_dark','top128_bg80']].mean().reset_index().to_csv(BASE/'attention_background_summary.csv',index=False)
    frame.groupby('fold')[['uniform_white_fraction','uniform_bg70','uniform_bg80','uniform_bg90','uniform_dark']].mean().to_csv(BASE/'thumbnail_background_summary.csv')
    print('QC completed; white/dark are coarse technical proxies, not tissue labels')


if __name__=='__main__':run()
