"""Outcome-independent native QC and private fold-local WSI cluster maps."""
from pathlib import Path
import argparse,json,sys
import h5py
import numpy as np
import pandas as pd
from PIL import Image,ImageDraw,ImageFont
import matplotlib
from matplotlib import get_data_path

BASE=Path(__file__).resolve().parent;ROOT=BASE.parents[2];ATTR=BASE.parent/'attention_risk_attribution'
sys.path.insert(0,str(ATTR))
from attribution import technical_qc


def cluster_map(thumbnail,coords,labels,side,dimensions,k,fold):
    image=thumbnail.convert('RGBA');image.thumbnail((1600,1600));overlay=Image.new('RGBA',image.size);draw=ImageDraw.Draw(overlay)
    palette=[tuple(int(v*255) for v in matplotlib.colormaps['tab20'](i)[:3]) for i in range(k)]
    width,height=dimensions
    for xy,label in zip(coords,labels):
        x,y=xy[0]/width*image.width,xy[1]/height*image.height
        box=(x,y,min(image.width,(xy[0]+side)/width*image.width),min(image.height,(xy[1]+side)/height*image.height))
        draw.rectangle(box,fill=(*palette[int(label)],85),outline=(*palette[int(label)],180),width=1)
    blended=Image.alpha_composite(image,overlay).convert('RGB');canvas=Image.new('RGB',(max(1000,blended.width),blended.height+95),'white');canvas.paste(blended,(0,95))
    text=ImageDraw.Draw(canvas);font=ImageFont.truetype(str(Path(get_data_path())/'fonts/ttf/DejaVuSans.ttf'),16)
    text.text((8,5),f'Fold {fold}: local UNI classes; colors are not matched across folds',font=font,fill='black')
    for c,color in enumerate(palette):
        x=8+(c%10)*98;y=34+(c//10)*27;text.rectangle((x,y,x+15,y+15),fill=color);text.text((x+20,y),f'C{c:02d}',font=font,fill='black')
    return canvas


def main(smoke=False):
    import openslide
    out=BASE/('validation/smoke' if smoke else 'private');status=json.loads((out/'assigned.json').read_text());assert status['passed']
    source=pd.read_csv(ROOT/'data/trident/features_uni_v1/all_label_roi64.csv',dtype={'pa_id':str}).set_index('pa_id')
    clusters=pd.read_csv(out/'patient_clusters.csv',dtype={'pa_id':str});patients=clusters.pa_id.drop_duplicates().tolist();prototypes=[]
    for fold in range(status['folds']):
        proto=pd.read_csv(out/f'fold_{fold}/prototypes.csv',dtype={'pa_id':str});proto['fold']=fold;proto['rank']=proto.groupby('cluster').cumcount();prototypes.append(proto)
    prototypes=pd.concat(prototypes,ignore_index=True)
    for row in prototypes.itertuples():
        if source.loc[row.pa_id].split==row.fold:raise ValueError('Val patch used as Train prototype')
    prototype_images=out/'prototype_images';prototype_images.mkdir(exist_ok=True);maps=out/'wsi_maps';maps.mkdir(exist_ok=True)
    qc=[];geometry=[];map_count=0
    for number,patient in enumerate(patients,1):
        row=source.loc[patient]
        with h5py.File(row.coord_path) as f:coords=f['coords'][:];attrs=dict(f['coords'].attrs)
        width,height=int(attrs['level0_width']),int(attrs['level0_height']);side=int(round(float(attrs['patch_size_level0'])));target=int(attrs['patch_size'])
        with np.load(out/f'technical/{patient}.npz',allow_pickle=False) as f:sample=f['qc_indices']
        chosen=prototypes[prototypes.pa_id.eq(patient)];needed=sorted(set(sample.tolist())|set(chosen.patch_index.tolist()))
        wsi=Path('/home/gly001/cqj/data/lung_cancer/pathology')/(patient+'.svs');before=wsi.stat()
        with openslide.OpenSlide(str(wsi)) as slide:
            if slide.dimensions!=(width,height) or target!=256 or str(attrs['name'])!=patient:raise ValueError('Native geometry differs')
            exact=[i for i,v in enumerate(slide.level_downsamples) if abs(v-side/target)<1e-6]
            if not exact:raise ValueError('No exact native pyramid FOV')
            for index in needed:
                rgba=slide.read_region(tuple(map(int,coords[index])),exact[0],(target,target)).convert('RGBA')
                tile=Image.alpha_composite(Image.new('RGBA',rgba.size,'white'),rgba).convert('RGB')
                if index in sample:qc.append(dict(pa_id=patient,patch_index=int(index),**technical_qc(np.asarray(tile))))
                for p in chosen[chosen.patch_index.eq(index)].itertuples():
                    with np.load(out/f'fold_{p.fold}/{patient}/assignment.npz',allow_pickle=False) as f:label=int(f['labels'][index])
                    if label!=p.cluster:raise ValueError('Prototype class no longer matches')
                    destination=prototype_images/f'fold_{p.fold}/cluster_{p.cluster:02d}';destination.mkdir(parents=True,exist_ok=True)
                    tile.save(destination/f'{p.rank}.png')
        after=wsi.stat()
        if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):raise ValueError('WSI changed')
        own_fold=int(row.split);assignment=out/f'fold_{own_fold}/{patient}/assignment.npz'
        if assignment.exists():
            with np.load(assignment,allow_pickle=False) as f:labels=f['labels']
            with Image.open(Path(row.coord_path).parents[2]/'thumbnails'/f'{patient}.jpg') as im:thumb=im.convert('RGB')
            cluster_map(thumb,coords,labels,side,(width,height),status['classes_per_fold'],own_fold).save(maps/(patient+'.jpg'),quality=92);map_count+=1
        geometry.append(dict(pa_id=patient,width=width,height=height,side_level0=side,target=target,level=exact[0],wsi_size=before.st_size,wsi_mtime_ns=before.st_mtime_ns))
        if number%50==0 or number==len(patients):print(f'Native random QC and WSI maps {number}/{len(patients)}',flush=True)
    pd.DataFrame(qc).to_csv(out/'random_patch_quality.csv',index=False);pd.DataFrame(geometry).to_csv(out/'wsi_geometry.csv',index=False)
    for (fold,k),group in prototypes.groupby(['fold','cluster']):
        canvas=Image.new('RGB',(1024,300),'white');draw=ImageDraw.Draw(canvas);draw.text((8,8),f'Fold {fold} local C{k:02d}: {len(group)} nearest Train samples; not tissue labels',fill='black')
        for n,p in enumerate(group.itertuples()):
            with Image.open(prototype_images/f'fold_{fold}/cluster_{k:02d}/{p.rank}.png') as image:canvas.paste(image.convert('RGB'),(n*256,40))
        canvas.save(prototype_images/f'fold_{fold}/cluster_{k:02d}.jpg',quality=95)
    if not smoke and map_count!=721:raise ValueError('Missing OOF WSI maps')
    (out/'qc_completed.json').write_text(json.dumps(dict(passed=True,unique_patients=len(patients),random_native_patches=len(qc),oof_maps=map_count,train_prototype_patches=len(prototypes),
        selection_independent_of_outcomes_and_ig=True,histology_labels=False),indent=2))
    print('Native QC and fold-local visualizations complete',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');main(p.parse_args().smoke)
