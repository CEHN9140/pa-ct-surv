"""Native-FOV technical proxies, matched within WSI controls and blinded review panels."""
from pathlib import Path
import argparse,hashlib,json
import h5py
import numpy as np
import pandas as pd
from PIL import Image,ImageDraw,ImageFont
from matplotlib import get_data_path
from attribution import technical_qc

BASE=Path(__file__).resolve().parent;ROOT=BASE.parents[2]
WSI=Path('/home/gly001/cqj/data/lung_cancer/pathology')


def representative_cases(patients):
    rows=[]
    for fold,group in patients.groupby('fold'):
        eligible=group.dropna(subset=['temporal_mean_pair_credit_change'])
        if eligible.empty:continue
        worst=eligible.sort_values(['temporal_mean_pair_credit_change','pa_id']).head(2)
        for rank,(_,row) in enumerate(worst.iterrows(),1):
            rows.append(dict(pa_id=row.pa_id,fold=int(fold),selection=f'temporal_worst_{rank}',review_case=f'R{len(rows)+1:03d}'))
        stable=eligible[~eligible.pa_id.isin(worst.pa_id)].sort_values(['temporal_changed_pair_fraction','pa_id']).head(1)
        for _,row in stable.iterrows():rows.append(dict(pa_id=row.pa_id,fold=int(fold),selection='temporal_stable',review_case=f'R{len(rows)+1:03d}'))
    return pd.DataFrame(rows)


def panel(images,letter):
    canvas=Image.new('RGB',(1040,620),'white');draw=ImageDraw.Draw(canvas)
    font=ImageFont.truetype(str(Path(get_data_path())/'fonts/ttf/DejaVuSans-Bold.ttf'),20)
    draw.text((8,8),f'Group {letter}',fill='black',font=font)
    for i,tile in enumerate(images):
        x=(i%4)*260;y=40+(i//4)*290;canvas.paste(tile,(x,y));draw.text((x+5,y+257),str(i+1),fill='black',font=font)
    return canvas


def run(smoke=False):
    import openslide
    private=BASE/('validation/smoke' if smoke else 'private')
    if not json.loads((private/'attribution_completed.json').read_text())['passed']:raise ValueError('Attribution incomplete')
    patches=pd.read_csv(private/'selected_patches.csv',dtype={'pa_id':str});patients=pd.read_csv(private/'patients.csv',dtype={'pa_id':str})
    cohort=pd.read_csv(ROOT/'data/trident/features_uni_v1/all_label_roi64.csv',dtype={'pa_id':str}).set_index('pa_id')
    planned=representative_cases(patients);planned.to_csv(private/'review_cases.csv',index=False)
    review_dir=private/'reviewer';review_dir.mkdir(exist_ok=True);rows=[];keys=[];forms=[];geometry=[]
    for count,(patient,selected) in enumerate(patches.groupby('pa_id',sort=False),1):
        original=cohort.loc[patient];wsi=WSI/(patient+'.svs');stat=wsi.stat()
        with h5py.File(original.coord_path) as f:
            attrs=dict(f['coords'].attrs);expected=f['coords'][:][selected.patch_index.to_numpy()]
        np.testing.assert_array_equal(expected,selected[['x','y']].to_numpy())
        case=planned[planned.pa_id.eq(patient)];images={};local=[]
        with openslide.OpenSlide(str(wsi)) as slide:
            width,height=slide.dimensions
            if (width,height)!=(int(attrs['level0_width']),int(attrs['level0_height'])):raise ValueError('WSI geometry mismatch')
            side=int(round(float(attrs['patch_size_level0'])));target=int(attrs['patch_size'])
            if target!=256:raise ValueError('Unexpected model patch pixels')
            matches=[i for i,v in enumerate(slide.level_downsamples) if abs(v-side/target)<1e-6]
            if not matches:raise ValueError('No exact native pyramid scale for TRIDENT FOV')
            level=matches[0]
            for row in selected.itertuples():
                rgba=slide.read_region((int(row.x),int(row.y)),level,(target,target)).convert('RGBA')
                tile=Image.alpha_composite(Image.new('RGBA',rgba.size,'white'),rgba).convert('RGB')
                fraction=max(0,min(side,width-row.x))*max(0,min(side,height-row.y))/(side*side)
                values=dict(pa_id=patient,fold=int(row.fold),patch_index=int(row.patch_index),group=row.group,
                    contribution=row.contribution,abs_contribution=row.abs_contribution,valid_bounds_fraction=fraction,**technical_qc(np.asarray(tile)))
                rows.append(values);local.append(values)
                if len(case):images[int(row.patch_index)]=tile
            if len(case):
                case_id=case.iloc[0].review_case;directory=review_dir/case_id;directory.mkdir(exist_ok=True)
                digest=int.from_bytes(hashlib.sha256(('review:'+patient).encode()).digest()[:8],'little');rng=np.random.default_rng(digest)
                groups=rng.permutation(['high_abs_ig','low_abs_ig']);thumbnail=slide.get_thumbnail((1600,1600)).convert('RGB')
                for letter,group in zip('AB',groups):
                    subset=selected[selected.group.eq(group)].iloc[rng.permutation(sum(selected.group.eq(group)))]
                    panel([images[int(i)] for i in subset.patch_index],letter).save(directory/f'{letter}_patches.png')
                    locations=thumbnail.copy();draw=ImageDraw.Draw(locations)
                    for number,row in enumerate(subset.itertuples(),1):
                        x=row.x/width*locations.width;y=row.y/height*locations.height
                        draw.rectangle((x,y,x+side/width*locations.width,y+side/height*locations.height),outline='#009b92',width=2)
                        draw.text((x,y),str(number),fill='black',stroke_width=1,stroke_fill='white')
                        keys.append(dict(review_case=case_id,group=letter,tile=number,pa_id=patient,patch_index=int(row.patch_index),strategy=group))
                        forms.append(dict(review_case=case_id,group=letter,tile=number,tissue_type='',artifact='',quality='',notes=''))
                    locations.save(directory/f'{letter}_locations.png')
        after=wsi.stat()
        if stat.st_size!=after.st_size or stat.st_mtime_ns!=after.st_mtime_ns:raise ValueError('WSI changed during extraction')
        geometry.append(dict(pa_id=patient,level=level,side_level0=side,target=target,width=width,height=height,wsi_size=stat.st_size,wsi_mtime_ns=stat.st_mtime_ns))
        if count%20==0 or count==patients.shape[0]:
            pd.DataFrame(rows).to_csv(private/'patch_quality.csv',index=False);print(f'Original-patch QC {count}/{len(patients)}',flush=True)
    frame=pd.DataFrame(rows);frame.to_csv(private/'patch_quality.csv',index=False)
    if len(frame)!=len(patches):raise ValueError('Some controls missing')
    pd.DataFrame(geometry).to_csv(private/'wsi_geometry.csv',index=False);pd.DataFrame(keys).to_csv(private/'review_key.csv',index=False)
    pd.DataFrame(forms).to_csv(review_dir/'review.csv',index=False)
    (private/'quality_completed.json').write_text(json.dumps(dict(passed=True,patients=len(patients),patches=len(frame),blinded_review_cases=len(planned),
        actual_native_tiles=True,within_wsi_controls=True,histology_automatically_labeled=False),indent=2))
    print('Quality and blinded review preparation complete',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');run(p.parse_args().smoke)
