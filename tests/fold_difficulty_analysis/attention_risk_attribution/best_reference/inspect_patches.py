"""Best-reference contextual harm patches, technical matching and blinded native review."""
from pathlib import Path
import argparse,hashlib,json,sys
import h5py
import numpy as np
import pandas as pd
from PIL import Image,ImageDraw

BASE=Path(__file__).resolve().parent;PARENT=BASE.parent;ROOT=PARENT.parents[2]
sys.path.insert(0,str(PARENT))
from attribution import technical_qc
from quality import panel
from best_tools import context_scores,matched_controls,white_fraction


def review_cases(patients):
    rows=[]
    for (fold,split),group in patients.groupby(['fold','split']):
        eligible=group.dropna(subset=['actual_mean_pair_credit_change'])
        chosen=eligible.sort_values(['actual_mean_pair_credit_change','pa_id'],ascending=[split=='val',True]).head(2)
        for rank,(_,row) in enumerate(chosen.iterrows(),1):rows.append(dict(fold=int(fold),split=split,pa_id=row.pa_id,selection=('deteriorated' if split=='val' else 'improved')+str(rank)))
        stable=eligible[~eligible.pa_id.isin(chosen.pa_id)].sort_values(['actual_changed_pair_fraction','pa_id']).head(1)
        for _,row in stable.iterrows():rows.append(dict(fold=int(fold),split=split,pa_id=row.pa_id,selection='stable'))
    result=pd.DataFrame(rows);order=np.random.default_rng(123).permutation(len(result));result['review_case']=[f'R{int(i)+1:03d}' for i in order]
    return result


def main(smoke=False):
    import openslide
    out=BASE/('validation/smoke' if smoke else 'private')
    assert json.loads((out/'attribution_completed.json').read_text())['passed']
    patients=pd.read_csv(out/'patients.csv',dtype={'pa_id':str});cohort=pd.read_csv(ROOT/'data/trident/features_uni_v1/all_label_roi64.csv',dtype={'pa_id':str}).set_index('pa_id')
    planned=review_cases(patients);planned.to_csv(out/'review_cases.csv',index=False)
    review_lookup={(r.fold,r.pa_id):r for r in planned.itertuples()}
    selected=patients[(patients.split.eq('val'))|patients.apply(lambda r:(r.fold,r.pa_id) in review_lookup,axis=1)]
    reviewer=out/'reviewer';reviewer.mkdir(exist_ok=True);qc=[];coverage=[];keys=[];forms=[];matches_all=[]
    for number,row in enumerate(selected.itertuples(),1):
        original=cohort.loc[row.pa_id];case=review_lookup.get((row.fold,row.pa_id))
        with np.load(out/f'fold_{row.fold}/{row.pa_id}/best_attribution.npz',allow_pickle=False) as f:c=f['contributions'];a=f['best_attention'];b=f['late_attention']
        scores=context_scores(c,row);ranking=scores['harm_support'];basis='broken_pair_margin'
        if not np.any(ranking>0):
            if case is None:
                coverage.append(dict(fold=row.fold,split=row.split,pa_id=row.pa_id,basis='no_harm',candidates=0,matches=0,native_caliper_pass=0));continue
            ranking=np.abs(c);basis='abs_ig_review_fallback'
        with h5py.File(original.pa_path) as f,h5py.File(original.coord_path) as g:
            features=f['features'][:];coords=f['coords'][:];np.testing.assert_array_equal(coords,g['coords'][:]);attrs=dict(g['coords'].attrs)
            for key in ['name','patch_size_level0','patch_size','level0_width','level0_height']:
                if attrs[key]!=f['coords'].attrs[key]:raise ValueError('Metadata mismatch')
        if len(c)!=len(coords) or str(attrs['name'])!=row.pa_id:raise ValueError('Attribution/coordinates mismatch')
        thumb_path=Path(original.coord_path).parents[2]/'thumbnails'/f'{row.pa_id}.jpg'
        with Image.open(thumb_path) as im:thumbnail_rgb=np.asarray(im.convert('RGB'))
        width,height=int(attrs['level0_width']),int(attrs['level0_height']);side=int(round(float(attrs['patch_size_level0'])));target=int(attrs['patch_size'])
        if abs(thumbnail_rgb.shape[1]/thumbnail_rgb.shape[0]-width/height)>.01:raise ValueError('Thumbnail aspect ratio differs')
        blank=white_fraction(thumbnail_rgb,coords,side,(width,height))
        matched=matched_controls(ranking,features,blank);images={};native={}
        wsi=Path('/home/gly001/cqj/data/lung_cancer/pathology')/(row.pa_id+'.svs');before=wsi.stat()
        with openslide.OpenSlide(str(wsi)) as slide:
            if slide.dimensions!=(width,height) or target!=256:raise ValueError('Native geometry mismatch')
            exact=[i for i,v in enumerate(slide.level_downsamples) if abs(v-side/target)<1e-6]
            if not exact:raise ValueError('No exact native scale')
            for rank,match in enumerate(matched.itertuples(),1):
                record=dict(fold=row.fold,split=row.split,pa_id=row.pa_id,match_id=rank,basis=basis,
                    high_index=int(match.high_index),low_index=int(match.low_index),thumbnail_white_difference=match.white_difference,uni_cosine=match.uni_cosine)
                for group,index in [('high_context',int(match.high_index)),('matched_low',int(match.low_index))]:
                    xy=coords[index];rgba=slide.read_region(tuple(map(int,xy)),exact[0],(target,target)).convert('RGBA')
                    tile=Image.alpha_composite(Image.new('RGBA',rgba.size,'white'),rgba).convert('RGB');values=technical_qc(np.asarray(tile));images[(group,rank)]=tile
                    native[(group,rank)]=values
                    qc.append(dict(fold=row.fold,split=row.split,pa_id=row.pa_id,match_id=rank,basis=basis,group=group,patch_index=index,
                        contribution=float(c[index]),delta_attention=float(b[index]-a[index]),thumbnail_white=float(blank[index]),uni_norm=float(np.linalg.norm(features[index])),
                        **{key:float(value[index]) for key,value in scores.items()},**values))
                record['native_white_difference']=abs(native[('high_context',rank)]['white_fraction']-native[('matched_low',rank)]['white_fraction'])
                record['native_caliper_pass']=record['native_white_difference']<=.1+1e-12;matches_all.append(record)
            if case is not None and len(matched):
                directory=reviewer/case.review_case;directory.mkdir(exist_ok=True)
                rng=np.random.default_rng(int.from_bytes(hashlib.sha256(case.review_case.encode()).digest()[:8],'little'))
                for letter,group in zip('AB',rng.permutation(['high_context','matched_low'])):
                    order=rng.permutation(np.arange(1,len(matched)+1));panel([images[(group,int(i))] for i in order],letter).save(directory/f'{letter}_patches.png')
                    location=slide.get_thumbnail((1600,1600)).convert('RGB');draw=ImageDraw.Draw(location)
                    for tile,rank in enumerate(order,1):
                        match=matched.iloc[int(rank)-1];index=int(match.high_index if group=='high_context' else match.low_index);xy=coords[index]
                        x,y=xy[0]/width*location.width,xy[1]/height*location.height
                        draw.rectangle((x,y,x+side/width*location.width,y+side/height*location.height),outline='#008f86',width=2);draw.text((x,y),str(tile),fill='black',stroke_width=1,stroke_fill='white')
                        keys.append(dict(review_case=case.review_case,fold=row.fold,split=row.split,pa_id=row.pa_id,group=letter,tile=tile,match_id=int(rank),patch_index=index,strategy=group,basis=basis))
                        forms.append(dict(review_case=case.review_case,group=letter,tile=tile,tissue_type='',artifact='',quality='',notes=''))
                    location.save(directory/f'{letter}_locations.png')
        after=wsi.stat()
        if before.st_size!=after.st_size or before.st_mtime_ns!=after.st_mtime_ns:raise ValueError('WSI changed')
        current=[r for r in matches_all if r['fold']==row.fold and r['pa_id']==row.pa_id]
        coverage.append(dict(fold=row.fold,split=row.split,pa_id=row.pa_id,basis=basis,candidates=int(min(8,len(c)//2,np.sum(ranking>0))),
            matches=len(matched),native_caliper_pass=sum(int(r['native_caliper_pass']) for r in current)))
        if number%20==0 or number==len(selected):print(f'Best-reference native QC {number}/{len(selected)}',flush=True)
    pd.DataFrame(qc).to_csv(out/'patch_quality.csv',index=False);pd.DataFrame(matches_all).to_csv(out/'matches.csv',index=False)
    pd.DataFrame(coverage).to_csv(out/'matching_coverage.csv',index=False);pd.DataFrame(keys).to_csv(out/'review_key.csv',index=False)
    pd.DataFrame(forms).to_csv(reviewer/'review.csv',index=False)
    (out/'quality_completed.json').write_text(json.dumps(dict(passed=True,patches=len(qc),planned_review_cases=len(planned),rendered_review_cases=len(set(r['review_case'] for r in keys)),
        actual_tissue_type_matching=False,matching='same WSI; thumbnail white caliper0.1; nearest UNI cosine; actual white difference flagged',histology_review_pending=True),indent=2))
    print('Best-reference matching and QC complete',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');main(p.parse_args().smoke)
