"""Prepare blinded WSI patch review; never infer tissue labels or train models."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import h5py
from matplotlib import get_data_path
import numpy as np
import pandas as pd
from PIL import Image,ImageDraw,ImageFont
import torch

from diagnose import ROOT,sha256,select_indices
from model.abmil_cox import ABMIL
from utils import seed_everything


def choose_cases(frame):
    if frame.pa_id.duplicated().any() or set(frame.split)!=set(range(5)):
        raise ValueError('Expected unique patients in original five folds')
    rows=[]
    for fold in range(5):
        group=frame.loc[frame.split.eq(fold)].sort_values(['patch_count','pa_id'])
        if len(group)<5: raise ValueError('Need at least five patients per fold')
        for quantile in (.1,.3,.5,.7,.9):
            row=group.iloc[int(np.rint((len(group)-1)*quantile))].copy()
            row['selection_quantile']=quantile;rows.append(row)
    result=pd.DataFrame(rows).reset_index(drop=True)
    result['review_case']=[f'R{i+1:03d}' for i in range(len(result))]
    return result


def load_aligned(feature_path,coord_path,chief_path,wsi_path,case_id):
    fields=('patch_size','patch_size_level0','level0_width','level0_height',
            'level0_magnification','target_magnification','name')
    with h5py.File(feature_path) as f,h5py.File(coord_path) as c,h5py.File(chief_path) as e:
        features=f['features'][:];coords=f['coords'][:];attrs=dict(c['coords'].attrs)
        if features.ndim!=2 or features.shape!=(len(coords),1024) or not np.isfinite(features).all():
            raise ValueError('Invalid complete raw UNI features')
        if coords.ndim!=2 or coords.shape[1]!=2 or not np.issubdtype(coords.dtype,np.integer) \
                or (coords<0).any() or len(np.unique(coords,axis=0))!=len(coords): raise ValueError('Invalid coordinates')
        if not np.array_equal(coords,c['coords'][:]): raise ValueError('Feature / coordinate order mismatch')
        for field in fields:
            if field not in attrs or field not in f['coords'].attrs or field not in e['coords'].attrs \
                    or attrs[field]!=f['coords'].attrs[field] or attrs[field]!=e['coords'].attrs[field]:
                raise ValueError(f'Coordinate FOV metadata mismatch: {field}')
        if str(attrs['name'])!=str(case_id) or Path(wsi_path).stem!=str(case_id): raise ValueError('WSI identity mismatch')
        if attrs['patch_size']<=0 or attrs['patch_size_level0']<=0 \
                or not np.isclose(attrs['patch_size_level0'],attrs['patch_size']*attrs['level0_magnification']/attrs['target_magnification']):
            raise ValueError('Invalid coordinate magnification or patch field of view')
        for key,path in [('source_uni',feature_path),('source_coords',coord_path),('source_wsi',wsi_path)]:
            if key not in e.attrs or Path(e.attrs[key]).resolve()!=Path(path).resolve(): raise ValueError('CHIEF source mismatch')
        if e.attrs.get('selector')!='frozen-CHIEF' or int(e.attrs['original_patch_count'])!=len(coords):
            raise ValueError('CHIEF selector provenance mismatch')
        indices=e['selected_indices'][:];scores=e['chief_scores'][:]
        if not np.issubdtype(indices.dtype,np.integer) or indices.ndim!=1 or len(np.unique(indices))!=len(indices) \
                or (indices<0).any() or (indices>=len(coords)).any() or scores.shape!=indices.shape or not np.isfinite(scores).all():
            raise ValueError('Invalid CHIEF selected indices / scores')
        if not np.array_equal(coords[indices],e['coords'][:]) or not np.array_equal(features[indices],e['features'][:]):
            raise ValueError('CHIEF coordinates or source UNI feature values mismatch')
    return features,coords,attrs,indices,scores


def blinded_groups(selections,seed,case_id):
    words=np.frombuffer(hashlib.sha256(('blinded-review:'+str(case_id)).encode()).digest()[:16],dtype='<u4').tolist()
    rng=np.random.default_rng(np.random.SeedSequence([seed,*words]))
    strategies=list(selections)
    return [(letter,strategy,rng.permutation(selections[strategy]))
            for letter,strategy in zip('ABCD',rng.permutation(strategies))]


def project_box(xy,side,slide_dimensions,thumbnail_dimensions):
    width,height=slide_dimensions;tw,th=thumbnail_dimensions;x,y=xy
    return (max(0,x/width*tw),max(0,y/height*th),min(tw,(x+side)/width*tw),min(th,(y+side)/height*th))


def read_tile(slide,xy,side,target):
    ratio=side/target
    matching=[i for i,d in enumerate(slide.level_downsamples) if abs(d-ratio)<=.1]
    level=matching[0] if matching else max(i for i,d in enumerate(slide.level_downsamples) if d<=ratio)
    read_size=round(side/slide.level_downsamples[level])
    rgba=slide.read_region(tuple(map(int,xy)),level,(read_size,read_size)).convert('RGBA')
    # Display transparent outside-slide pixels as white, not artificial black tissue.
    image=Image.alpha_composite(Image.new('RGBA',rgba.size,'white'),rgba).convert('RGB')
    return image.resize((target,target),Image.Resampling.BICUBIC) if image.size!=(target,target) else image


def tile_grid(images,group):
    if len(images)!=16 or any(image.size!=(256,256) for image in images): raise ValueError('Expected 16 native 256px tiles')
    canvas=Image.new('RGB',(1044,1216),'white');draw=ImageDraw.Draw(canvas)
    font=ImageFont.truetype(str(Path(get_data_path())/'fonts/ttf/DejaVuSans-Bold.ttf'),22)
    draw.text((8,8),f'Group {group} | 16 patches',fill='black',font=font)
    for i,image in enumerate(images):
        x=4+(i%4)*260;y=48+(i//4)*292
        canvas.paste(image,(x,y));draw.text((x+4,y+259),f'{i+1:02d}',fill='black',font=font)
    return canvas


def location_map(thumbnail,coords,side,dimensions,group):
    image=thumbnail.copy();draw=ImageDraw.Draw(image)
    font=ImageFont.truetype(str(Path(get_data_path())/'fonts/ttf/DejaVuSans-Bold.ttf'),14)
    for i,xy in enumerate(coords):
        box=project_box(xy,side,dimensions,image.size)
        draw.rectangle(box,outline='#008a8a',width=2)
        draw.text((box[0],max(0,box[1]-16)),str(i+1),fill='black',stroke_width=1,stroke_fill='white',font=font)
    result=Image.new('RGB',(image.width,image.height+40),'white');result.paste(image,(0,40))
    ImageDraw.Draw(result).text((8,8),f'Group {group} | level-0 locations',fill='black',font=font)
    return result


@torch.inference_mode()
def render_case(row,model,wsi_dir,chief_dir,output,seed):
    import openslide
    output=Path(output);case_dir=output/'reviewer'/row.review_case;case_dir.mkdir(parents=True)
    wsi=Path(wsi_dir)/(row.pa_id+'.svs');chief=Path(chief_dir)/(row.pa_id+'.h5')
    paths=[Path(row.pa_path),Path(row.coord_path),chief]
    hashes={str(path.resolve()):sha256(path) for path in paths}
    stat=wsi.stat();wsi_stat=dict(size=stat.st_size,mtime_ns=stat.st_mtime_ns)
    features,coords,attrs,chief_indices,chief_scores=load_aligned(row.pa_path,row.coord_path,chief,wsi,row.pa_id)
    if len(coords)!=int(row.patch_count) or len(chief_indices)<16: raise ValueError('Patch count or CHIEF review availability differs')
    x=torch.from_numpy(features).to(next(model.parameters()).device)
    risk,_,weights,h=model(x,return_patch_features=True);logits=model.attention(h)[0,:,0]
    if not np.isclose(risk.item(),np.float32(row.full_risk),atol=2e-6,rtol=2e-6): raise ValueError('Stage6 Full risk replay mismatch')
    selections={name:select_indices(logits,name,16,seed,row.pa_id,0).cpu().numpy() for name in ('top','bottom','random')}
    selections['chief']=chief_indices[np.argsort(-chief_scores,kind='stable')[:16]]
    all_attention=weights[0,:,0].cpu().numpy()
    key=[];review=[];previews=[]
    with openslide.OpenSlide(str(wsi)) as slide:
        dimensions=tuple(map(int,slide.dimensions))
        if dimensions!=(int(attrs['level0_width']),int(attrs['level0_height'])): raise ValueError('WSI dimensions differ from coordinate metadata')
        side=round(float(attrs['patch_size_level0']));target=int(attrs['patch_size'])
        if target!=256 or side!=512: raise ValueError('Expected audited 20x256px / level0 512px cohort')
        exact_levels=[i for i,value in enumerate(slide.level_downsamples) if value==side/target]
        if not exact_levels: raise ValueError('Audited cohort requires an exact 2x pyramid level for faithful TRIDENT reads')
        thumbnail=slide.get_thumbnail((1800,1800)).convert('RGB')
        thumbnail.save(case_dir/'wsi.png')
        for group,strategy,indices in blinded_groups(selections,seed,row.pa_id):
            images=[read_tile(slide,coords[index],side,target) for index in indices]
            grid=tile_grid(images,group);grid.save(case_dir/f'{group}_tiles.png')
            mapping=location_map(thumbnail,coords[indices],side,dimensions,group)
            mapping.save(case_dir/f'{group}_locations.png')
            small_grid=grid.resize((520,round(grid.height*520/grid.width)),Image.Resampling.LANCZOS)
            small_map=mapping.resize((520,round(mapping.height*520/mapping.width)),Image.Resampling.LANCZOS)
            panel=Image.new('RGB',(520,small_grid.height+small_map.height),'white')
            panel.paste(small_grid,(0,0));panel.paste(small_map,(0,small_grid.height));previews.append(panel)
            for tile,index in enumerate(indices,1):
                xy=coords[index];valid_width=max(0,min(side,dimensions[0]-int(xy[0])))
                valid_height=max(0,min(side,dimensions[1]-int(xy[1])))
                key.append(dict(review_case=row.review_case,group=group,tile=tile,pa_id=row.pa_id,fold=int(row.split),
                                strategy=strategy,feature_index=int(index),x_level0=int(xy[0]),y_level0=int(xy[1]),
                                patch_size_level0=side,attention=float(all_attention[index]),
                                valid_bounds_fraction=valid_width*valid_height/(side*side)))
                review.append(dict(review_case=row.review_case,group=group,tile=f'{tile:02d}',dominant_tissue='',
                                   artifact='',quality='',notes=''))
        overview=Image.new('RGB',(2080,max(p.height for p in previews)),'white')
        for i,panel in enumerate(previews):overview.paste(panel,(520*i,0))
        overview.save(case_dir/'overview.jpg',quality=95)
        geometry=dict(level0_width=dimensions[0],level0_height=dimensions[1],patch_size_level0=side,
                      target_patch_size=target,nominal_magnification=float(attrs['target_magnification']),
                      read_level=exact_levels[0],read_downsample=float(slide.level_downsamples[exact_levels[0]]),
                      mpp_x=slide.properties.get('openslide.mpp-x'),mpp_y=slide.properties.get('openslide.mpp-y'))
    if any(sha256(path)!=value for path,value in hashes.items()): raise RuntimeError('Review feature/coordinate inputs changed')
    current=wsi.stat()
    if current.st_size!=stat.st_size or current.st_mtime_ns!=stat.st_mtime_ns: raise RuntimeError('WSI changed during reading')
    audit=dict(review_case=row.review_case,fold=int(row.split),n_patches=len(coords),n_review_tiles=len(key),
               full_risk_max_abs_error=abs(risk.item()-np.float32(row.full_risk)),coordinate_order_exact=True,
               chief_indices_coords_features_exact=True,**geometry)
    source=dict(sha256=hashes,wsi_path=str(wsi),wsi_stat=wsi_stat)
    return key,review,audit,source


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv',type=Path,default=ROOT/'data/trident/features_uni_v1/all_label_roi64.csv')
    parser.add_argument('--reference',type=Path,default=Path(__file__).parent.parent/'pathology_stage6/results')
    parser.add_argument('--diagnostic',type=Path,default=Path(__file__).parent/'results')
    parser.add_argument('--wsi_dir',type=Path,default=Path('/home/gly001/cqj/data/lung_cancer/pathology'))
    parser.add_argument('--chief_dir',type=Path,default=Path('/home/gly001/cqj/EAGLE/output/pa_ct_surv/chief_top25/features'))
    parser.add_argument('--output',type=Path,default=Path(__file__).parent/'wsi_review')
    parser.add_argument('--seed',type=int,default=123)
    parser.add_argument('--device',default='cuda:0')
    args=parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()): raise FileExistsError('Use a new output; existing review images are not overwritten')
    (args.output/'private').mkdir(parents=True);(args.output/'reviewer').mkdir()
    status=dict(status='running',pid=os.getpid(),completed_cases=0,training_started=False,tissue_review_completed=False)
    sources={}
    def track(path):
        path=Path(path).resolve();value=sha256(path)
        if str(path) in sources and sources[str(path)]!=value: raise RuntimeError('Source changed during review preparation')
        sources[str(path)]=value;return path
    code_paths=[Path(__file__),Path(__file__).with_name('diagnose.py'),ROOT/'model/abmil_cox.py',ROOT/'utils.py']
    code_hashes={str(path.relative_to(ROOT)):sha256(path) for path in code_paths}
    try:
        (args.output/'status.json').write_text(json.dumps(status,indent=2))
        torch.set_num_threads(4);seed_everything(args.seed)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        reference=json.loads(track(args.reference/'validation.json').read_text())
        diagnosis=json.loads(track(args.diagnostic/'validation.json').read_text())
        if reference['status']!='completed' or diagnosis['status']!='completed' or not diagnosis['full_replay_verified']:
            raise ValueError('Completed Stage6 and frozen patch diagnosis required')
        if sha256(args.csv)!=reference['source_sha256'][str(args.csv.resolve())]: raise ValueError('Cohort differs from Stage6')
        for path,value in code_hashes.items():
            if path in reference['code_sha256'] and value!=reference['code_sha256'][path]: raise ValueError('ABMIL source differs from Stage6')
        frame=pd.read_csv(track(args.csv),dtype={'pa_id':str,'ct_id':str})
        counts=[];full=[]
        for fold in range(5):
            attention=pd.read_csv(track(args.diagnostic/f'private/fold_{fold}_attention.csv'),dtype={'case_id':str})
            risk=pd.read_csv(track(args.reference/f'fold_{fold}/private/outer/val_predictions.csv'),dtype={'case_id':str})
            expected=set(frame.loc[frame.split.eq(fold),'pa_id'])
            if set(attention.case_id)!=expected or set(risk.case_id)!=expected or attention.case_id.duplicated().any() or risk.case_id.duplicated().any():
                raise ValueError('Review source fold cohort differs')
            counts.append(attention[['case_id','patch_count']]);full.append(risk[['case_id','risk_score']])
        frame=frame.merge(pd.concat(counts).rename(columns={'case_id':'pa_id'}),on='pa_id',validate='one_to_one')
        frame=frame.merge(pd.concat(full).rename(columns={'case_id':'pa_id','risk_score':'full_risk'}),on='pa_id',validate='one_to_one')
        selected=choose_cases(frame);selected.to_csv(args.output/'private/planned_cases.csv',index=False)
        provenance=Path(__file__).parent.parent/'pathology_stage5/results/private/feature_provenance.json'
        if sha256(provenance)!=reference['source_sha256'][str(provenance.resolve())]: raise ValueError('Raw UNI provenance differs')
        original_features=json.loads(track(provenance).read_text())
        keys,reviews,audits,case_sources=[],[],[],[]
        print('Blinded WSI review: 25 predefined validation cases; four groups x 16 tiles; no training or tissue diagnosis',flush=True)
        for fold in range(5):
            checkpoint=track(args.reference/f'fold_{fold}/private/outer/model.pth')
            state=torch.load(checkpoint,map_location=args.device,weights_only=True)
            model=ABMIL(in_dim=1024,dropout=.25).to(args.device);model.load_state_dict(state['model'],strict=True)
            model.eval();model.requires_grad_(False)
            before={key:value.clone() for key,value in model.state_dict().items()}
            for row in selected.loc[selected.split.eq(fold)].itertuples():
                if sha256(row.pa_path)!=original_features[row.pa_path]: raise ValueError('Raw UNI features differ from Stage6')
                key,review,audit,source=render_case(row,model,args.wsi_dir,args.chief_dir,args.output,args.seed)
                keys.extend(key);reviews.extend(review);audits.append(audit);case_sources.append(source)
                pd.DataFrame(keys).to_csv(args.output/'private/selection_key.csv',index=False)
                pd.DataFrame(keys)[['review_case','group','strategy']].drop_duplicates().to_csv(args.output/'private/group_key.csv',index=False)
                pd.DataFrame(reviews).to_csv(args.output/'reviewer/review.csv',index=False)
                pd.DataFrame(audits).to_csv(args.output/'private/case_qc.csv',index=False)
                status['completed_cases']+=1
                (args.output/'status.json').write_text(json.dumps(status,indent=2))
                print(f"Prepared {row.review_case}: {status['completed_cases']}/25, coordinate/CHIEF alignment and Full replay passed",flush=True)
            if not all(torch.equal(before[key],value) for key,value in model.state_dict().items()) or any(p.grad is not None for p in model.parameters()):
                raise RuntimeError('Frozen review model changed')
            del model
        if len(keys)!=1600 or len(reviews)!=1600 or len(audits)!=25: raise RuntimeError('Incomplete review grid')
        for source in case_sources:
            if any(sha256(path)!=value for path,value in source['sha256'].items()): raise RuntimeError('Review input changed after rendering')
            stat=Path(source['wsi_path']).stat()
            if dict(size=stat.st_size,mtime_ns=stat.st_mtime_ns)!=source['wsi_stat']: raise RuntimeError('WSI source changed')
        if any(sha256(path)!=value for path,value in sources.items()) or any(sha256(ROOT/path)!=value for path,value in code_hashes.items()):
            raise RuntimeError('Source or code changed during rendering')
        (args.output/'private/source_manifest.json').write_text(json.dumps(dict(sources=sources,cases=case_sources),indent=2))
        guide=['# Blinded patch review','',
               'Do not open private/. Groups A-D are independently randomized in each case; image order does not encode attention rank.',
               'Each native PNG grid contains 16 numbered patches; the corresponding location map uses the same numbers.',
               'Overview JPGs are previews only. Use the full PNGs for assessment. Images retain their original stain colors.',
               'Nominal field: 20x, 256px patch (512 level-0 pixels). No DFS labels, risks or strategy names are shown.','',
               'Fill review.csv; blank cells mean unreviewed. Use uncertain when interpretation is unclear.',
               'dominant_tissue: tumor / stroma / necrosis / normal / mixed / other / uncertain',
               'artifact: none / blur / fold / pen / contamination / other / uncertain',
               'quality: adequate / poor / uncertain. Use notes for mixed findings.',
               'These are manual review fields, not machine-generated tissue diagnoses.','',
               'The same patch can appear in more than one group. Please record each presented tile independently.','',
               '|Case|Overview|Full-size grids and maps|','|---|---|---|']
        for case in selected.review_case:
            links=' '.join(f'[{g} tiles]({case}/{g}_tiles.png) / [{g} map]({case}/{g}_locations.png)' for g in 'ABCD')
            guide.append(f'|{case}|[Preview]({case}/overview.jpg)|{links}|')
        (args.output/'reviewer/REVIEW_GUIDE.md').write_text('\n'.join(guide)+'\n')
        completion=dict(status='completed',n_cases=25,n_outer_folds=5,groups_per_case=4,patches_per_group=16,n_presented_patches=1600,
                        neural_fits=0,full_risk_max_abs_error=max(row['full_risk_max_abs_error'] for row in audits),
                        coordinate_order_exact=True,chief_indices_coords_features_exact=True,model_parameters_unchanged=True,
                        tissue_review_completed=False,source_sha256_verified=True,wsi_size_mtime_verified=True,
                        code_sha256=code_hashes,seed=args.seed)
        (args.output/'completion.json').write_text(json.dumps(completion,indent=2))
        (args.output/'report.md').write_text('# WSI组织审查材料准备完成\n\n'
            '25例：每个Outer Val按patch数10/30/50/70/90百分位秩选择5例，与Risk/结局无关。\n'
            '四组各16个，共1600个展示patch；组间自然重叠允许。\n'
            '25例UNI特征/坐标逐行一致，CHIEF索引、坐标及UNI值完全对应，视野为level0 512px。\n'
            '使用Stage6冻结模型，Full Risk复现通过，未训练网络、未判读组织标签。\n\n'
            '本地reviewer/为去病例标识、隐藏策略的原图位置和拼图及空白审查表；private/为解盲键和来源。\n'
            '**材料生成完成不代表组织学审查完成；不能据此断言高Attention落在肿瘤或低Attention为伪影。**\n')
        status['status']='completed'
        print('Review materials completed: 25 cases / 1600 presented tiles. Tissue review remains pending.',flush=True)
    except Exception as exc:
        status.update(status='failed',error=f'{type(exc).__name__}: {exc}');raise
    finally:
        (args.output/'status.json').write_text(json.dumps(status,indent=2))


if __name__=='__main__':main()
