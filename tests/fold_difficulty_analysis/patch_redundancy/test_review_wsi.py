import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from PIL import Image
import pytest

sys.path.insert(0,str(Path(__file__).parent))


def test_case_selection_depends_only_on_patch_count_and_id():
    from review_wsi import choose_cases
    frame=pd.DataFrame([dict(pa_id=f'p{fold}_{i:02d}',split=fold,patch_count=100+i,
                             event=i%2,time=20+i) for fold in range(5) for i in range(10)])
    chosen=choose_cases(frame)
    assert len(chosen)==25 and chosen.groupby('split').size().eq(5).all()
    altered=frame.sample(frac=1,random_state=3).copy()
    altered.event=1-altered.event;altered.time=-altered.time
    assert chosen.pa_id.tolist()==choose_cases(altered).pa_id.tolist()
    assert chosen.loc[chosen.split.eq(0),'pa_id'].tolist()==['p0_01','p0_03','p0_04','p0_06','p0_08']


def write_h5_pair(tmp_path):
    coords=np.array([[0,0],[512,0],[1024,0],[0,512]],dtype=np.int64)
    features=np.arange(4*1024,dtype=np.float32).reshape(4,1024)
    attrs=dict(patch_size=256,patch_size_level0=512.,level0_width=2048,level0_height=1024,
               level0_magnification=40,target_magnification=20.,name='slide')
    for path in (tmp_path/'features.h5',tmp_path/'coords.h5',tmp_path/'chief.h5'):
        with h5py.File(path,'w') as f:
            is_chief=path.name=='chief.h5';idx=np.array([2,0]) if is_chief else np.arange(4)
            d=f.create_dataset('coords',data=coords[idx]);d.attrs.update(attrs)
            if path.name!='coords.h5':f.create_dataset('features',data=features[idx])
            if is_chief:
                f.create_dataset('selected_indices',data=idx);f.create_dataset('chief_scores',data=[.7,.2])
                f.attrs.update(source_uni=str(tmp_path/'features.h5'),source_coords=str(tmp_path/'coords.h5'),
                               source_wsi=str(tmp_path/'slide.svs'),selector='frozen-CHIEF',original_patch_count=4)
    return tmp_path/'features.h5',tmp_path/'coords.h5',tmp_path/'chief.h5',tmp_path/'slide.svs'


def test_exact_coordinate_and_chief_row_alignment_is_required(tmp_path):
    from review_wsi import load_aligned
    feature,coords,chief,wsi=write_h5_pair(tmp_path)
    result=load_aligned(feature,coords,chief,wsi,'slide')
    assert result[0].shape==(4,1024) and result[3].tolist()==[2,0]
    with h5py.File(coords,'r+') as f:f['coords'][:]=f['coords'][:][::-1]
    with pytest.raises(ValueError,match='coordinate order'):load_aligned(feature,coords,chief,wsi,'slide')


@pytest.mark.parametrize('mismatch',['features','fov','source','indices'])
def test_rejects_chief_feature_fov_source_and_index_mismatches(tmp_path,mismatch):
    from review_wsi import load_aligned
    feature,coords,chief,wsi=write_h5_pair(tmp_path)
    with h5py.File(chief,'r+') as f:
        if mismatch=='features':f['features'][0,0]+=1
        if mismatch=='fov':f['coords'].attrs['patch_size_level0']=256.
        if mismatch=='source':f.attrs['source_uni']='other.h5'
        if mismatch=='indices':f['selected_indices'][:]=[2,2]
    with pytest.raises(ValueError):load_aligned(feature,coords,chief,wsi,'slide')


def test_xy_projection_uses_true_thumbnail_dimensions():
    from review_wsi import project_box
    assert project_box((1000,500),512,(10000,4000),(1000,300))==pytest.approx((100,37.5,151.2,75.9))


def test_blinding_is_stable_per_case_with_all_strategies_and_no_duplicates():
    from review_wsi import blinded_groups
    selections={'top':np.arange(16),'bottom':np.arange(16,32),'random':np.arange(32,48),'chief':np.arange(48,64)}
    a=blinded_groups(selections,123,'slideA');b=blinded_groups(selections,123,'slideA')
    assert [r[0] for r in a]==list('ABCD')
    assert len({r[1] for r in a})==4
    assert any(not np.array_equal(r[2],selections[r[1]]) for r in a)
    assert all(x[0:2]==y[0:2] and np.array_equal(x[2],y[2]) for x,y in zip(a,b))


def test_tile_reading_uses_correct_level_and_montage_does_not_cover_tissue(tmp_path):
    from review_wsi import read_tile,tile_grid
    class Slide:
        level_downsamples=(1.,2.,4.)
        def read_region(self,location,level,size):
            assert location==(1024,512) and level==1 and size==(256,256)
            return Image.new('RGBA',size,(130,70,120,255))
    image=read_tile(Slide(),(1024,512),512,256)
    assert image.mode=='RGB' and image.size==(256,256)
    grid=tile_grid([image]*16,'A')
    assert grid.size==(1044,1216)
    assert grid.getpixel((12,70))==(130,70,120)
    grid.save(tmp_path/'A_tiles.png')
