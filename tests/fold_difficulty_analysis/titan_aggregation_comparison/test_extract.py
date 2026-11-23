from pathlib import Path
import sys
import h5py
import numpy as np
import pytest
import torch
sys.path.insert(0,str(Path(__file__).parent))
from extract import read_bag,encode


def make(path,coords=None):
    x=np.arange(3*768,dtype=np.float32).reshape(3,768)/100
    coords=np.array([[0,0],[1024,0],[0,1024]]) if coords is None else coords
    with h5py.File(path,'w') as f:
        f['features']=x;d=f.create_dataset('coords',data=coords)
        d.attrs.update(patch_size=512,patch_size_level0=1024.,target_magnification=20.)
    return x


def test_exact_mean_and_integer_level0_geometry(tmp_path):
    p=tmp_path/'bag.h5';x=make(p)
    values,coords,step,mean,meta=read_bag(p)
    assert step==1024 and values.shape==(3,768)
    np.testing.assert_array_equal(mean,x.mean(axis=0,dtype=np.float64))
    assert meta['patches']==3


def test_grid_collisions_fail_instead_of_merging_tokens(tmp_path):
    p=tmp_path/'bad.h5';make(p,np.array([[0,0],[1,0],[0,1024]]))
    with pytest.raises(ValueError,match='grid'):read_bag(p)


def test_official_api_receives_same_unmodified_features_and_coords(tmp_path):
    p=tmp_path/'bag.h5';x=make(p);values,coords,step,mean,meta=read_bag(p)
    class Fake(torch.nn.Module):
        def encode_slide_from_patch_features(self,f,c,s):
            assert not self.training and not torch.is_grad_enabled()
            assert f.shape==(1,3,768) and c.shape==(1,3,2) and s==1024
            np.testing.assert_array_equal(f.cpu().numpy()[0],x)
            return f.mean(1)
    m=Fake().eval();result=encode(m,values,coords,step,'cpu')
    np.testing.assert_allclose(result,x.mean(axis=0),atol=1e-5)


def test_local_dynamic_module_staging_preserves_nested_official_imports(tmp_path):
    from extract import stage_python_modules
    source=tmp_path/'weights';source.mkdir();cache=tmp_path/'modules'
    (source/'modeling_titan.py').write_text('from .child import X\n')
    (source/'child.py').write_text('from .leaf import X\n')
    (source/'leaf.py').write_text('X = 1\n')
    target=stage_python_modules(source,cache)
    from transformers.dynamic_module_utils import get_relative_import_files
    files=get_relative_import_files(target/'modeling_titan.py')
    assert {Path(p).name for p in files}=={'child.py','leaf.py'}
    assert (target/'leaf.py').read_bytes()==(source/'leaf.py').read_bytes()


@pytest.mark.parametrize('field,value',[('patch_size',256),('target_magnification',10),('patch_size_level0',1024.5)])
def test_wrong_patch_recipe_cannot_be_silently_used(tmp_path,field,value):
    p=tmp_path/'bag.h5';make(p)
    with h5py.File(p,'r+') as f:f['coords'].attrs[field]=value
    with pytest.raises(ValueError):read_bag(p)
