import sys
from pathlib import Path
import numpy as np
import torch
sys.path[:0]=[str(Path(__file__).parent),str(Path(__file__).parents[3])]
from model.build import Pa_Model
from subbag import subset_indices,cached_predictions


def test_sampling_is_fixed_without_replacement_and_keeps_order():
    for n in [1,7,101]:
        a=subset_indices(n,'p',3);b=subset_indices(n,'p',3)
        np.testing.assert_array_equal(a,b)
        assert len(a)==int(np.ceil(.8*n)) and len(np.unique(a))==len(a)
        assert np.all(a[:-1]<a[1:])


def test_cached_eval_equals_direct_full_and_subbags():
    torch.manual_seed(123);torch.set_num_threads(1)
    model=Pa_Model(feature_dim=1024,abmil_dropout=.25).eval();x=torch.randn(41,1024)
    observed=cached_predictions(model,x,'p',repeats=4)
    with torch.no_grad():
        expected=[float(model(x)[0])]+[float(model(x[subset_indices(len(x),'p',r)])[0]) for r in range(4)]
    np.testing.assert_allclose(observed,expected,atol=2e-7,rtol=2e-6)
    assert not any(m.training for m in model.modules())


def test_publication_excludes_patient_arrays_and_weights(tmp_path):
    from publish_completed import publication_paths
    files=publication_paths(tmp_path)
    assert len(files)==len(set(files))
    assert all(p.is_relative_to(tmp_path) for p in files)
    assert not any('private' in p.parts or p.suffix in ['.pth','.npz','.npy'] for p in files)
    assert tmp_path/'results/report.md' in files
