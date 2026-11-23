from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
from fit import load_matrix


def test_both_representation_types_enter_cox_as_float64_without_value_change(tmp_path):
    ids=np.array(['a','b']);x=np.arange(1536,dtype=np.float32).reshape(2,768)/100
    p=tmp_path/'x.npz';np.savez_compressed(p,features=x,case_ids=ids)
    out=load_matrix(p,ids)
    assert out.dtype==np.float64
    np.testing.assert_array_equal(out,x.astype(np.float64))
