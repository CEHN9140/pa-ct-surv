from pathlib import Path
import sys,hashlib
import numpy as np
import h5py,pytest,torch,pandas as pd
sys.path.insert(0,str(Path(__file__).parent))
from run import load_aligned,fresh_model,inner_roles


def test_raw_conch_matches_reference_hash_without_added_normalization(tmp_path):
    x=np.arange(3*768,dtype=np.float16).reshape(3,768)/100
    coords=np.array([[0,0],[1024,0],[0,1024]],dtype=np.int64);p=tmp_path/'x.h5'
    with h5py.File(p,'w') as f:
        f['features']=x;d=f.create_dataset('coords',data=coords);d.attrs.update(patch_size=512,patch_size_level0=1024)
    reference=dict(patches=3,patch_size_lv0=1024,feature_values_sha256=hashlib.sha256(x.tobytes()).hexdigest(),coords_sha256=hashlib.sha256(coords.tobytes()).hexdigest())
    bag,stat=load_aligned(p,reference);np.testing.assert_array_equal(bag.numpy(),x.astype(np.float32));assert bag.dtype==torch.float32
    with h5py.File(p,'r+') as f:f['features'][0,0]=999
    with pytest.raises(ValueError,match='values differ'):load_aligned(p,reference)


def test_768_model_is_fresh_seeded_and_inner_roles_match_reference():
    a=fresh_model(768,123,'cpu');b=fresh_model(768,123,'cpu')
    assert a.compression[0].in_features==768
    for key in a.state_dict():assert torch.equal(a.state_dict()[key],b.state_dict()[key])
    samples=pd.DataFrame(dict(split=np.arange(30)%5,event=np.arange(30)%2))
    train=np.flatnonzero(samples.split.ne(1));roles=inner_roles(samples,train,123)
    from sklearn.model_selection import StratifiedKFold
    expected=np.full(len(train),-1)
    for i,(_,held) in enumerate(StratifiedKFold(3,shuffle=True,random_state=123).split(np.zeros(len(train)),samples.iloc[train].event)):expected[held]=i
    np.testing.assert_array_equal(roles,expected)
