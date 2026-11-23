from pathlib import Path
import sys
import pandas as pd
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).parent))
from report import aggregate

def fixture():
    return pd.DataFrame([dict(model=m,fold=f,size_label=s,repeat=r,n_train=128 if s=='128' else 577,val_cindex=.6+.1*r) for m in ['abmil','titan'] for f in range(5) for s in ['128','full'] for r in (range(3) if s=='128' else range(1))])

def test_repeat_first_not_independent_patient_evidence():
    folds,summary=aggregate(fixture(),['128','full'],['val_cindex'])
    assert len(folds)==20 and len(summary)==4
    assert np.allclose(summary.loc[summary.size_label=='128','val_cindex_mean'],.7)
    assert np.allclose(summary.val_cindex_sd,0)

def test_missing_repeat_rejected():
    with pytest.raises(ValueError):aggregate(fixture().iloc[1:],['128','full'],['val_cindex'])

def test_missing_auc_preserved():
    x=fixture();x['auc_60m']=.7;x.loc[0,'auc_60m']=np.nan
    _,summary=aggregate(x,['128','full'],['auc_60m'])
    assert np.isnan(summary.loc[(summary.model=='abmil')&(summary.size_label=='128'),'auc_60m_mean'].iloc[0])
