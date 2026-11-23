from pathlib import Path
import sys
import numpy as np,pandas as pd
sys.path.insert(0,str(Path(__file__).parent))
from abmil_curve import epoch_steps,choose_epoch,fit_epochs


def test_epoch_step_grid_changes_updates_but_not_patient_passes():
    assert epoch_steps(85,[1,2,30],64)==[1,2,30]
    assert epoch_steps(256,[1,2,30],64)==[4,8,120]
    assert epoch_steps(577,[1,2,30],64)==[9,18,270]
    rows=pd.DataFrame([dict(inner_fold=i,epoch=e,cindex=.6) for i in range(3) for e in [1,2]])
    assert choose_epoch(rows,[1,2])[0]==1


def test_training_complete_epochs_tracks_exposure_without_outer_evaluation(tmp_path):
    from nested_train import fresh_model
    import torch
    bags=[torch.randn(3,5) for _ in range(12)]
    c=pd.DataFrame(dict(time=np.arange(1,13),event=np.ones(12,int),pa_id=[str(i) for i in range(12)]))
    model=fresh_model(5,123,'cpu')
    metadata,curves=fit_epochs(model,bags,c,np.arange(12),None,[2],123,tmp_path,4)
    assert metadata['patient_exposures']==24 and metadata['optimizer_steps']==6 and metadata['evaluation_calls']==0
    assert not curves
