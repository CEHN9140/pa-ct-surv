import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import torch


class ResidualFollowupTests(unittest.TestCase):
    def test_control_matches_initial_pathology_weights_and_rng_but_removes_ct(self):
        from residual_teacher import ResidualTeacher
        from residual_pathology_control import MatchedPathologyControl
        torch.manual_seed(123);teacher=ResidualTeacher(dropout=.25);state=torch.get_rng_state()
        torch.manual_seed(123);control=MatchedPathologyControl(dropout=.25)
        torch.testing.assert_close(torch.get_rng_state(),state,rtol=0,atol=0)
        for key,value in teacher.pa_branch.state_dict().items():
            torch.testing.assert_close(value,control.pa_branch.state_dict()[key],rtol=0,atol=0)
        self.assertIsNone(control.ct_backbone)
        self.assertIsNone(control.ct_delta)
        self.assertTrue(all(name.startswith('pa_branch.') for name,_ in control.named_parameters()))
        # CT is not needed for this negative control.
        total,delta,path=control(None,torch.randn(3,25,1280))
        torch.testing.assert_close(total,path,rtol=0,atol=0)
        torch.testing.assert_close(delta,torch.zeros_like(delta),rtol=0,atol=0)

    def test_first_update_pathology_gradients_match_zero_correction_teacher(self):
        from residual_teacher import ResidualTeacher,residual_objective
        from residual_pathology_control import MatchedPathologyControl
        torch.manual_seed(123);teacher=ResidualTeacher(dropout=.25).train()
        torch.manual_seed(123);control=MatchedPathologyControl(dropout=.25).train()
        ct=torch.rand(3,1,16,16,16);pa=torch.randn(3,25,1280)
        t=torch.tensor([1.,2.,3.]);e=torch.tensor([1,0,1])
        for model in (teacher,control):
            torch.manual_seed(777)
            r,d,p=model(ct,pa)
            residual_objective(r,d,p,t,e,.3,0.)['total'].backward()
        for (name,left),(other,right) in zip(teacher.pa_branch.named_parameters(),control.pa_branch.named_parameters()):
            self.assertEqual(name,other)
            torch.testing.assert_close(left.grad,right.grad,rtol=0,atol=0)

    def test_three_epoch_mean_uses_complete_windows_and_ignores_outer_scores(self):
        from diagnose_residual_epochs import epoch_choices
        history=pd.DataFrame(dict(epoch=[1,2,3,4,5],val_cindex=[.9,.5,.5,.7,.8],outer_score=[1.,0.,0.,0.,0.]))
        result=epoch_choices(history)
        self.assertEqual(result['raw_epoch'],1)
        self.assertEqual(result['rolling3_epoch'],5)
        history.outer_score=[0.,0.,1.,0.,0.]
        self.assertEqual(epoch_choices(history),result)
        short=epoch_choices(history.iloc[:2])
        self.assertIsNone(short['rolling3_epoch'])

    def test_shared_trainer_accepts_control_without_ct_and_preserves_objective(self):
        from residual_train import fit_teacher
        from residual_pathology_control import MatchedPathologyControl
        from torch.utils.data import TensorDataset
        rng=torch.Generator().manual_seed(123)
        dataset=TensorDataset(torch.rand(5,1,16,16,16,generator=rng),torch.randn(5,25,1280,generator=rng),
            torch.ones(5,dtype=torch.long),torch.arange(1.,6.),torch.arange(5))
        args=SimpleNamespace(seed=123,ct_pretrained_path=None,dropout=.25,lr=1e-4,ct_backbone_lr=1e-5,
            weight_decay=5e-4,num_workers=0,cox_batch_size=4,num_epochs=1,patience=10,lambda_pa=.3)
        with tempfile.TemporaryDirectory() as folder:
            model,meta=fit_teacher(dataset,np.arange(5),None,args,torch.device('cpu'),folder,0.,
                                   fixed_epochs=1,model_factory=MatchedPathologyControl)
            history=pd.read_csv(Path(folder)/'training_history.csv')
            np.testing.assert_allclose(history.train_total,1.3*history.train_fused_cox,rtol=1e-6)
            self.assertEqual(history.train_correction_l2.iloc[0],0.)
            self.assertIn('CT removed',meta['initialization'])
            self.assertTrue(all(name.startswith('pa_branch.') for name,_ in model.named_parameters()))


if __name__=='__main__':
    torch.set_num_threads(2)
    unittest.main()
