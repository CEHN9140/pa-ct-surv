import sys
import unittest
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))


class CTShapeAuxTests(unittest.TestCase):
    def test_original_initialization_rng_and_training_risk_match(self):
        from ct_shape_aux_model import CTWithShape
        from model.build import CT_Model
        torch.manual_seed(123);base=CT_Model(dropout=.25);state=torch.get_rng_state().clone()
        torch.manual_seed(123);model=CTWithShape(n_shape=1,dropout=.25)
        torch.testing.assert_close(torch.get_rng_state(),state,rtol=0,atol=0)
        for key,value in base.state_dict().items():torch.testing.assert_close(value,model.state_dict()[key],rtol=0,atol=0)
        torch.manual_seed(123);allshape=CTWithShape(n_shape=3,dropout=.25)
        torch.testing.assert_close(model.shape_head.weight,allshape.shape_head.weight[2:3],rtol=0,atol=0)
        x=torch.randn(3,1,16,16,16)
        base.train();model.train()
        torch.manual_seed(789);expected=base(x)
        torch.manual_seed(789);actual,shape=model(x)
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        self.assertEqual(shape.shape,(3,1))

    def test_lambda_zero_matches_baseline_updates_and_aux_head_is_head_lr(self):
        from ct_shape_aux_model import CTWithShape,auxiliary_objective,shape_optimizer
        from model.build import CT_Model
        from utils import build_ct_optimizer
        from loss import cox_loss
        torch.manual_seed(123);base=CT_Model(dropout=.25)
        torch.manual_seed(123);model=CTWithShape(n_shape=1,dropout=.25)
        model.set_shape_scaler(np.array([[.4],[.6],[.8]]))
        a=build_ct_optimizer(base,1e-4,5e-4,1e-5);b=shape_optimizer(model,1e-4,5e-4,1e-5)
        group_lr={id(p):g['lr'] for g in b.param_groups for p in g['params']}
        self.assertEqual(group_lr[id(model.shape_head.weight)],1e-4)
        self.assertEqual(group_lr[id(model.ct.conv1.weight)],1e-5)
        self.assertEqual(len(group_lr),len(list(model.parameters())))
        x=torch.randn(3,1,16,16,16);t=torch.tensor([1.,2.,3.]);e=torch.tensor([1.,0.,1.])
        base.train();model.train();initial=model.shape_head.weight.detach().clone()
        torch.manual_seed(999);loss=cox_loss(base(x),t,e);loss.backward();a.step();a.zero_grad()
        torch.manual_seed(999);r,s=model(x)
        values=auxiliary_objective(model,r,s,torch.tensor([[.4],[.6],[.8]]),t,e,0.)
        values['total'].backward();b.step();b.zero_grad()
        for key,value in base.state_dict().items():torch.testing.assert_close(value,model.state_dict()[key],rtol=0,atol=0)
        torch.testing.assert_close(initial,model.shape_head.weight,rtol=0,atol=0)

    def test_shape_gradient_updates_backbone_and_scaler_survives_reload(self):
        from ct_shape_aux_model import CTWithShape,auxiliary_objective
        model=CTWithShape(n_shape=3,dropout=.25)
        labels=np.array([[2.,10.,.2],[4.,20.,.4],[6.,30.,.6]])
        model.set_shape_scaler(labels)
        np.testing.assert_allclose(model.shape_mean.numpy(),labels.mean(0))
        np.testing.assert_allclose(model.shape_std.numpy(),labels.std(0),rtol=1e-6)
        model.train();buffers={k:v.clone() for k,v in model.ct.named_buffers()}
        r,s=model(torch.randn(3,1,16,16,16));loss=auxiliary_objective(model,r,s,torch.tensor(labels,dtype=torch.float32),torch.arange(1.,4.),torch.ones(3),.2)
        expected=((s-(torch.tensor(labels,dtype=torch.float32)-model.shape_mean)/model.shape_std)**2).mean()
        torch.testing.assert_close(loss['shape_mse'],expected)
        loss['shape_mse'].backward()
        self.assertGreater(float(model.ct.conv1.weight.grad.norm()),0.)
        self.assertIsNone(model.ct.fc[0].weight.grad)
        for key,value in model.ct.named_buffers():torch.testing.assert_close(value,buffers[key],rtol=0,atol=0)
        other=CTWithShape(n_shape=3);other.load_state_dict(model.state_dict(),strict=True)
        torch.testing.assert_close(other.shape_mean,model.shape_mean)
        torch.testing.assert_close(other.shape_std,model.shape_std)
        with self.assertRaises(ValueError):model.set_shape_scaler(np.ones((3,3)))

    def test_lambda_selected_on_held_scores_and_epochs_from_stop_only(self):
        from ct_shape_aux_model import select_auxiliary
        rows=[]
        for weight,score,epochs in [(0.,.7,[1,3,8]),(.05,.72,[4,8,12]),(.2,.71,[20,25,30])]:
            rows.extend(dict(weight=weight,inner_fold=i,held_cindex=score,selected_epoch=epoch) for i,epoch in enumerate(epochs))
        chosen=select_auxiliary(rows)
        self.assertEqual(chosen['weight'],.05);self.assertEqual(chosen['epochs'],8)
        for row in rows:row['held_cindex']=.7
        self.assertEqual(select_auxiliary(rows)['weight'],0.)
        with self.assertRaises(ValueError):select_auxiliary(rows[:-1])

    def test_conditional_sphere_labels_do_not_fit_held_shape(self):
        from ct_shape_aux_probe import conditional_sphere
        rng=np.random.default_rng(123);train=rng.normal(size=(80,3));held=rng.normal(size=(20,3))
        train[:,2]=.7*train[:,0]-.3*train[:,1]+.1*train[:,2]
        nuisance,a,b=conditional_sphere(train,held)
        before=nuisance.coef_.copy()
        _,again,changed=conditional_sphere(train,held+1000)
        np.testing.assert_array_equal(a,again)
        np.testing.assert_array_equal(before,nuisance.coef_)
        self.assertEqual(a.shape,(80,1));self.assertEqual(b.shape,(20,1))
        np.testing.assert_allclose(b[:,0],held[:,2]-nuisance.predict(held[:,:2]))

    def test_single_target_residual_probe_has_column_shape(self):
        from ct_shape_aux_probe import run_shape_probes
        rng=np.random.default_rng(123);x=rng.normal(size=(100,8));y=rng.normal(size=(100,3))
        y[:,2]=.3*y[:,0]-.2*y[:,1]+x[:,0]
        context=dict(train_ids=list(range(60)),val_ids=list(range(60,80)),train_ct=x[:60],val_ct=x[60:80],train_shapes=y[:60],val_shapes=y[60:80])
        with tempfile.TemporaryDirectory() as directory:
            scores,raw,residual=run_shape_probes([context],x[:80],x[80:],y[:80],y[80:],[.1,1.],Path(directory)/'probe')
        self.assertEqual(raw.shape,(20,3));self.assertEqual(residual.shape,(20,1));self.assertEqual(len(scores),4)
        self.assertTrue(np.isfinite([r['r2'] for r in scores]).all())


if __name__=='__main__':
    torch.set_num_threads(2)
    unittest.main()
