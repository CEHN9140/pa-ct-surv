import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
from torch.utils.data import DataLoader, TensorDataset

from increment_cox import IncrementCox, select_increment, stable_increment
from feature_increment import extract_ct_features
from model.build import CT_Model
from risk_calibration import paired_comparison
from shape_increment import mask_shape, volume_probe


class FeatureIncrementTests(unittest.TestCase):
    def data(self):
        rng = np.random.default_rng(123)
        p, x = rng.normal(size=100), rng.normal(size=(100, 40))
        t = rng.exponential(np.exp(-.5*p-.6*x[:,0]))+.01
        e = rng.binomial(1,.7,100)
        return p,x,t,e

    def test_m0_m1_m2_calibrate_pathology_and_only_penalize_ct(self):
        p,x,t,e=self.data()
        for ct,dim,ridge,n in [(None,None,0.,1),(x[:,:1],None,0.,2),(x,16,10.,17)]:
            model=IncrementCox(dim=dim,ridge=ridge).fit(p,ct,t,e)
            self.assertEqual(len(model.model.coef_),n)
            self.assertEqual(model.model.alpha[0],0.)
            np.testing.assert_array_equal(model.model.alpha[1:],np.full(n-1,ridge))
            np.testing.assert_allclose(model.predict(p,ct),model.design(p,ct)@model.model.coef_)
            self.assertEqual(model.model.ties,'breslow')

    def test_pca_and_scaling_are_fit_on_training_only(self):
        p,x,t,e=self.data()
        model=IncrementCox(dim=16,ridge=10.).fit(p[:70],x[:70],t[:70],e[:70])
        before=model.ct_pipeline.named_steps['scale'].mean_.copy()
        components=model.ct_pipeline.named_steps['pca'].components_.copy()
        model.predict(p[70:]+1e4,x[70:]+1e4)
        np.testing.assert_allclose(before,x[:70].mean(0))
        np.testing.assert_array_equal(before,model.ct_pipeline.named_steps['scale'].mean_)
        np.testing.assert_array_equal(components,model.ct_pipeline.named_steps['pca'].components_)

    def test_tuning_requires_disjoint_patients(self):
        p,x,t,e=self.data()
        context=dict(train_pa=p[:70],train_ct=x[:70],train_time=t[:70],train_event=e[:70],
                     val_pa=p[70:],val_ct=x[70:],val_time=t[70:],val_event=e[70:],
                     train_ids=np.arange(70),val_ids=np.arange(70,100))
        selected=select_increment([context],dimensions=[16],penalties=[1.,10.])
        self.assertEqual(selected['dim'],16)
        self.assertIn(selected['ridge'],[1.,10.])
        broken=dict(context,val_ids=np.arange(30))
        with self.assertRaisesRegex(ValueError,'overlap'):
            select_increment([broken],dimensions=[16],penalties=[1.])

    def test_stability_gate_requires_both_references(self):
        comparison={'fusion_minus_M0':{'delta_cindex':.01,'lower':.001},
                    'fusion_minus_M1':{'delta_cindex':.01,'lower':.001}}
        self.assertTrue(stable_increment(comparison,{'M0':4,'M1':4},5))
        self.assertFalse(stable_increment(comparison,{'M0':4,'M1':3},5))
        comparison['fusion_minus_M1']['lower']=-.001
        self.assertFalse(stable_increment(comparison,{'M0':5,'M1':5},5))

    def test_generic_paired_references_keep_old_interface(self):
        t=np.arange(1.,41.)
        frame=pd.DataFrame({'dfs.month':t,'dfs.status':np.tile([1,0],20),
                            'm0':-t,'m1':t,'risk_score':-t})
        result=paired_comparison([frame],repeats=50,reference_columns={'M0':'m0','M1':'m1'})
        self.assertEqual(result['fusion_minus_M0']['delta_cindex'],0.)
        self.assertEqual(result['fusion_minus_M1']['delta_cindex'],1.)

    def test_ct_extraction_returns_512_before_risk_head(self):
        model=CT_Model(dropout=.25).eval()
        loader=DataLoader(TensorDataset(torch.rand(2,1,16,16,16),torch.tensor([1,0]),
                          torch.tensor([1.,3.]),torch.arange(2)),batch_size=2)
        calls=[]
        handle=model.ct.fc.register_forward_hook(lambda *args:calls.append(1))
        ids,features=extract_ct_features(model,loader,'cpu')
        handle.remove()
        self.assertEqual(features.shape,(2,512))
        self.assertEqual(ids,['0','1'])
        self.assertFalse(calls)
        self.assertTrue(np.isfinite(features).all())

    def test_shape_uses_physical_spacing_and_keeps_all_components(self):
        a=np.zeros((14,14,14),dtype=np.uint8);a[2:7,2:7,2:7]=1;a[9:11,9:11,9:11]=1
        mask=sitk.GetImageFromArray(a);mask.SetSpacing((.7,1.1,1.5))
        first=mask_shape(mask)
        self.assertAlmostEqual(first['volume_mm3'],int(a.sum())*.7*1.1*1.5)
        mask.SetSpacing((1.4,2.2,3.))
        second=mask_shape(mask)
        self.assertAlmostEqual(second['volume_mm3']/first['volume_mm3'],8.)
        self.assertAlmostEqual(second['diameter_mm']/first['diameter_mm'],2.)
        self.assertAlmostEqual(second['sphericity'],first['sphericity'])
        a[0,0,0]=2
        with self.assertRaises(ValueError):mask_shape(sitk.GetImageFromArray(a))
        with self.assertRaises(ValueError):mask_shape(sitk.GetImageFromArray(np.zeros_like(a)))

    def test_volume_probe_evaluates_held_out_volume(self):
        rng=np.random.default_rng(123);x=rng.normal(size=(100,8));y=2*x[:,0]+.01*rng.normal(size=100)
        context=dict(train_ct=x[:70],val_ct=x[70:],train_volume=y[:70],val_volume=y[70:],
                     train_ids=np.arange(70),val_ids=np.arange(70,100))
        result=volume_probe([context],x[:70],y[:70],x[70:],y[70:],dimensions=[None],penalties=[.01,1.])
        self.assertGreater(result['r2_log_volume'],.99)
        self.assertEqual(len(result['prediction']),30)

    def test_volume_probe_uses_stable_double_precision_for_float32_features(self):
        rng=np.random.default_rng(123)
        x=rng.normal(size=(48,64)).astype(np.float32)
        x[:,32:]=x[:,:32]
        y=2*x[:,0]+.1*rng.normal(size=48)
        context=dict(train_ct=x[:32],val_ct=x[32:],train_volume=y[:32],val_volume=y[32:],
                     train_ids=np.arange(32),val_ids=np.arange(32,48))
        result=volume_probe([context],x[:32],y[:32],x[32:],y[32:],penalties=[.01])
        self.assertEqual(result['model'].coef_.dtype,np.dtype('float64'))
        self.assertEqual(result['model'].solver,'svd')
        self.assertTrue(np.isfinite(result['prediction']).all())


if __name__=='__main__':
    torch.set_num_threads(2)
    unittest.main()
