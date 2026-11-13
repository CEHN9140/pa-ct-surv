import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import SimpleITK as sitk


class CTPhenotypeTests(unittest.TestCase):
    def test_recover_exact_clipped_integer_intensity(self):
        from ct_phenotype_features import recover_intensity
        values=np.arange(-1000,401,dtype=np.float32)
        stored=(values+1000)/1400
        np.testing.assert_array_equal(recover_intensity(stored),values)
        for bad in [np.array([-1.]),np.array([1.1]),np.array([np.nan]),np.array([.1234567])]:
            with self.assertRaises(ValueError):recover_intensity(bad)

    def test_small_predefined_feature_set_matches_numpy_first_order(self):
        from ct_phenotype_features import region_features,feature_groups
        z,y,x=np.indices((16,16,16));image=((x+y+z)*25-700).astype(np.float32)
        mask=np.zeros_like(image,dtype=bool);mask[4:12,4:12,4:12]=True
        result=region_features(image,mask)
        self.assertEqual(len(result),12)
        self.assertAlmostEqual(result['mean'],float(image[mask].mean()))
        self.assertAlmostEqual(result['std'],float(image[mask].std()),places=5)
        self.assertAlmostEqual(result['p10'],float(np.quantile(image[mask],.1)))
        self.assertAlmostEqual(result['p90'],float(np.quantile(image[mask],.9)))
        self.assertTrue(np.isfinite(list(result.values())).all())
        changed=image.copy();changed[~mask]=315
        again=region_features(changed,mask)
        for key in result:self.assertAlmostEqual(result[key],again[key],places=10)
        self.assertEqual({k:len(v) for k,v in feature_groups().items()},
                         {'Shape':3,'FirstOrder':6,'Texture':6,'Peritumoral':12,'Combined':27})

    def test_shell_uses_observed_roi_and_excludes_padding(self):
        from ct_phenotype_features import peritumor_shell
        mask=np.zeros((64,64,64),dtype=bool);mask[30:34,30:34,30:34]=True
        valid=np.ones_like(mask)
        shell,info=peritumor_shell(mask,valid)
        self.assertEqual(info['coverage'],1.)
        self.assertFalse((shell&mask).any())
        mask[:]=False;mask[30:34,30:34,:4]=True
        shell,info=peritumor_shell(mask,valid)
        self.assertLess(info['coverage'],1.)
        valid[:,:,:6]=False
        shell,less=peritumor_shell(mask,valid)
        self.assertLess(less['coverage'],info['coverage'])
        self.assertFalse((shell&~valid).any())

    def test_available_grid_matches_physical_audit_with_permuted_axes(self):
        from ct_phenotype_features import roi_support
        from ct_input_audit import sample_grid
        image=sitk.Image([44,52,36],sitk.sitkInt16);image.SetSpacing((.7,1.,1.5))
        image.SetDirection((0.,-1.,0.,1.,0.,0.,0.,0.,1.))
        center=np.array([22,20,15])
        result=roi_support(image.GetSize(),image.GetSpacing(),image.GetDirection(),center)
        oriented=sitk.DICOMOrient(image,'LPS')
        _,expected=sample_grid(oriented,(center-32)[::-1],[64]*3)
        np.testing.assert_array_equal(result,expected)

    def test_ct_only_scaling_and_conditional_penalty(self):
        from ct_phenotype_increment import fit_ct_head
        from increment_cox import IncrementCox
        rng=np.random.default_rng(123);x=rng.normal(size=(80,6));p=rng.normal(size=80)
        t=rng.exponential(np.exp(-x[:,0]-.3*p))+.01;e=rng.binomial(1,.7,80)
        head=fit_ct_head(x[:60],t[:60],e[:60],10.)
        np.testing.assert_allclose(head.named_steps['scale'].mean_,x[:60].mean(0))
        original=head.named_steps['scale'].mean_.copy();head.predict(x[60:]+1000)
        np.testing.assert_array_equal(head.named_steps['scale'].mean_,original)
        self.assertEqual(head.named_steps['cox'].alpha,10.)
        conditional=IncrementCox(ridge=10.).fit(p[:60],x[:60],t[:60],e[:60])
        np.testing.assert_array_equal(conditional.model.alpha,[0.,10.,10.,10.,10.,10.,10.])

    def test_ct_only_tuning_rejects_overlapping_patients(self):
        from ct_phenotype_increment import select_ct_penalty
        c=dict(train_ids=['a','b'],val_ids=['b','c'])
        with self.assertRaisesRegex(ValueError,'overlap'):select_ct_penalty([c],[1.])

    def test_missing_pa_cache_never_triggers_neural_training(self):
        from ct_phenotype_increment import load_cached_contexts
        oof=pd.DataFrame({'pathology_risk':np.arange(12.),'inner_fold':np.arange(12)%3,
                          'dfs.month':np.arange(1.,13.),'dfs.status':np.ones(12)},index=[f'c{i}' for i in range(12)])
        bundle={'oof':oof,'val':pd.DataFrame(index=['outer'])}
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(FileNotFoundError):load_cached_contexts(0,bundle,Path(folder))


if __name__=='__main__':
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
    unittest.main()
