import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))


class ShapeAblationTests(unittest.TestCase):
    def test_exact_nonempty_groups_and_conditional_contrasts(self):
        from shape_ablation import shape_groups,conditional_contrasts
        groups=shape_groups()
        self.assertEqual(list(groups),['V','D','S','VD','VS','DS','VDS'])
        self.assertEqual(groups['VDS'],['log_volume_mm3','diameter_mm','sphericity'])
        self.assertEqual(len({tuple(v) for v in groups.values()}),7)
        self.assertEqual(conditional_contrasts(),[('VDS','DS','V'),('VDS','VS','D'),('VDS','VD','S'),('VD','V','D'),('VD','D','V')])

    def write_cache(self,folder):
        columns=['log_volume_mm3','diameter_mm','sphericity']
        a=pd.DataFrame(np.arange(12.).reshape(4,3),index=['a','b','c','d'],columns=columns)
        b=pd.DataFrame([[1.,2.,3.]],index=['held'],columns=columns)
        a.rename_axis('case_id').to_csv(folder/'crossfit_shapes.csv')
        b.rename_axis('case_id').to_csv(folder/'predicted_shapes.csv')
        steps=[]
        for train,val in [(['a','b'],['c','d']),(['c','d'],['a','b'])]:
            steps.append(dict(train_ids=train,val_ids=val,tuning_roles=[dict(train_ids=train[:1],val_ids=train[1:])]))
        roles=dict(oof=steps,final=dict(train_ids=list(a.index),val_ids=list(b.index),
                    tuning_roles=[dict(train_ids=['a','b'],val_ids=['c','d'])]))
        (folder/'crossfit_roles.json').write_text(json.dumps(roles))
        return a,b,roles

    def test_saved_predictions_align_by_case_and_preserve_slices(self):
        from shape_ablation import read_shape_context
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory);a,b,_=self.write_cache(folder)
            train,held=read_shape_context(folder,['d','c','b','a'],['held'],['outer'],{})
            np.testing.assert_array_equal(train,a.loc[['d','c','b','a']])
            np.testing.assert_array_equal(held,b)
            np.testing.assert_array_equal(train[['sphericity','log_volume_mm3']],a.loc[train.index,['sphericity','log_volume_mm3']])

    def test_cache_identity_nonfinite_and_role_leakage_fail(self):
        from shape_ablation import read_shape_context
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory);a,b,roles=self.write_cache(folder)
            with self.assertRaisesRegex(ValueError,'identity'):
                read_shape_context(folder,['a','b','c','wrong'],['held'],['outer'],{})
            with self.assertRaisesRegex(ValueError,'overlap'):
                read_shape_context(folder,['a','b','c','d'],['a'],['outer'],{})
            a.iloc[0,0]=np.nan;a.rename_axis('case_id').to_csv(folder/'crossfit_shapes.csv')
            with self.assertRaisesRegex(ValueError,'finite'):
                read_shape_context(folder,['a','b','c','d'],['held'],['outer'],{})
            self.write_cache(folder);roles['oof'][0]['train_ids'].append('held')
            (folder/'crossfit_roles.json').write_text(json.dumps(roles))
            with self.assertRaisesRegex(ValueError,'partition'):
                read_shape_context(folder,['a','b','c','d'],['held'],['outer'],{})

    def test_reference_parity_allows_solver_roundoff_but_not_rank_changes(self):
        from shape_ablation import verify_reference_risk
        reference=np.array([-2.,-.5,1.,3.])
        self.assertLess(verify_reference_risk(reference+np.array([1e-9,-2e-9,3e-9,0]),reference),1e-8)
        with self.assertRaises(AssertionError):verify_reference_risk(reference+1e-5,reference)
        with self.assertRaises(AssertionError):verify_reference_risk(np.array([0.,.99e-8]),np.array([0.,1.01e-8]))


if __name__=='__main__':unittest.main()
