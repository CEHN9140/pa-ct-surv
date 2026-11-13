import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))


class PredictedShapeTests(unittest.TestCase):
    def data(self):
        rng=np.random.default_rng(123)
        ids=[f'c{i}' for i in range(60)]
        x=pd.DataFrame(rng.normal(size=(60,8)),index=ids)
        y=pd.DataFrame(np.column_stack([x[0]*2,x[1]*3,x[2]+x[3]]),index=ids,columns=['volume','diameter','sphere'])
        train=ids[:48];held=ids[48:]
        splits=[dict(train_ids=[v for j,v in enumerate(train) if j%3!=i],
                     val_ids=[v for j,v in enumerate(train) if j%3==i]) for i in range(3)]
        return x,y,train,held,splits

    def test_held_target_changes_cannot_change_probe_tuning_or_predictions(self):
        from shape_crossfit import crossfit_shape
        x,y,train,held,splits=self.data()
        original=crossfit_shape(x,y,train,held,splits,[.1,10.],123)
        changed=y.copy();changed.loc[held]=1e6
        repeat=crossfit_shape(x,changed,train,held,splits,[.1,10.],123)
        np.testing.assert_allclose(original['train_pred'],repeat['train_pred'],rtol=0,atol=1e-12)
        np.testing.assert_allclose(original['val_pred'],repeat['val_pred'],rtol=0,atol=1e-12)
        self.assertEqual(original['selection']['alphas'],repeat['selection']['alphas'])
        missing=crossfit_shape(x,y.loc[train],train,held,splits,[.1,10.],123)
        np.testing.assert_allclose(original['val_pred'],missing['val_pred'],rtol=0,atol=1e-12)

    def test_each_cox_training_patient_is_predicted_once_without_self_target(self):
        from shape_crossfit import crossfit_shape
        x,y,train,held,splits=self.data()
        result=crossfit_shape(x,y,train,held,splits,[.1,10.],123)
        self.assertEqual(list(result['train_pred'].index),train)
        np.testing.assert_allclose(result['head'].predict(x.loc[held]),result['val_pred'])
        counts={v:0 for v in train}
        for step in result['roles']['oof']:
            self.assertFalse(set(step['train_ids'])&set(step['val_ids']))
            self.assertFalse(set(step['train_ids'])&set(held))
            for partition in step['tuning_roles']:
                self.assertFalse(set(partition['train_ids'])&set(partition['val_ids']))
                self.assertEqual(set(partition['train_ids'])|set(partition['val_ids']),set(step['train_ids']))
            for v in step['val_ids']:counts[v]+=1
        self.assertEqual(set(counts.values()),{1})
        # A patient's own target cannot affect the OOF prediction of that patient.
        changed=y.copy();changed.loc[train[0]]+=1e5
        repeat=crossfit_shape(x,changed,train,held,splits,[.1,10.],123)
        np.testing.assert_allclose(result['train_pred'].loc[train[0]],repeat['train_pred'].loc[train[0]],rtol=0,atol=1e-12)

    def test_invalid_splits_and_overlap_fail_before_fitting(self):
        from shape_crossfit import crossfit_shape
        x,y,train,held,splits=self.data()
        with self.assertRaisesRegex(ValueError,'overlap'):
            crossfit_shape(x,y,train,[train[0]],splits,[1.],123)
        with self.assertRaisesRegex(ValueError,'partition'):
            crossfit_shape(x,y,train,held,splits[:2],[1.],123)
        with self.assertRaisesRegex(ValueError,'partition'):
            crossfit_shape(x,y,train,held,[dict(train_ids=train,val_ids=train)],[1.],123)


if __name__=='__main__':unittest.main()
