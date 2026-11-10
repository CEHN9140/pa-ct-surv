"""KEEP preparation and five-fold CPU training against a frozen GL reference."""
import contextlib
import io
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import h5py
import numpy as np
import pandas as pd
import torch
import yaml
from sksurv.metrics import concordance_index_censored

import test_experiment as existing


class KeepTests(unittest.TestCase):
    def setUp(self):
        fixture=existing.Experiment08Tests();fixture.setUp();self.addCleanup(fixture.tearDown)
        self.root=fixture.root;self.keep=self.root/'features_keep';self.keep.mkdir()
        rng=np.random.default_rng(9)
        for path in fixture.features.glob('*.h5'):
            with h5py.File(path,'r+')as f:
                coords=np.arange(f['features'].shape[0]*2).reshape(-1,2);f['coords']=coords
            with h5py.File(self.keep/path.name,'w')as f:
                f['features']=rng.normal(size=(len(coords),768)).astype(np.float32);f['coords']=coords
        self.data_module,self.reference=fixture.prepare()

    def test_keep_uses_exact_same_patients_labels_folds_and_patch_order(self):
        prep=existing.load_module('gl_keep_prepare_test','prepare_keep.py')
        with contextlib.redirect_stdout(io.StringIO()):
            samples=prep.prepare_keep(self.root/'data',self.keep,self.root/'data_keep')
        pd.testing.assert_frame_equal(samples[['pa_id','event','time','split']],self.reference[['pa_id','event','time','split']])
        dataset=self.data_module.GLDataset(self.root/'data_keep')
        self.assertEqual(dataset.feature_dim,768)
        self.assertEqual(dataset.metadata['feature_encoder'],'keep')
        for index in [0,7,21]:
            features,event,time,case=dataset[index]
            self.assertEqual(features.shape[1],768)
            with h5py.File(self.keep/f'{case}.h5')as f:
                torch.testing.assert_close(features,torch.from_numpy(f['features'][:]),rtol=0,atol=0)
        before={p.name:p.read_bytes()for p in (self.root/'data_keep').iterdir()}
        with contextlib.redirect_stdout(io.StringIO()):prep.prepare_keep(self.root/'data',self.keep,self.root/'data_keep')
        self.assertEqual(before,{p.name:p.read_bytes()for p in (self.root/'data_keep').iterdir()})
        ref=self.root/'data/all_label.csv';changed=pd.read_csv(ref);changed.loc[0,'split']=(changed.loc[0,'split']+1)%5
        changed.to_csv(ref,index=False)
        with self.assertRaisesRegex(ValueError,'reference'):self.data_module.GLDataset(self.root/'data_keep')

    def test_missing_keep_or_mismatched_coordinates_cannot_drop_patients(self):
        prep=existing.load_module('gl_keep_invalid_test','prepare_keep.py')
        path=self.keep/f'{self.reference.iloc[0].pa_id}.h5';before=path.read_bytes();path.unlink()
        with self.assertRaises(FileNotFoundError):prep.prepare_keep(self.root/'data',self.keep,self.root/'missing')
        self.assertFalse((self.root/'missing/all_label.csv').exists())
        path.write_bytes(before)
        with h5py.File(path,'r+')as f:f['coords'][:]=f['coords'][:][::-1]
        with self.assertRaisesRegex(ValueError,'coordinate'):prep.prepare_keep(self.root/'data',self.keep,self.root/'wrong_coords')

    def test_keep_fivefold_native_time_metrics_and_reload(self):
        prep=existing.load_module('gl_keep_prepare_training','prepare_keep.py')
        with contextlib.redirect_stdout(io.StringIO()):prep.prepare_keep(self.root/'data',self.keep,self.root/'data_keep')
        train=existing.load_module('gl_keep_training','train.py')
        argv=['train.py','--data_dir',str(self.root/'data_keep'),'--output_dir',str(self.root/'runs'),
              '--survival_loss','cox','--num_epochs','1','--cox_batch_size','8','--num_workers','0']
        with patch.object(sys,'argv',argv),patch.object(torch.cuda,'is_available',return_value=False),contextlib.redirect_stdout(io.StringIO()):
            train.main()
        result=next((self.root/'runs/results').iterdir());self.assertIn('_keep_abmil_cox_',result.name)
        config=yaml.safe_load((result/'run_config.yaml').read_text())
        self.assertEqual(config['feature_dim'],768);self.assertEqual(config['feature_encoder'],'keep')
        self.assertEqual(config['normalization'],'none');self.assertEqual(config['time_unit'],'original_unconfirmed')
        metrics=pd.read_csv(result/'metrics.csv');self.assertFalse(any(c.startswith('auc_')for c in metrics))
        seen=[]
        for fold in range(5):
            val=pd.read_csv(result/f'fold_{fold}/val_predictions.csv')
            expected=self.reference.loc[self.reference.split.eq(fold)].set_index('pa_id')
            self.assertEqual(set(val.case_id),set(expected.index))
            np.testing.assert_allclose(val.survival_time,expected.loc[val.case_id].time,rtol=1e-6)
            ci=concordance_index_censored(val.event.astype(bool),val.survival_time,val.risk_score)[0]
            self.assertAlmostEqual(ci,metrics.loc[fold,'val_cindex']);seen.extend(val.case_id)
            self.assertTrue((result/f'fold_{fold}/km_high_vs_low.png').exists())
        self.assertEqual(len(seen),len(set(seen)))
        before=[(result/f).read_bytes()for f in ['run_config.yaml','data.csv']]
        with patch.object(sys,'argv',[*argv,'--eval_only']),patch.object(torch.cuda,'is_available',return_value=False),contextlib.redirect_stdout(io.StringIO()):
            train.main()
        self.assertEqual(before,[(result/f).read_bytes()for f in ['run_config.yaml','data.csv']])


if __name__=='__main__':unittest.main()
