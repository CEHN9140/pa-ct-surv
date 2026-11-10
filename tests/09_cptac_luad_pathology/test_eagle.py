"""Synthetic CPU checks for CPTAC EAGLE top25 MeanPool; no real training."""
import contextlib
import io
from pathlib import Path
import subprocess
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


class EagleTests(unittest.TestCase):
    def setUp(self):
        fixture = existing.Experiment09Tests()
        fixture.setUp()
        self.addCleanup(fixture.tearDown)
        self.root = fixture.root
        self.data_module, self.data = fixture.prepare()
        self.eagle = self.root/'eagle_source'
        features = self.eagle/'eagle/features'; features.mkdir(parents=True)
        inputs = self.eagle/'inputs'; inputs.mkdir()
        source = fixture.source/'survival/PathoBench_CPTAC_LUAD_OS.tsv'
        pd.read_csv(source, sep='\t').to_csv(inputs/'slides.csv', index=False)
        rng = np.random.default_rng(9)
        selections = []
        for row in self.data.itertuples():
            path = features/f'{row.slide_id}.h5'
            with h5py.File(path, 'w') as handle:
                handle['features'] = rng.normal(size=(25,1280)).astype(np.float32)
                handle.attrs['downstream_encoder'] = 'Virchow2-CLS'
                handle.attrs['top_k'] = 25
            selections.append(dict(pa_id=row.slide_id, selected=25, feature_path=str(path)))
        pd.DataFrame(selections).to_csv(self.eagle/'eagle/selection_manifest.csv', index=False)

    def test_cli_has_eagle_and_meanpool(self):
        result = subprocess.run([sys.executable, str(Path(__file__).with_name('train.py')), '--help'],
                                capture_output=True, text=True, check=True)
        self.assertIn('eagle_top25', result.stdout)
        self.assertIn('--pa_model {abmil,meanpool,meanpool_linear}', result.stdout)

    def test_linear_head_is_exactly_mean_then_linear(self):
        module=existing.module('meanpool_linear_test','meanpool_linear.py')
        model=module.MeanPoolLinear(1280)
        self.assertEqual(sum(p.numel()for p in model.parameters()),1281)
        self.assertEqual(len(list(model.children())),1)
        features=torch.randn(4,25,1280,requires_grad=True)
        risk,pooled,attention=model(features)
        torch.testing.assert_close(pooled,features.mean(1))
        torch.testing.assert_close(risk,model.risk_head(features.mean(1)).squeeze(-1))
        self.assertIsNone(attention)
        torch.testing.assert_close(model(features.flip(1))[0],risk)
        torch.testing.assert_close(model.eval()(features)[0],model.train()(features)[0])
        risk.sum().backward()
        torch.testing.assert_close(features.grad,features.grad[:,:1].expand_as(features))
        self.assertGreater(features.grad.abs().sum().item(),0)

    def test_linear_fivefold_training_reload_and_separate_outputs(self):
        train=existing.module('cptac_linear_training_test','train.py')
        argv=['train.py','--data_dir',str(self.root/'data'),'--eagle_root',str(self.eagle),
              '--output_dir',str(self.root/'linear_run'),'--feature_source','eagle_top25',
              '--pa_model','meanpool_linear','--cox_batch_size','12','--num_epochs','1','--num_workers','0']
        with patch.object(sys,'argv',argv),patch.object(torch.cuda,'is_available',return_value=False), \
             contextlib.redirect_stdout(io.StringIO()):train.main()
        result=next((self.root/'linear_run/results').iterdir())
        self.assertIn('single_eagle_top25_meanpool_linear_cox_epochs',result.name)
        self.assertNotIn('dropout',result.name)
        config=yaml.safe_load((result/'run_config.yaml').read_text())
        self.assertNotIn('dropout',config)
        self.assertEqual(config['trainable_parameters'],1281)
        self.assertEqual(config['patient_batch_size'],12)
        self.assertEqual(config['risk_head'],'linear')
        metrics=pd.read_csv(result/'metrics.csv');predictions=[]
        for fold in range(5):
            val=pd.read_csv(result/f'fold_{fold}/val_predictions.csv')
            self.assertEqual(set(val.case_id),set(self.data.loc[self.data.split.eq(fold),'case_id']))
            self.assertAlmostEqual(concordance_index_censored(val.OS_event.astype(bool),val.OS_days,val.risk_score)[0],
                                   metrics.loc[fold,'val_cindex'])
            state=torch.load(self.root/'linear_run/checkpoints'/result.name/f'fold_{fold}/best_model.pth',
                             map_location='cpu',weights_only=True)
            self.assertEqual(set(state),{'risk_head.weight','risk_head.bias'})
            self.assertEqual(tuple(state['risk_head.weight'].shape),(1,1280))
            predictions.append(val.sort_values('case_id').reset_index(drop=True))
        before=[(result/name).read_bytes()for name in ['run_config.yaml','data.csv']]
        with patch.object(sys,'argv',[*argv,'--eval_only']),patch.object(torch.cuda,'is_available',return_value=False), \
             contextlib.redirect_stdout(io.StringIO()):train.main()
        self.assertEqual(before,[(result/name).read_bytes()for name in ['run_config.yaml','data.csv']])
        for fold,expected in enumerate(predictions):
            actual=pd.read_csv(result/f'fold_{fold}/val_predictions.csv').sort_values('case_id').reset_index(drop=True)
            pd.testing.assert_frame_equal(actual,expected)

    def test_frozen_slides_labels_and_top25_features(self):
        dataset = self.data_module.CPTACEagleDataset(self.root/'data', self.eagle)
        self.assertEqual(dataset.samples.slide_id.tolist(), self.data.slide_id.tolist())
        self.assertEqual(dataset.samples.split.tolist(), self.data.split.tolist())
        self.assertTrue(dataset.samples.n_patches.eq(25).all())
        self.assertEqual(dataset.feature_dim,1280)
        for index in [0,7,21]:
            features,event,time,case = dataset[index]
            row = self.data.iloc[index]
            with h5py.File(self.eagle/'eagle/features'/f'{row.slide_id}.h5') as handle:
                torch.testing.assert_close(features,torch.from_numpy(handle['features'][:]),rtol=0,atol=0)
            self.assertEqual(features.shape,(25,1280))
            self.assertEqual(case,row.case_id)
            self.assertEqual(int(event),row.event)
            self.assertEqual(float(time),row.time)
        source = self.eagle/'inputs/slides.csv'
        slides = pd.read_csv(source)
        selected = slides.index[slides.slide_id.eq(self.data.iloc[0].slide_id)][0]
        slides.loc[selected,'OS_event'] = 1-int(self.data.iloc[0].event)
        slides.to_csv(source,index=False)
        with self.assertRaisesRegex(ValueError,'labels'):
            self.data_module.CPTACEagleDataset(self.root/'data',self.eagle)

    def test_invalid_top25_features_rejected(self):
        path = self.eagle/'eagle/features'/f'{self.data.iloc[0].slide_id}.h5'
        with h5py.File(path,'r+') as handle:handle['features'][0,0]=np.nan
        with self.assertRaisesRegex(ValueError,'finite'):
            self.data_module.CPTACEagleDataset(self.root/'data',self.eagle)

    def test_batched_fivefold_train_reload_and_os_outputs(self):
        train = existing.module('cptac_eagle_training_test','train.py')
        argv=['train.py','--data_dir',str(self.root/'data'),'--eagle_root',str(self.eagle),
              '--output_dir',str(self.root/'runs'),'--feature_source','eagle_top25',
              '--pa_model','meanpool','--cox_batch_size','12','--num_epochs','1','--num_workers','0']
        output=io.StringIO()
        with patch.object(sys,'argv',argv),patch.object(torch.cuda,'is_available',return_value=False), \
             contextlib.redirect_stdout(output):train.main()
        self.assertIn('[12, 20]',output.getvalue())
        result=next((self.root/'runs/results').iterdir())
        self.assertIn('single_eagle_top25_meanpool_cox',result.name)
        config=yaml.safe_load((result/'run_config.yaml').read_text())
        self.assertEqual(config['feature_dim'],1280)
        self.assertEqual(config['batch_mode'],'patient_batch')
        self.assertEqual(config['patient_batch_size'],12)
        self.assertEqual(config['endpoint'],'OS')
        self.assertEqual(config['time_unit'],'days')
        self.assertTrue(pd.read_csv(result/'data.csv').n_patches.eq(25).all())
        metric=pd.read_csv(result/'metrics.csv');seen=[]
        for fold in range(5):
            val=pd.read_csv(result/f'fold_{fold}/val_predictions.csv')
            self.assertEqual(set(val.case_id),set(self.data.loc[self.data.split.eq(fold),'case_id']))
            expected=self.data.set_index('case_id').loc[val.case_id]
            np.testing.assert_array_equal(val.OS_days,expected.time)
            np.testing.assert_array_equal(val.OS_event,expected.event)
            self.assertAlmostEqual(concordance_index_censored(val.OS_event.astype(bool),val.OS_days,val.risk_score)[0],
                                   metric.loc[fold,'val_cindex'])
            seen.extend(val.case_id)
        self.assertEqual(len(seen),len(set(seen)))
        before=[(result/name).read_bytes()for name in ['run_config.yaml','data.csv']]
        with patch.object(sys,'argv',[*argv,'--eval_only']),patch.object(torch.cuda,'is_available',return_value=False), \
             contextlib.redirect_stdout(io.StringIO()):train.main()
        self.assertEqual(before,[(result/name).read_bytes()for name in ['run_config.yaml','data.csv']])
        feature=self.eagle/'eagle/features'/f'{self.data.iloc[0].slide_id}.h5'
        with h5py.File(feature,'r+') as handle:handle['features'][0,0] += 1
        with patch.object(sys,'argv',[*argv,'--eval_only']):
            with self.assertRaisesRegex(ValueError,'provenance'):train.main()


if __name__=='__main__':unittest.main()
