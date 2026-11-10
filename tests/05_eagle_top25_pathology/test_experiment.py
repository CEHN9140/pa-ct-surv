"""CPU-only checks for the EAGLE MeanPool control and legacy ABMIL entry."""

import contextlib
import hashlib
import importlib.util
import io
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np
import pandas as pd
import torch
import yaml
from sksurv.metrics import concordance_index_censored

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from model.build import Pa_Model


class EagleExperimentTests(unittest.TestCase):
    def test_cli_declares_both_models(self):
        result = subprocess.run(
            [sys.executable, str(Path(__file__).with_name('train.py')), '--help'],
            capture_output=True, text=True, check=True,
        )
        self.assertIn('--pa_model {abmil,meanpool}', result.stdout)

    def test_meanpool_uses_raw_mean_and_equal_patch_gradients(self):
        torch.manual_seed(123)
        model = Pa_Model(model_name='meanpool', feature_dim=1280, abmil_dropout=.25).eval()
        features = torch.randn(3, 25, 1280, requires_grad=True)
        risk, pooled, attention = model(features)
        expected = model.mil.risk_head(model.mil.post_compression(features.mean(1))).squeeze(-1)
        torch.testing.assert_close(risk, expected)
        torch.testing.assert_close(model(features.flip(1))[0], risk)
        self.assertEqual(tuple(pooled.shape), (3, 256))
        self.assertIsNone(attention)
        risk.sum().backward()
        self.assertGreater(features.grad.abs().sum().item(), 0)
        torch.testing.assert_close(features.grad, features.grad[:, :1].expand_as(features))

    def test_five_fold_training_and_output_separation(self):
        torch.set_num_threads(1)
        script = Path(__file__).with_name('train.py')
        spec = importlib.util.spec_from_file_location('eagle_train_check', script)
        train = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(train)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rng = np.random.default_rng(7)
            records = []
            for index in range(20):
                feature_path = root / f'patient{index}.h5'
                with h5py.File(feature_path, 'w') as handle:
                    handle['features'] = rng.normal(size=(25, 1280)).astype(np.float32)
                records.append(dict(pa_id=str(index), pa_path=str(feature_path),
                                    event=int(index % 4 < 2),
                                    time=[12., 24., 48., 84.][index % 4], split=index // 4))
            cohort = pd.DataFrame(records)
            label_path = root / 'all_label_roi64.csv'
            cohort.to_csv(label_path, index=False)
            original_hash = hashlib.sha256(label_path.read_bytes()).hexdigest()
            for model_name in ['abmil', 'meanpool']:
                command = ['train.py', '--data_dir', str(root), '--batch_size', '6',
                           '--num_epochs', '1', '--num_workers', '0']
                if model_name == 'meanpool':
                    command += ['--pa_model', model_name]
                with patch.object(train, '__file__', str(root / 'experiment' / 'train.py')), \
                     patch.object(sys, 'argv', command), \
                     patch('torch.cuda.is_available', return_value=False), \
                     contextlib.redirect_stdout(io.StringIO()), \
                     contextlib.redirect_stderr(io.StringIO()):
                    train.main()
                outputs = list((root / 'experiment/results').glob(f'*_{model_name}_cox_*'))
                self.assertEqual(len(outputs), 1)
                destination = outputs[0]
                config = yaml.safe_load((destination / 'run_config.yaml').read_text())
                self.assertEqual(config['pa_model'], model_name)
                self.assertEqual(config['batch_size'], 6)
                self.assertEqual(config['feature_dim'], 1280)
                self.assertNotIn('cox_batch_size', config)
                self.assertEqual(len(pd.read_csv(destination / 'metrics.csv')), 7)
                seen = []
                for fold in range(5):
                    predictions = pd.read_csv(destination / f'fold_{fold}/val_predictions.csv',
                                              dtype={'case_id': str})
                    self.assertEqual(set(predictions.case_id), set(cohort.loc[cohort.split == fold, 'pa_id']))
                    seen.extend(predictions.case_id)
                    checkpoint = root / 'experiment/checkpoints' / destination.name / f'fold_{fold}'
                    self.assertTrue((checkpoint / 'best_model.pth').is_file())
                    self.assertTrue((checkpoint / 'last_model.pth').is_file())
                    self.assertTrue((destination / f'fold_{fold}/km_high_vs_low.png').is_file())
                    metrics = pd.read_csv(destination / 'metrics.csv')
                    ci = concordance_index_censored(predictions['dfs.status'].astype(bool),
                                                   predictions['dfs.month'], predictions.risk_score)[0]
                    self.assertAlmostEqual(ci, metrics.loc[fold, 'val_cindex'])
                self.assertEqual(len(seen), len(set(seen)))
            self.assertEqual(hashlib.sha256(label_path.read_bytes()).hexdigest(), original_hash)
            self.assertEqual(len(list((root / 'experiment/results').iterdir())), 2)


if __name__ == '__main__':
    unittest.main()
