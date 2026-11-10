"""Synthetic CPU checks only; no actual GL survival training."""
import contextlib
import importlib.util
import io
from pathlib import Path
import sys
import tempfile
import unittest
import warnings
from unittest.mock import patch

import h5py
import numpy as np
import pandas as pd
import torch
import yaml


def load_module(name, filename):
    path = Path(__file__).with_name(filename)
    assert path.is_file(), f'Experiment08 implementation missing: {filename}'
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Experiment08Tests(unittest.TestCase):
    def setUp(self):
        warnings.filterwarnings('ignore', message='Conversion of an array with ndim > 0 to a scalar',
                                category=DeprecationWarning, module=r'lifelines\.plotting')
        torch.set_num_threads(1)
        self.temporary = tempfile.TemporaryDirectory(prefix='pact08_')
        self.root = Path(self.temporary.name)
        self.features = self.root/'features_uni_v1'
        self.features.mkdir()
        self.source = self.root/'labels.csv'
        rng = np.random.default_rng(123)
        rows = []
        for i in range(42):
            case = f'case{i:03d}'
            rows.append(dict(patient_id=case, status=i%2 if i<40 else (2 if i==40 else 1),
                             survival_time=float(100+i*37.3), **{'CT path': np.nan}))
            if i<40:
                with h5py.File(self.features/f'{case}.h5', 'w') as f:
                    f['features'] = rng.normal(size=(3+i%3, 16)).astype('float32')
        pd.DataFrame(rows).to_csv(self.source, index=False)

    def tearDown(self):
        self.temporary.cleanup()

    def prepare(self):
        module = load_module('gl_dataset_test', 'gl_dataset.py')
        with contextlib.redirect_stdout(io.StringIO()):
            data = module.prepare_cohort(self.source, self.features, self.root/'data')
        return module, data

    def test_exact_matching_audit_original_time_and_saved_folds(self):
        module, data = self.prepare()
        self.assertEqual(len(data), 40)
        self.assertEqual(data.event.value_counts().to_dict(), {0:20, 1:20})
        self.assertEqual(set(data.split), set(range(5)))
        source = pd.read_csv(self.source).set_index('patient_id')
        np.testing.assert_allclose(data.time, source.loc[data.pa_id, 'survival_time'])
        audit = pd.read_csv(self.root/'data/excluded_cases.csv')
        self.assertEqual(set(audit.reason), {'status_not_0_or_1', 'missing_feature'})
        before = (self.root/'data/all_label.csv').read_bytes()
        module.prepare_cohort(self.source, self.features, self.root/'data')
        self.assertEqual(before, (self.root/'data/all_label.csv').read_bytes())
        dataset = module.GLDataset(self.root/'data')
        self.assertEqual(dataset.feature_dim, 16)
        features, event, time, case = dataset[0]
        with h5py.File(dataset.samples.pa_path.iloc[0]) as f:
            np.testing.assert_array_equal(features.numpy(), f['features'][:])
        self.assertEqual(float(time), float(np.float32(dataset.samples.time.iloc[0])))

    def test_changed_source_and_changed_feature_shape_are_rejected(self):
        module, data = self.prepare()
        original = self.source.read_bytes()
        changed = pd.read_csv(self.source)
        changed.loc[0, 'status'] = 1-changed.loc[0, 'status']
        changed.to_csv(self.source, index=False)
        with self.assertRaisesRegex(ValueError, 'source'):
            module.GLDataset(self.root/'data')
        self.source.write_bytes(original)
        with h5py.File(data.pa_path.iloc[0], 'w') as f:
            f['features'] = np.ones((4, 8))
        with self.assertRaisesRegex(ValueError, 'shape|dimension'):
            module.GLDataset(self.root/'data')

    def test_per_patch_l2_preserves_source_labels_and_folds(self):
        path = self.features/'case000.h5'
        with h5py.File(path, 'r+') as handle:
            handle['features'][0] = 0
        module, data = self.prepare()
        source_before = path.read_bytes()
        raw = module.GLDataset(self.root/'data')
        normalized = module.GLDataset(self.root/'data', l2_normalize=True)
        pd.testing.assert_frame_equal(raw.samples, normalized.samples)
        for index in [0, 1, 10, 39]:
            features, event, time, case = raw[index]
            unit, new_event, new_time, new_case = normalized[index]
            expected = features/features.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-12)
            torch.testing.assert_close(unit, expected)
            nonzero = features.norm(dim=-1)>0
            torch.testing.assert_close(unit[nonzero].norm(dim=-1), torch.ones(int(nonzero.sum())))
            self.assertTrue(torch.isfinite(unit).all())
            self.assertEqual((int(event), float(time), case), (int(new_event), float(new_time), new_case))
        self.assertEqual(source_before, path.read_bytes())

    def test_event_time_strata_balance_counts_without_model_predictions(self):
        module = load_module('gl_dataset_balance', 'gl_dataset.py')
        samples = pd.DataFrame({'pa_id': [f'p{i}' for i in range(154)],
                                'event': [0]*87+[1]*67,
                                'time': np.r_[np.arange(87)*11+354, np.arange(67)*17+63]})
        divided, cuts = module.add_event_time_splits(samples)
        self.assertEqual(set(cuts), {'0', '1'})
        self.assertEqual(divided.pa_id.tolist(), samples.pa_id.tolist())
        self.assertEqual(divided.time.tolist(), samples.time.tolist())
        self.assertEqual(set(divided.split), set(range(5)))
        sizes = divided.groupby('split').size()
        self.assertLessEqual(sizes.max()-sizes.min(), 1)
        for event, group in divided.groupby('event'):
            sizes = group.groupby('split').size().reindex(range(5), fill_value=0)
            self.assertLessEqual(sizes.max()-sizes.min(), 1)
            for time_group, section in group.groupby('time_group'):
                sizes = section.groupby('split').size().reindex(range(5), fill_value=0)
                self.assertLessEqual(sizes.max()-sizes.min(), 1)

    def test_new_cohort_uses_distinct_run_tag(self):
        module, data = self.prepare()
        metadata = module.validate_cohort(self.root/'data')
        self.assertEqual(metadata['cohort_tag'], 'gl40_eventtime')
        self.assertIn('time', metadata['split_method'])

    def test_all_losses_train_fivefold_export_native_times_and_reload(self):
        self.prepare()
        training = load_module('gl_train_test', 'train.py')
        for loss, l2 in [('cox', False), ('nll', False), ('cox_rank', False), ('cox', True)]:
            output = self.root/f'run_{loss}{"_l2" if l2 else ""}'
            argv = ['train.py', '--data_dir', str(self.root/'data'), '--output_dir', str(output),
                    '--survival_loss', loss, '--num_epochs', '1', '--num_workers', '0', '--cox_batch_size', '8']
            if l2:
                argv.append('--l2_normalize')
            updates = []
            real_adam = torch.optim.Adam

            class CountAdam(real_adam):
                def step(self, closure=None):
                    updates.append(1)
                    return super().step(closure)

            with self.subTest(loss=loss), patch.object(sys, 'argv', argv), \
                 patch.object(torch.cuda, 'is_available', return_value=False), \
                 patch.object(torch.optim, 'Adam', CountAdam), \
                 contextlib.redirect_stdout(io.StringIO()):
                training.main()
            self.assertEqual(len(updates), 160 if loss=='nll' else 20)
            results = next((output/'results').iterdir())
            config = yaml.safe_load((results/'run_config.yaml').read_text())
            self.assertEqual(config['time_unit'], 'original_unconfirmed')
            self.assertEqual(config['feature_dim'], 16)
            self.assertEqual(config['normalization'], 'per_patch_l2' if l2 else 'none')
            self.assertEqual('normalization_eps' in config, l2)
            self.assertEqual('_l2_abmil_' in results.name, l2)
            self.assertEqual('rank_weight' in config, loss=='cox_rank')
            self.assertEqual('cox_batch_size' in config, loss!='nll')
            metrics = pd.read_csv(results/'metrics.csv')
            self.assertEqual(metrics.fold.astype(str).tolist(), ['0','1','2','3','4','mean','std'])
            self.assertFalse(any(c.startswith('auc_') for c in metrics))
            self.assertTrue(np.isfinite(metrics[['train_cindex','val_cindex']]).all().all())
            validation_ids = []
            for fold in range(5):
                directory = results/f'fold_{fold}'
                val = pd.read_csv(directory/'val_predictions.csv')
                self.assertIn('survival_time', val)
                self.assertIn('event', val)
                self.assertFalse(any(c.startswith('dfs.') for c in val))
                source_times = pd.read_csv(self.source).set_index('patient_id').survival_time
                np.testing.assert_array_equal(val.survival_time.to_numpy(np.float32),
                                              source_times.loc[val.case_id].to_numpy(np.float32))
                self.assertTrue((directory/'km_high_vs_low.png').is_file())
                validation_ids.extend(val.case_id)
                state = torch.load(output/'checkpoints'/results.name/f'fold_{fold}'/'best_model.pth', weights_only=True)
                if loss=='nll':
                    samples = pd.read_csv(results/'data.csv')
                    events = samples[(samples.split!=fold)&(samples.event==1)].time.to_numpy()
                    bins = np.unique(np.quantile(events,[.25,.5,.75]).astype('float32'))
                    np.testing.assert_array_equal(state['time_bins'].numpy(), bins)
            self.assertEqual(len(set(validation_ids)), 40)
            protected = [results/'run_config.yaml', results/'data.csv']
            original = [p.read_bytes() for p in protected]
            with patch.object(sys, 'argv', [*argv, '--eval_only']), \
                 patch.object(torch.cuda, 'is_available', return_value=False), \
                 contextlib.redirect_stdout(io.StringIO()):
                training.main()
            self.assertEqual(original, [p.read_bytes() for p in protected])
            checkpoint = output/'checkpoints'/results.name/'fold_4'/'best_model.pth'
            checkpoint.unlink()
            with patch.object(sys, 'argv', [*argv, '--eval_only']), \
                 patch.object(torch.cuda, 'is_available', return_value=False), \
                 contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(FileNotFoundError):
                    training.main()
            self.assertEqual(original, [p.read_bytes() for p in protected])


if __name__ == '__main__':
    unittest.main()
