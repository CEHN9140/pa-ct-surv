"""CPU-only checks for experiment06; all generated files stay in /tmp."""

import contextlib
import importlib.util
import io
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import h5py
import numpy as np
import pandas as pd
import torch
import yaml

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('experiment06_train', HERE / 'train.py')
train = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train)


class Experiment06Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.temporary = tempfile.TemporaryDirectory(prefix='pact_exp06_')
        cls.root = Path(cls.temporary.name)
        cls.reference, cls.eagle, cls.keep = [cls.root / name for name in ['reference', 'eagle', 'keep']]
        for directory in [cls.reference, cls.eagle, cls.keep]:
            directory.mkdir()
        rng = np.random.default_rng(123)
        rows, eagle_rows = [], []
        times = [6., 12., 24., 36., 48., 60., 72., 84., 96.]
        events = [1, 0, 1, 0, 1, 0, 1, 0, 0]
        for i in range(45):
            case = f'slide{i:03d}'
            count = 3 + i % 3
            coords = np.arange(count * 2).reshape(count, 2)
            uni = rng.normal(size=(count, 1024)).astype('float32')
            keep = uni[:, :768].copy()
            for path, features, xy in [
                (cls.reference / f'{case}.h5', uni, coords),
                (cls.keep / f'{case}.h5', keep, coords),
                (cls.eagle / f'{case}.h5', rng.normal(size=(25, 1280)).astype('float32'), np.zeros((25, 2)))
            ]:
                with h5py.File(path, 'w') as file:
                    file['features'], file['coords'] = features, xy
            row = dict(pa_id=case, ct_id=str(10000+i), pa_path=str(cls.reference / f'{case}.h5'),
                       coord_path=str(cls.reference / f'{case}.h5'), event=events[i//5], time=times[i//5], split=i%5)
            rows.append(row)
            eagle_rows.append({**row, 'pa_path': str(cls.eagle / f'{case}.h5'),
                               'coord_path': str(cls.eagle / f'{case}.h5')})
        cls.labels = pd.DataFrame(rows)
        cls.labels.to_csv(cls.reference / 'all_label_roi64.csv', index=False)
        # Deliberately reorder EAGLE rows; the experiment must restore reference order.
        pd.DataFrame(eagle_rows).iloc[::-1].to_csv(cls.eagle / 'all_label_roi64.csv', index=False)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def arguments(self, source, loss):
        return ['train.py', '--feature_source', source, '--survival_loss', loss,
                '--data_dir', str(self.eagle if source == 'eagle' else self.reference),
                '--reference_data_dir', str(self.reference), '--keep_feature_dir', str(self.keep),
                '--num_epochs', '1', '--num_workers', '0', '--batch_size', '8', '--cox_batch_size', '8']

    def test_sources_dimensions_normalization_and_patient_alignment(self):
        for source, dim in [('uni_l2', 1024), ('keep', 768), ('eagle', 1280)]:
            with self.subTest(source=source), patch.object(sys, 'argv', self.arguments(source, 'nll')):
                args = train.parse_args()
                dataset = train.make_dataset(args)
                self.assertEqual(dataset.samples.pa_id.tolist(), self.labels.pa_id.tolist())
                self.assertEqual(args.feature_dim, dim)
                feature = dataset[0][0]
                self.assertEqual(feature.shape[1], dim)
                if source == 'uni_l2':
                    torch.testing.assert_close(feature.norm(dim=-1), torch.ones(len(feature)))
                else:
                    if source == 'eagle':
                        self.assertEqual(feature.shape, (25, 1280))
                    with h5py.File(dataset.samples.pa_path.iloc[0], 'r') as file:
                        np.testing.assert_array_equal(feature.numpy(), file['features'][:])

    def test_eval_only_missing_checkpoint_preserves_training_provenance(self):
        output = self.root / 'provenance'
        with patch.object(train, '__file__', str(output / 'train.py')), \
             patch.object(sys, 'argv', self.arguments('uni_l2', 'nll')), \
             patch.object(train, 'cv_splits', return_value=[]), patch.object(train, 'save_cv_results'), \
             patch.object(torch.cuda, 'is_available', return_value=False), contextlib.redirect_stdout(io.StringIO()):
            train.main()
        results = next((output / 'results').iterdir())
        before = {name: (results / name).read_bytes() for name in ['run_config.yaml', 'data.csv']}
        with patch.object(train, '__file__', str(output / 'train.py')), \
             patch.object(sys, 'argv', [*self.arguments('uni_l2', 'nll'), '--eval_only']), \
             patch.object(torch.cuda, 'is_available', return_value=False), contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(FileNotFoundError):
                train.main()
        for name, content in before.items():
            self.assertEqual((results / name).read_bytes(), content)

    def test_mismatched_patient_outcomes_are_rejected(self):
        path = self.eagle / 'all_label_roi64.csv'
        original = path.read_bytes()
        changed = pd.read_csv(path)
        changed.loc[0, 'event'] = 1 - changed.loc[0, 'event']
        changed.to_csv(path, index=False)
        try:
            with patch.object(sys, 'argv', self.arguments('eagle', 'nll')):
                with self.assertRaises(ValueError):
                    train.make_dataset(train.parse_args())
        finally:
            path.write_bytes(original)

    def test_all_six_fivefold_runs_export_consistent_outputs(self):
        real_adam = torch.optim.Adam
        for source in ['uni_l2', 'keep', 'eagle']:
            for loss in ['nll', 'cox_rank']:
                with self.subTest(source=source, loss=loss):
                    updates = []

                    class CountAdam(real_adam):
                        def step(self, closure=None):
                            updates.append(1)
                            return super().step(closure)

                    output = self.root / f'run_{source}_{loss}'
                    capture = io.StringIO()
                    with patch.object(train, '__file__', str(output / 'train.py')), \
                         patch.object(sys, 'argv', self.arguments(source, loss)), \
                         patch.object(torch.cuda, 'is_available', return_value=False), \
                         patch.object(torch.optim, 'Adam', CountAdam), contextlib.redirect_stdout(capture):
                        train.main()
                    expected_steps = 180 if loss == 'nll' and source != 'eagle' else 20
                    self.assertEqual(len(updates), expected_steps)
                    results = next((output / 'results').iterdir())
                    config = yaml.safe_load((results / 'run_config.yaml').read_text())
                    self.assertEqual(config['survival_loss'], loss)
                    self.assertEqual(config['feature_dim'], {'uni_l2':1024, 'keep':768, 'eagle':1280}[source])
                    self.assertEqual('rank_weight' in config, loss == 'cox_rank')
                    self.assertEqual('batch_size' in config, source == 'eagle')
                    self.assertEqual('cox_batch_size' in config, source != 'eagle' and loss == 'cox_rank')
                    self.assertEqual('keep_feature_dir' in config, source == 'keep')
                    snapshot = pd.read_csv(results / 'data.csv', dtype={'pa_id':str, 'ct_id':str})
                    pd.testing.assert_frame_equal(snapshot[['pa_id','ct_id','event','time','split']],
                                                  self.labels[['pa_id','ct_id','event','time','split']])
                    summary = pd.read_csv(results / 'metrics.csv')
                    self.assertEqual(summary.fold.astype(str).tolist(), ['0','1','2','3','4','mean','std'])
                    self.assertTrue(np.isfinite(summary[['train_loss','val_loss','val_cindex','auc_36m','auc_60m']]).all().all())
                    epoch_losses = re.findall(r'Epoch 1/1.*?Val Loss: ([\d.]+)', capture.getvalue())
                    self.assertEqual(len(epoch_losses), 5)
                    np.testing.assert_allclose(np.array(epoch_losses, dtype=float), summary.val_loss.iloc[:5], atol=5.1e-5, rtol=0)
                    val_ids = []
                    for fold in range(5):
                        directory = results / f'fold_{fold}'
                        val = pd.read_csv(directory / 'val_predictions.csv', dtype={'case_id':str})
                        training = pd.read_csv(directory / 'train_predictions.csv', dtype={'case_id':str})
                        self.assertEqual(set(val.case_id), set(self.labels.loc[self.labels.split == fold, 'pa_id']))
                        self.assertTrue(set(val.case_id).isdisjoint(training.case_id))
                        val_ids.extend(val.case_id)
                        self.assertTrue((directory / 'km_high_vs_low.png').is_file())
                        checkpoint = output / 'checkpoints' / results.name / f'fold_{fold}'
                        self.assertTrue((checkpoint / 'last_model.pth').is_file())
                        state = torch.load(checkpoint / 'best_model.pth', weights_only=True)
                        if loss == 'nll':
                            subset = self.labels[(self.labels.split != fold) & (self.labels.event == 1)]
                            bins = np.unique(np.quantile(subset.time, [.25,.5,.75]).astype('float32'))
                            np.testing.assert_array_equal(state['time_bins'].numpy(), bins)
                        else:
                            self.assertNotIn('time_bins', state)
                    self.assertEqual(len(set(val_ids)), 45)
                    before = pd.read_csv(results / 'fold_0/val_predictions.csv').sort_values('case_id')
                    provenance = {name: (results / name).read_bytes() for name in ['run_config.yaml','data.csv']}
                    with patch.object(train, '__file__', str(output / 'train.py')), \
                         patch.object(sys, 'argv', [*self.arguments(source, loss), '--eval_only']), \
                         patch.object(torch.cuda, 'is_available', return_value=False), \
                         patch.object(torch.optim, 'Adam', side_effect=AssertionError('eval_only must not train')), \
                         contextlib.redirect_stdout(io.StringIO()):
                        train.main()
                    after = pd.read_csv(results / 'fold_0/val_predictions.csv').sort_values('case_id')
                    np.testing.assert_allclose(before.risk_score, after.risk_score, rtol=1e-6, atol=1e-6)
                    for name, content in provenance.items():
                        self.assertEqual((results / name).read_bytes(), content)

    def test_matrix_runner_has_six_jobs_and_stops_on_failure(self):
        runner_spec = importlib.util.spec_from_file_location('experiment06_runner', HERE / 'run_all.py')
        runner = importlib.util.module_from_spec(runner_spec)
        runner_spec.loader.exec_module(runner)
        for fail in [False, True]:
            commands = []

            def execute(command, **kwargs):
                commands.append(command)
                return SimpleNamespace(returncode=3 if fail else 0)

            with patch.object(runner, '__file__', str(self.root / f'matrix_{fail}' / 'run_all.py')), \
                 patch.object(sys, 'argv', ['run_all.py', '--num_workers', '0']), \
                 patch.object(runner.subprocess, 'run', side_effect=execute), \
                 contextlib.redirect_stdout(io.StringIO()):
                if fail:
                    with self.assertRaises(SystemExit) as caught:
                        runner.main()
                    self.assertEqual(caught.exception.code, 3)
                else:
                    runner.main()
            self.assertEqual(len(commands), 1 if fail else 6)
            if not fail:
                pairs = {(cmd[cmd.index('--feature_source')+1], cmd[cmd.index('--survival_loss')+1]) for cmd in commands}
                self.assertEqual(pairs, {(source, loss) for source in ['uni_l2','keep','eagle'] for loss in ['nll','cox_rank']})

    def test_launchers_have_valid_bash_syntax(self):
        for name in ['train.sh', 'run_all.sh']:
            subprocess.run(['bash', '-n', str(HERE / name)], check=True)


if __name__ == '__main__':
    unittest.main()
