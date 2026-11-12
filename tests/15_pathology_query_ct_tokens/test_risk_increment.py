import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import torch
from sksurv.metrics import concordance_index_censored
from torch.utils.data import DataLoader, TensorDataset

from loss import cox_loss
from risk_calibration import fit_alpha, inner_splits, paired_comparison, risk_scaler, standardize_risk
from risk_increment import predict_risk, train_risk_model


class RiskIncrementTests(unittest.TestCase):
    def test_inner_oof_fit_and_stop_roles_are_disjoint_and_exclude_outer_validation(self):
        frame = pd.DataFrame({'event': np.tile([0, 1], 60), 'time': np.arange(120) + 1})
        outer_train = np.arange(96)
        partitions = inner_splits(frame, outer_train, 3, .2, 123)
        predicted = []
        for fit, stop, oof in partitions:
            self.assertFalse(set(fit) & set(stop) or set(fit) & set(oof) or set(stop) & set(oof))
            self.assertEqual(set(fit) | set(stop) | set(oof), set(outer_train))
            for indices in (fit, stop, oof):
                self.assertEqual(set(frame.iloc[indices].event), {0, 1})
            predicted.extend(oof)
        self.assertEqual(sorted(predicted), outer_train.tolist())
        changed = frame.copy()
        changed.loc[96:, 'event'] = 1
        changed.loc[96:, 'time'] = 1e9
        altered = inner_splits(changed, outer_train, 3, .2, 123)
        for first, second in zip(partitions, altered):
            for a, b in zip(first, second):
                np.testing.assert_array_equal(a, b)

    def test_standardization_uses_only_provided_fit_predictions(self):
        training = np.array([1., 2., 3., 4.])
        scaler = risk_scaler(training)
        normalized = standardize_risk(training, scaler)
        self.assertAlmostEqual(normalized.mean(), 0.)
        self.assertAlmostEqual(normalized.std(ddof=0), 1.)
        before = dict(scaler)
        standardize_risk(np.array([-1e6, 1e6]), scaler)
        self.assertEqual(scaler, before)
        np.testing.assert_array_equal(standardize_risk(np.ones(3), risk_scaler(np.ones(5))), np.zeros(3))
        with self.assertRaises(ValueError):
            risk_scaler(np.array([1., np.nan]))

    def test_alpha_uses_existing_cox_loss_and_can_be_negative(self):
        rng = np.random.default_rng(123)
        p = rng.normal(size=80)
        ct = rng.normal(size=80)
        time = rng.exponential(scale=np.exp(-.3*p - .8*ct)) + .01
        event = rng.binomial(1, .7, size=80)
        fitted = fit_alpha(p, ct, time, event)
        negative = fit_alpha(p, -ct, time, event)
        self.assertGreater(fitted['alpha'], 0)
        self.assertLess(negative['alpha'], 0)
        self.assertAlmostEqual(fitted['alpha'], -negative['alpha'], places=5)
        self.assertLessEqual(fitted['fitted_cox'], fitted['pathology_only_cox'] + 1e-10)
        loss = cox_loss(torch.tensor(p + fitted['alpha'] * ct), torch.tensor(time), torch.tensor(event))
        self.assertAlmostEqual(float(loss), fitted['fitted_cox'], places=10)
        for alpha in (-2., 0., .5, 1., 2.):
            trial = cox_loss(torch.tensor(p+alpha*ct), torch.tensor(time), torch.tensor(event))
            self.assertLessEqual(fitted['fitted_cox'], float(trial)+1e-7)
        with self.assertRaises(ValueError):
            fit_alpha(p, ct, time, np.zeros_like(event))
        constant = fit_alpha(p, np.zeros_like(ct), time, event)
        self.assertEqual(constant['alpha'], 0.)

    def test_paired_bootstrap_compares_only_within_folds(self):
        predictions = []
        for shift in (0., 1000.):
            time = np.arange(1., 41.)
            predictions.append(pd.DataFrame({'dfs.month': time, 'dfs.status': np.tile([1, 0], 20),
                'pathology_risk': -time + shift, 'ct_risk': time - shift,
                'raw_pathology_risk': -time + shift, 'raw_ct_risk': time - shift,
                'risk_score': -time + shift - 5000.}))
        result = paired_comparison(predictions, repeats=50)
        for key in ('delta_cindex', 'lower', 'upper'):
            self.assertEqual(result['fusion_minus_pathology'][key], 0.)
            self.assertEqual(result['fusion_minus_ct'][key], 1.)

    def test_paired_bootstrap_keeps_raw_baseline_risk_tie_tolerance(self):
        time = np.arange(1., 41.)
        event = np.tile([1, 0], 20)
        raw = -time * 1e-9
        frame = pd.DataFrame({'dfs.month': time, 'dfs.status': event,
            'pathology_risk': raw/raw.std(), 'ct_risk': time,
            'raw_pathology_risk': raw, 'raw_ct_risk': time, 'risk_score': -time})
        expected = 1. - concordance_index_censored(event.astype(bool), time, raw)[0]
        self.assertGreater(expected, 0)
        result = paired_comparison([frame], repeats=50)
        self.assertAlmostEqual(result['fusion_minus_pathology']['delta_cindex'], expected)

    def test_fixed_epoch_refit_has_no_stop_predictions_and_keeps_tail_patients(self):
        torch.manual_seed(123)
        x = torch.randn(7, 3)
        dataset = TensorDataset(x, torch.tensor([1, 0, 1, 0, 1, 0, 1]), torch.arange(1., 8.), torch.arange(7))
        loader = DataLoader(dataset, batch_size=1)
        model = torch.nn.Linear(3, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=.01)
        steps = []
        optimizer.register_step_post_hook(lambda *args: steps.append(1))
        settings = SimpleNamespace(num_epochs=2, patience=10)
        with tempfile.TemporaryDirectory() as directory:
            model, info = train_risk_model(model, optimizer, loader, None, settings, 'cpu',
                                           Path(directory), forwards_per_update=3, fixed_epochs=2)
            self.assertEqual(info['selection'], 'fixed_epochs_from_inner_training')
            self.assertEqual(info['selected_epoch'], 2)
            history = pd.read_csv(Path(directory) / 'training_history.csv')
            self.assertNotIn('val_cindex', history)
            self.assertEqual(len(history), 2)
            predictions = predict_risk(model, loader, 'cpu')
            self.assertEqual(len(predictions), 7)
            self.assertEqual(set(predictions.case_id), set(str(i) for i in range(7)))
            restored = torch.nn.Linear(3, 1)
            restored.load_state_dict(torch.load(Path(directory)/'best_model.pth', weights_only=True), strict=True)
            np.testing.assert_allclose(predict_risk(restored, loader, 'cpu').risk_score,
                                       predictions.risk_score)
        # Seven patients, risk sets [3,4], two epochs: four joint-Cox updates.
        self.assertEqual(len(steps), 4)

    def test_early_stopping_uses_only_supplied_stop_loader(self):
        dataset = TensorDataset(torch.tensor([[2.], [1.], [0.], [-1.]]), torch.tensor([1, 0, 1, 0]),
                                torch.arange(1., 5.), torch.arange(4))
        loader = DataLoader(dataset, batch_size=1)
        model = torch.nn.Linear(1, 1)
        with torch.no_grad():
            model.weight.fill_(1.); model.bias.zero_()
        settings = SimpleNamespace(num_epochs=8, patience=2)
        with tempfile.TemporaryDirectory() as directory:
            _, info = train_risk_model(model, torch.optim.SGD(model.parameters(), lr=0), loader, loader,
                                       settings, 'cpu', Path(directory), forwards_per_update=2)
            self.assertEqual(info['selection'], 'inner_stop_cindex')
            self.assertEqual(info['selected_epoch'], 1)
            self.assertEqual(info['epochs_completed'], 3)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                train_risk_model(model, torch.optim.SGD(model.parameters(), lr=0), loader, loader,
                                 settings, 'cpu', Path(directory), forwards_per_update=2, fixed_epochs=2)


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
