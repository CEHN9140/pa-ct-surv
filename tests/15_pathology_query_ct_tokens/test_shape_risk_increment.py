import sys
import tempfile
import unittest
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from shape_risk_increment import (fixed_shape_policy, checked_predictions, assemble_ct_oof,
                                  fit_risk_heads, check_shape_metadata)


class ShapeRiskIncrementTests(unittest.TestCase):
    def test_fixed_weight_epochs_do_not_use_held_scores_or_selected_weight(self):
        selection = dict(weight=.05, epochs=30, inner_models=[
            dict(weight=weight, inner_fold=i, selected_epoch=epoch, held_cindex=score)
            for weight, epochs, score in [(.05, [30, 30, 30], .99), (.2, [29, 9, 16], .01)]
            for i, epoch in enumerate(epochs)])
        policy = fixed_shape_policy(selection, .2)
        self.assertEqual(policy['epochs'], 16)
        for row in selection['inner_models']:
            row['held_cindex'] = 1 - row['held_cindex']
        selection.update(weight=0., epochs=1)
        self.assertEqual(fixed_shape_policy(selection, .2), policy)
        selection['inner_models'].pop()
        with self.assertRaises(ValueError):
            fixed_shape_policy(selection, .2)

    def sample_blocks(self):
        samples = pd.DataFrame(dict(pa_id=list('abcdefgh'), ct_id=[str(i) for i in range(8)],
                                    time=np.arange(1., 9.), event=[1, 0] * 4))
        training, validation = np.arange(6), np.arange(6, 8)
        blocks = []
        for inner in range(3):
            held = np.arange(inner * 2, inner * 2 + 2)
            remaining = np.setdiff1d(training, held)
            fit, stop = remaining[:-1], remaining[-1:]
            def frame(indices):
                part = samples.iloc[indices]
                return pd.DataFrame(dict(case_id=(part.pa_id + '|' + part.ct_id).to_numpy(),
                    **{'dfs.month': part.time.to_numpy(), 'dfs.status': part.event.to_numpy()},
                    risk_score=indices.astype(float) * (inner + 1) + inner * 100))
            blocks.append(dict(inner_fold=inner, fit=fit, stop=stop, held=held,
                               fit_predictions=frame(fit), held_predictions=frame(held)))
        return samples, training, validation, blocks

    def test_oof_excludes_fit_stop_and_outer_patients_and_scales_on_fit_only(self):
        samples, training, validation, blocks = self.sample_blocks()
        oof, scalers = assemble_ct_oof(samples, training, validation, blocks)
        self.assertEqual(len(oof), 6)
        self.assertTrue(oof.index.is_unique)
        expected_mean = float(blocks[0]['fit_predictions'].risk_score.mean())
        self.assertEqual(scalers[0]['mean'], expected_mean)
        first = oof.loc['a|0', 'ct_risk']
        blocks[0]['held_predictions']['risk_score'] += 1e6
        altered, again = assemble_ct_oof(samples, training, validation, blocks)
        self.assertEqual(scalers, again)
        self.assertNotEqual(first, altered.loc['a|0', 'ct_risk'])
        blocks[0]['stop'] = np.r_[blocks[0]['stop'], blocks[0]['held'][0]]
        with self.assertRaises(ValueError):
            assemble_ct_oof(samples, training, validation, blocks)

    def test_prediction_identity_event_and_float32_time_validation(self):
        samples, training, validation, blocks = self.sample_blocks()
        samples.loc[0, 'time'] = 3.8666666666666663
        frame = blocks[0]['held_predictions'].copy()
        frame.loc[0, 'dfs.month'] = np.float32(samples.loc[0, 'time'])
        checked = checked_predictions(frame, samples, blocks[0]['held'])
        self.assertEqual(checked.iloc[0]['dfs.month'], samples.loc[0, 'time'])
        frame.loc[0, 'dfs.status'] = 0
        with self.assertRaises(ValueError):
            checked_predictions(frame, samples, blocks[0]['held'])

    def test_missing_oof_coverage_or_outer_leak_is_rejected(self):
        samples, training, validation, blocks = self.sample_blocks()
        with self.assertRaises(ValueError):
            assemble_ct_oof(samples, training, validation, blocks[:-1])
        blocks[0]['fit'] = np.r_[blocks[0]['fit'], validation[0]]
        with self.assertRaises(ValueError):
            assemble_ct_oof(samples, training, validation, blocks)

    def test_cox_heads_use_oof_rows_and_share_fixed_regularization(self):
        rng = np.random.default_rng(9)
        frame = pd.DataFrame(rng.normal(size=(100, 3)), columns=['pathology_risk', 'ct_A', 'ct_C'])
        frame['dfs.month'] = np.exp(3 - frame.pathology_risk - .4 * frame.ct_C + rng.normal(size=100))
        frame['dfs.status'] = np.tile([1, 0, 1, 1], 25)
        heads = fit_risk_heads(frame, 1.)
        self.assertEqual(set(heads), {'M0', 'M1', 'M2'})
        for name, head in heads.items():
            self.assertEqual(head.pa_scaler.n_samples_seen_, 100)
            self.assertAlmostEqual(head.pa_scaler.mean_[0], frame.pathology_risk.mean())
            if name != 'M0':
                np.testing.assert_array_equal(head.model.alpha, [0., 1.])
                key = 'ct_A' if name == 'M1' else 'ct_C'
                self.assertAlmostEqual(head.ct_pipeline.named_steps['scale'].mean_[0], frame[key].mean())
        before = heads['M2'].pa_scaler.mean_.copy()
        shifted_ct = (frame[['ct_C']] - 1000).to_numpy()
        risk = heads['M2'].predict(frame.pathology_risk + 1000, shifted_ct)
        np.testing.assert_array_equal(heads['M2'].pa_scaler.mean_, before)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'head.joblib'
            joblib.dump(heads['M2'], path)
            np.testing.assert_array_equal(joblib.load(path).predict(frame.pathology_risk + 1000, shifted_ct), risk)

    def test_checkpoint_shape_scaler_is_bound_to_exact_fit_population(self):
        samples, training, validation, blocks = self.sample_blocks()
        columns = ['log_volume_mm3', 'diameter_mm', 'sphericity']
        shapes = pd.DataFrame(np.arange(24.).reshape(8, 3), index=samples.ct_id, columns=columns)
        fit = blocks[0]['fit']
        ids = samples.iloc[fit].ct_id.tolist()
        target = shapes.loc[ids].to_numpy()
        metadata = dict(weight=.2, selected_epoch=16,
            initialization='PUBLIC MedicalNet + fresh original risk head; no downstream warm-start',
            scaler=dict(columns=columns, n_fit=len(fit), mean=target.mean(0).tolist(), std=target.std(0).tolist(),
                        fit_ct_ids_sha256=hashlib.sha256('\n'.join(ids).encode()).hexdigest()))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'metadata.json'
            path.write_text(json.dumps(metadata))
            check_shape_metadata(path, samples, fit, shapes, SimpleNamespace(shape_columns=columns), .2, 16)
            with self.assertRaises(ValueError):
                check_shape_metadata(path, samples, fit[::-1], shapes, SimpleNamespace(shape_columns=columns), .2, 16)
            with self.assertRaises(ValueError):
                check_shape_metadata(path, samples, fit, shapes, SimpleNamespace(shape_columns=columns), .05, 16)


if __name__ == '__main__':
    unittest.main()
