import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import h5py
import numpy as np
import pandas as pd
import unittest
from tempfile import TemporaryDirectory
from sksurv.metrics import concordance_index_censored


def module():
    path = Path(__file__).with_name('analyze.py')
    assert path.is_file(), 'Experiment07 implementation is missing'
    spec = importlib.util.spec_from_file_location('distribution_analysis', path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_all_patch_summary_not_sampled(tmp_path):
    m = module()
    x = np.array([[3., 4.], [0., 0.], [6., 8.], [1., 2.], [20., 30.]])
    path = tmp_path / 'features.h5'
    with h5py.File(path, 'w') as f:
        f['features'] = x
    mean, stats = m.summarize_features(path, input_dim=2, block_size=2)
    np.testing.assert_allclose(mean, x.mean(0))
    assert stats['n_patches'] == 5
    assert stats['zero_patches'] == 1
    np.testing.assert_allclose(stats['patch_norm_mean'], np.linalg.norm(x, axis=1).mean())
    np.testing.assert_allclose(stats['patch_norm_sd'], np.linalg.norm(x, axis=1).std())
    with h5py.File(path, 'r+') as f:
        f['features'][3, 1] = np.nan
    with unittest.TestCase().assertRaisesRegex(ValueError, 'finite'):
        m.summarize_features(path, input_dim=2)


def test_comparable_pairs_ties_patient_errors_and_strata():
    m = module()
    t = np.array([3., 3., 12., 24., 36., 40., 60., 70.])
    e = np.array([1, 0, 1, 0, 1, 1, 0, 1], dtype=bool)
    r = np.array([2., 2., 1., 4., 3., 2., 0., -1.])
    frame = pd.DataFrame({'dfs.month': t, 'dfs.status': e, 'risk_score': r})
    result, errors, strata = m.pair_analysis(frame)
    np.testing.assert_allclose(result['cindex'], concordance_index_censored(e, t, r)[0])
    assert result['n_pairs'] == strata.n_pairs.sum()
    np.testing.assert_allclose(np.average(errors.error_rate, weights=errors.pair_count), 1-result['cindex'])
    assert errors.pair_count.sum() == 2*result['n_pairs']
    scaled = frame.copy()
    scaled.risk_score = 3*r+10
    np.testing.assert_allclose(m.pair_analysis(scaled)[0]['cindex'], result['cindex'])


def test_predictions_must_match_exact_patients_labels_and_partition(tmp_path):
    m = module()
    data = pd.DataFrame({'pa_id': ['01', '02'], 'time': [12., 24.], 'event': [1, 0]})
    prediction = pd.DataFrame({'case_id': ['02', '01'], 'dfs.month': [24., 12.],
                               'dfs.status': [0, 1], 'risk_score': [0., 1.]})
    path = tmp_path/'pred.csv'
    prediction.to_csv(path, index=False)
    aligned = m.read_predictions(path, data)
    assert aligned.case_id.tolist() == ['01', '02']
    # Existing training loaders/exporters use float32 DFS time, while source CSV uses float64.
    data.loc[0, 'time'] = 97.86666666666667
    prediction.loc[1, 'dfs.month'] = np.float32(data.loc[0, 'time'])
    prediction.to_csv(path, index=False)
    m.read_predictions(path, data)
    prediction.loc[0, 'dfs.status'] = 1
    prediction.to_csv(path, index=False)
    with unittest.TestCase().assertRaisesRegex(ValueError, 'labels'):
        m.read_predictions(path, data)
    prediction.iloc[:1].to_csv(path, index=False)
    with unittest.TestCase().assertRaisesRegex(ValueError, 'patients'):
        m.read_predictions(path, data)


def test_event_stratified_permutations_preserve_origin_counts():
    m = module()
    event = np.repeat([0, 1], 30)
    origin = np.tile([0, 0, 0, 0, 1], 12)
    permutations = m.permute_within_events(origin, event, 25, 123)
    assert permutations.shape == (25, 60)
    for labels in permutations:
        for e in [0, 1]:
            assert labels[event == e].sum() == origin[event == e].sum()
    assert not np.array_equal(permutations[0], origin)


def test_classifier_detects_shift_with_heldout_predictions():
    m = module()
    rng = np.random.default_rng(42)
    origin = np.tile([0, 0, 0, 0, 1], 40)
    event = np.tile([0, 1], 100)
    x = rng.normal(size=(200, 12)) + origin[:, None]*3
    result, probabilities, null = m.origin_test(x, origin, event, seed=123,
                                               n_permutations=19, pca_dim=4)
    assert result['origin_auc'] > .97
    assert result['origin_p'] <= .05
    assert len(null) == 19
    assert probabilities.shape == (200,)
    assert np.isfinite(probabilities).all()
    # A held-out change cannot affect the preprocessing fit of its inner fold.
    inner = m.origin_embeddings(x, event, 123, 4)
    changed = x.copy()
    changed[inner[0][1]] += 100
    inner_changed = m.origin_embeddings(changed, event, 123, 4)
    np.testing.assert_allclose(inner[0][2], inner_changed[0][2])


def test_distance_is_fit_only_on_survival_training_patients():
    m = module()
    rng = np.random.default_rng(3)
    train = rng.normal(size=(40, 8))
    val = rng.normal(size=(10, 8))
    a = m.training_distances(train, val, seed=123, pca_dim=4)
    changed = val.copy()
    changed[-1] += 100
    b = m.training_distances(train, changed, seed=123, pca_dim=4)
    np.testing.assert_allclose(a[:-1], b[:-1])
    assert b[-1] > a[-1]


def test_feature_cache_invalidates_when_input_changes(tmp_path):
    m = module()
    path = tmp_path/'input.h5'
    with h5py.File(path, 'w') as f:
        f['features'] = np.ones((3, 1024))
    data = pd.DataFrame({'pa_path': [str(path)]})
    cache = tmp_path/'cache.npz'
    first = m.load_feature_summaries(data, cache, 'csv_hash')
    second = m.load_feature_summaries(data, cache, 'csv_hash')
    np.testing.assert_array_equal(first[0], second[0])
    assert first[2] == second[2]
    with h5py.File(path, 'r+') as f:
        f['features'][0] = 4
    third = m.load_feature_summaries(data, cache, 'csv_hash')
    assert third[2] != first[2]
    np.testing.assert_allclose(third[0], 2.)


def test_end_to_end_analysis_outputs_and_preserves_inputs(tmp_path):
    rng = np.random.default_rng(32)
    n = 200
    cohort = pd.DataFrame({'pa_id': [str(i) for i in range(n)],
                           'ct_id': [str(i+1000) for i in range(n)],
                           'event': np.tile([0, 1], n//2),
                           'time': rng.uniform(1, 100, n), 'split': np.arange(n)%5})
    paths = []
    for i in range(n):
        path = tmp_path/f'{i}.h5'
        with h5py.File(path, 'w') as f:
            f['features'] = rng.normal(size=(3, 1024)).astype('float32')
        paths.append(str(path))
    cohort['pa_path'] = paths
    csv = tmp_path/'cohort.csv'
    cohort.to_csv(csv, index=False)
    inputs = [csv]
    for fold in range(5):
        directory = tmp_path/'predictions'/f'fold_{fold}'
        directory.mkdir(parents=True)
        for part in ['train', 'val']:
            selected = cohort.split == fold if part == 'val' else cohort.split != fold
            f = cohort.loc[selected, ['pa_id', 'time', 'event']].copy()
            f.columns = ['case_id', 'dfs.month', 'dfs.status']
            f['risk_score'] = -f['dfs.month']+rng.normal(size=len(f))*40
            path = directory/f'{part}_predictions.csv'
            f.to_csv(path, index=False)
            inputs.append(path)
    original = {str(p): p.read_bytes() for p in inputs}
    output = tmp_path/'results'
    result = subprocess.run([sys.executable, str(Path(__file__).with_name('analyze.py')),
                             '--data_csv', str(csv), '--prediction_root', str(tmp_path/'predictions'),
                             '--metadata_csv', str(tmp_path/'absent.csv'), '--output_root', str(output),
                             '--permutations', '9', '--pca_dim', '4'],
                            env={**os.environ, 'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1'},
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout+'\n'+result.stderr
    assert {str(p): p.read_bytes() for p in inputs} == original
    manifest = json.loads((output/'manifest.json').read_text())
    assert manifest['n_patients'] == n and manifest['n_patches'] == 3*n
    summary = pd.read_csv(output/'fold_summary.csv')
    assert len(summary) == 5
    assert summary.train_n.eq(160).all() and summary.val_n.eq(40).all()
    assert len(pd.read_csv(output/'origin_permutations.csv')) == 45
    patients = pd.read_csv(output/'patient_errors.csv')
    assert len(patients) == 5*n
    assert len(patients.loc[patients.part == 'val']) == n
    assert (output/'report.md').is_file() and (output/'diagnostic_overview.png').stat().st_size > 1000


if __name__ == '__main__':
    suite = unittest.TestSuite()
    for name, function in list(globals().items()):
        if name.startswith('test_'):
            if 'tmp_path' in function.__code__.co_varnames[:function.__code__.co_argcount]:
                def wrapped(function=function):
                    with TemporaryDirectory() as directory:
                        function(Path(directory))
                suite.addTest(unittest.FunctionTestCase(wrapped, description=name))
            else:
                suite.addTest(unittest.FunctionTestCase(function))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(not result.wasSuccessful())
