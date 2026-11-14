import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
import SimpleITK as sitk

sys.path.insert(0, str(Path(__file__).resolve().parent))
from radiomics_ct import (RadiomicsFilter, extract_case, feature_names, fit_cox, load_cohort,
                          load_features, make_inner_splits, select_penalty)


def test_filter_is_training_only_and_removes_constants_and_duplicates():
    x = pd.DataFrame({'a': [1., 4., 2., 3.], 'b': [3., 2., 4., 1.],
                      'duplicate': [2., 8., 4., 6.], 'constant': [7.] * 4})
    selector = RadiomicsFilter().fit(x)
    assert selector.selected_ == ['a', 'b']
    held = pd.DataFrame({'a': [100.], 'b': [-100.], 'duplicate': [0.], 'constant': [9.]})
    np.testing.assert_array_equal(selector.transform(held), [[100., -100.]])
    assert selector.audit_.set_index('feature').loc['constant', 'reason'] == 'constant'
    assert selector.audit_.set_index('feature').loc['duplicate', 'correlated_with'] == 'a'
    with pytest.raises(ValueError, match='finite'):
        selector.transform(held.assign(a=np.nan))
    with pytest.raises(ValueError, match='constant'):
        RadiomicsFilter().fit(pd.DataFrame({'a': [1., 1., 1.]}))


def survival_sample():
    rng = np.random.default_rng(123)
    x = pd.DataFrame(rng.normal(size=(60, 4)), columns=list('abcd'))
    x['constant'] = 1.
    time = np.exp(3 - 0.7 * x.a + rng.normal(size=60))
    event = np.tile([0, 1, 1], 20)
    return x, np.asarray(time), event


def test_cox_uses_training_scaler_and_reload_preserves_predictions(tmp_path):
    x, time, event = survival_sample()
    model = fit_cox(x, time, event, 10.)
    chosen = model.named_steps['filter'].selected_
    np.testing.assert_allclose(model.named_steps['scale'].mean_, x[chosen].mean())
    before = model.named_steps['scale'].mean_.copy()
    prediction = model.predict(x.iloc[:5] + 100.)
    np.testing.assert_array_equal(model.named_steps['scale'].mean_, before)
    joblib.dump(model, tmp_path / 'model.joblib')
    np.testing.assert_array_equal(joblib.load(tmp_path / 'model.joblib').predict(x.iloc[:5] + 100.), prediction)


def test_inner_splits_and_tuning_never_receive_outer_patients():
    x, time, event = survival_sample()
    splits = make_inner_splits(event)
    held = []
    for fit, val in splits:
        assert not set(fit) & set(val)
        assert set(fit) | set(val) == set(range(len(x)))
        held.extend(val)
    assert sorted(held) == list(range(len(x)))
    selected, table = select_penalty(x, time, event, penalties=[1., 10.])
    assert selected in [1., 10.]
    assert len(table) == 6
    assert np.isfinite(table.cindex).all()


def make_images(tmp_path, mismatch=False):
    z, y, x = np.indices((18, 20, 22))
    mask_values = ((z - 9)**2 + (y - 10)**2 + (x - 11)**2 < 25).astype(np.uint8)
    values = (-800 + 20*x + 7*y + 3*z).astype(np.float32)
    image, mask = sitk.GetImageFromArray(values), sitk.GetImageFromArray(mask_values)
    image.SetOrigin((5., -9., 12.))
    mask.CopyInformation(image)
    if mismatch:
        mask.SetOrigin((6., -9., 12.))
    for name, obj in [('image.nii.gz', image), ('mask.nii.gz', mask)]:
        sitk.WriteImage(obj, str(tmp_path / name))
    return dict(ct_id='1', image_path=str(tmp_path / 'image.nii.gz'),
                mask_path=str(tmp_path / 'mask.nii.gz'))


def test_original_extraction_has_107_features_and_tracks_geometry(tmp_path):
    result = extract_case(make_images(tmp_path, mismatch=True))
    assert result['error'] is None
    assert len(result['features']) == 107
    assert all(name.startswith('original_') for name in result['features'])
    assert np.isfinite(list(result['features'].values())).all()
    assert result['qc']['geometry_corrected']
    native_mask = sitk.GetArrayFromImage(sitk.ReadImage(str(tmp_path / 'mask.nii.gz'))).astype(bool)
    native_image = sitk.GetArrayFromImage(sitk.ReadImage(str(tmp_path / 'image.nii.gz')))
    assert result['features']['original_shape_VoxelVolume'] == int(native_mask.sum())
    # Mask origin is shifted +1 mm along x; CT increases by 20 units per x voxel.
    np.testing.assert_allclose(result['features']['original_firstorder_Mean'], native_image[native_mask].mean() + 20.)
    # Shape uses physical mask geometry; image intensity shift must not change it.
    path = tmp_path / 'image.nii.gz'
    image = sitk.ReadImage(str(path))
    sitk.WriteImage(image + 100, str(path))
    shifted = extract_case(dict(ct_id='1', image_path=str(path), mask_path=str(tmp_path/'mask.nii.gz')))
    assert shifted['features']['original_shape_VoxelVolume'] == result['features']['original_shape_VoxelVolume']
    np.testing.assert_allclose(shifted['features']['original_firstorder_Mean'] - result['features']['original_firstorder_Mean'], 100.)


def test_invalid_mask_is_recorded_as_failure(tmp_path):
    row = make_images(tmp_path)
    mask = sitk.ReadImage(row['mask_path'])
    sitk.WriteImage(mask * 2, row['mask_path'])
    result = extract_case(row)
    assert 'binary' in result['error']
    assert 'features' not in result


def test_cohort_validation_and_cache_identity(tmp_path):
    cohort = pd.DataFrame({'pa_id': list('abcde'), 'ct_id': list('12345'),
                           'event': [1, 0, 1, 0, 1], 'time': [1., 2., 3., 4., 5.],
                           'split': list(range(5))})
    path = tmp_path / 'cohort.csv'
    cohort.to_csv(path, index=False)
    data = load_cohort(path, tmp_path)
    assert data.ct_id.tolist() == list('12345')
    assert data['dfs.month'].tolist() == [1., 2., 3., 4., 5.]
    features = pd.concat([data, pd.DataFrame({name: [float(i)] * len(data)
                         for i, name in enumerate(feature_names())})], axis=1)
    features.to_csv(tmp_path / 'features.csv', index=False)
    loaded = load_features(tmp_path / 'features.csv', data)
    assert len(loaded) == 5
    features.loc[0, 'dfs.status'] = 0
    features.to_csv(tmp_path / 'features.csv', index=False)
    with pytest.raises(ValueError, match='labels|cohort'):
        load_features(tmp_path / 'features.csv', data)
    features.loc[0, 'dfs.status'] = 1
    features = features.drop(columns=feature_names()[0])
    features.to_csv(tmp_path / 'features.csv', index=False)
    with pytest.raises(ValueError, match='107'):
        load_features(tmp_path / 'features.csv', data)
    cohort.loc[0, 'split'] = -1
    cohort.to_csv(path, index=False)
    with pytest.raises(ValueError, match='five|fold'):
        load_cohort(path, tmp_path)


def test_cache_csv_roundtrip_preserves_time_and_still_rejects_label_changes(tmp_path):
    cohort = pd.DataFrame({'pa_id': ['a'], 'ct_id': ['1'], 'case_id': ['1'],
                           'dfs.month': [3.8666666666666663], 'dfs.status': [1],
                           'split': [0], 'image_path': ['image.nii.gz'], 'mask_path': ['mask.nii.gz']})
    features = pd.concat([cohort, pd.DataFrame({name: [float(i)]
                         for i, name in enumerate(feature_names())})], axis=1)
    path = tmp_path / 'features.csv'
    features.to_csv(path, index=False)
    loaded = load_features(path, cohort)
    np.testing.assert_array_equal(loaded['dfs.month'], cohort['dfs.month'])
    # Exact validation remains active, even for a one-ULP actual cache change.
    features.loc[0, 'dfs.month'] = np.nextafter(cohort.loc[0, 'dfs.month'], np.inf)
    features.to_csv(path, index=False)
    with pytest.raises(ValueError, match='dfs.month'):
        load_features(path, cohort)
