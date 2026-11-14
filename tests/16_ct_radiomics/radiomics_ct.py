"""Experiment 16: Original 3D CT Radiomics and nested Ridge-Cox; no neural models."""

import hashlib
import json
import logging
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import radiomics
import SimpleITK as sitk
import yaml
from radiomics import featureextractor
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
from sksurv.util import Surv


def extraction_settings():
    return dict(binWidth=25, normalize=False, resampledPixelSpacing=[1., 1., 1.],
                interpolator=sitk.sitkBSpline, correctMask=True, geometryTolerance=1e-6,
                force2D=False, minimumROIDimensions=3, distances=[1],
                symmetricalGLCM=True, weightingNorm=None, voxelArrayShift=0,
                additionalInfo=False, label=1)


def feature_names():
    names = sorted('original_' + group + '_' + name
                   for group, cls in radiomics.getFeatureClasses().items() if group != 'shape2D'
                   for name, deprecated in cls.getFeatureNames().items() if not deprecated)
    if radiomics.__version__ != 'v3.0.1' or len(names) != 107:
        raise RuntimeError('Use PyRadiomics 3.0.1 with its 107 nondeprecated 3D features')
    return names


def load_cohort(csv_path, ct_root):
    data = pd.read_csv(csv_path, dtype={'pa_id': str, 'ct_id': str})
    required = ['pa_id', 'ct_id', 'time', 'event', 'split']
    if not set(required).issubset(data) or data[required].isna().any().any():
        raise ValueError('Incomplete patient identities, labels or folds')
    if data.ct_id.duplicated().any() or data.pa_id.duplicated().any():
        raise ValueError('Require exactly one CT per independent patient')
    for key in ['time', 'event', 'split']:
        data[key] = pd.to_numeric(data[key], errors='raise')
    if not np.isfinite(data[['time', 'event', 'split']]).all().all():
        raise ValueError('Nonfinite cohort labels')
    if not data.event.isin([0, 1]).all() or (data.time < 0).any():
        raise ValueError('Invalid survival labels')
    if set(data.split) != set(range(5)):
        raise ValueError('Require the existing five folds (0..4), without test split')
    root = Path(ct_root).resolve()
    result = data[required].rename(columns={'time': 'dfs.month', 'event': 'dfs.status'}).copy()
    result['case_id'] = result.ct_id
    result['image_path'] = [str(root / 'image' / (ct + '.nii.gz')) for ct in result.ct_id]
    result['mask_path'] = [str(root / 'mask' / (ct + '.nii.gz')) for ct in result.ct_id]
    return result


def extract_case(row):
    try:
        image = sitk.ReadImage(row['image_path'], sitk.sitkFloat32)
        mask = sitk.ReadImage(row['mask_path'])
        if image.GetDimension() != 3 or mask.GetDimension() != 3:
            raise ValueError('Require 3D CT and mask')
        values, labels = sitk.GetArrayViewFromImage(image), sitk.GetArrayViewFromImage(mask)
        if not np.isfinite(values).all():
            raise ValueError('CT values must be finite')
        if not np.isin(np.unique(labels), [0, 1]).all() or not (labels == 1).any():
            raise ValueError('Require a nonempty binary mask with tumor label 1')
        same = image.GetSize() == mask.GetSize() and all(np.allclose(a, b, rtol=0, atol=1e-6)
            for a, b in [(image.GetOrigin(), mask.GetOrigin()),
                         (image.GetSpacing(), mask.GetSpacing()), (image.GetDirection(), mask.GetDirection())])
        extractor = featureextractor.RadiomicsFeatureExtractor(**extraction_settings())
        extractor.disableAllImageTypes()
        extractor.enableImageTypeByName('Original')
        extractor.disableAllFeatures()
        for group in radiomics.getFeatureClasses():
            if group != 'shape2D':
                extractor.enableFeatureClassByName(group)
        raw = extractor.execute(image, mask)
        names = feature_names()
        if set(raw) != set(names):
            raise ValueError('Extracted feature schema differs from Original 107')
        features = {key: float(np.asarray(raw[key]).item()) for key in names}
        if not np.isfinite(list(features.values())).all():
            raise ValueError('Nonfinite Radiomics features; investigate before model fitting')
        qc = dict(ct_id=row['ct_id'], image_path=row['image_path'], mask_path=row['mask_path'],
                  geometry_corrected=not same, image_size_xyz=list(image.GetSize()),
                  image_spacing_xyz=list(image.GetSpacing()), mask_voxels_native=int((labels == 1).sum()),
                  image_min=float(values.min()), image_max=float(values.max()))
        return dict(ct_id=row['ct_id'], features=features, qc=qc, error=None)
    except Exception as exc:
        return dict(ct_id=row['ct_id'], error=f'{type(exc).__name__}: {exc}')


def load_features(path, cohort):
    # Recover the exact floats written by to_csv; keep label equality strict.
    data = pd.read_csv(path, dtype={'case_id': str, 'ct_id': str, 'pa_id': str},
                       float_precision='round_trip')
    if data.ct_id.duplicated().any() or set(data.ct_id) != set(cohort.ct_id):
        raise ValueError('Cached features differ from the complete cohort')
    data = data.set_index('ct_id').loc[cohort.ct_id].reset_index()
    for key in ['case_id', 'pa_id', 'dfs.month', 'dfs.status', 'split', 'image_path', 'mask_path']:
        if key not in data or not np.array_equal(data[key].to_numpy(), cohort[key].to_numpy()):
            raise ValueError(f'Cached labels/cohort identity changed: {key}')
    columns = [key for key in data if key.startswith('original_')]
    if set(columns) != set(feature_names()) or not np.isfinite(data[columns].to_numpy()).all():
        raise ValueError('Require a complete finite Original 107 feature table')
    return data


def extraction_signature(cohort, csv_path):
    sources = []
    for row in cohort.to_dict('records'):
        for key in ['image_path', 'mask_path']:
            stat = Path(row[key]).stat()
            sources.append([row[key], stat.st_size, stat.st_mtime_ns])
    return dict(cohort_sha256=hashlib.sha256(Path(csv_path).read_bytes()).hexdigest(),
                settings=extraction_settings(), feature_names=feature_names(), sources=sources,
                pyradiomics=radiomics.__version__, simpleitk=sitk.Version_VersionString(),
                source_check='file path + size + mtime_ns; not full image-content hashes')


def prepare_features(cohort, csv_path, root, workers, allow_extract):
    path, manifest = root / 'features.csv', root / 'extraction_manifest.json'
    signature = extraction_signature(cohort, csv_path)
    if path.exists():
        if not manifest.is_file() or json.loads(manifest.read_text()) != signature:
            raise ValueError('Stale/incomplete feature cache; use a new output directory')
        print('Reusing verified Original 107 feature table', flush=True)
        return load_features(path, cohort)
    if not allow_extract:
        raise FileNotFoundError('No features.csv; run --stage extract or --stage all first')
    results = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for number, result in enumerate(pool.map(extract_case, cohort.to_dict('records')), 1):
            results.append(result)
            if number % 25 == 0 or number == len(cohort):
                print(f'Extraction {number}/{len(cohort)}', flush=True)
    failures = [row for row in results if row['error'] is not None]
    (root / 'failed_cases.json').write_text(json.dumps(failures, indent=2))
    if failures:
        raise RuntimeError(f'{len(failures)} extraction failures; no patients silently dropped. See failed_cases.json')
    if signature != extraction_signature(cohort, csv_path):
        raise ValueError('Source cohort or NIfTI files changed during extraction')
    features = pd.DataFrame([row['features'] for row in results])
    combined = pd.concat([cohort.reset_index(drop=True), features], axis=1)
    pd.DataFrame([row['qc'] for row in results]).to_csv(root / 'extraction_qc.csv', index=False)
    combined.to_csv(root / 'features.csv.tmp', index=False)
    (root / 'features.csv.tmp').replace(path)
    manifest.write_text(json.dumps(signature, indent=2))
    return load_features(path, cohort)


class RadiomicsFilter(TransformerMixin, BaseEstimator):
    """Fixed-order greedy Spearman filter, fitted exclusively on training rows."""

    def __init__(self, correlation_threshold=0.90):
        self.correlation_threshold = correlation_threshold

    def fit(self, x, y=None):
        if not isinstance(x, pd.DataFrame) or not np.isfinite(x.to_numpy()).all():
            raise ValueError('Radiomics feature table must be a finite DataFrame')
        varying = [key for key in sorted(x.columns) if x[key].nunique() > 1]
        if not varying:
            raise ValueError('All training features are constant')
        correlation = x[varying].corr(method='spearman').abs()
        self.selected_, audit = [], []
        for key in sorted(x.columns):
            reason, representative = 'retained', ''
            if key not in varying:
                reason = 'constant'
            else:
                redundant = [other for other in self.selected_
                             if correlation.loc[key, other] > self.correlation_threshold]
                if redundant:
                    reason, representative = 'correlated', redundant[0]
                else:
                    self.selected_.append(key)
            audit.append(dict(feature=key, reason=reason, correlated_with=representative))
        self.audit_ = pd.DataFrame(audit)
        return self

    def transform(self, x):
        result = x.loc[:, self.selected_].to_numpy(dtype=np.float64)
        if not np.isfinite(result).all():
            raise ValueError('Model input must be finite')
        return result


def fit_cox(x, time, event, penalty):
    model = Pipeline([('filter', RadiomicsFilter()), ('scale', StandardScaler()),
                      ('cox', CoxPHSurvivalAnalysis(alpha=float(penalty), ties='breslow',
                                                   n_iter=200, tol=1e-9))])
    with warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        model.fit(x, Surv.from_arrays(np.asarray(event).astype(bool), np.asarray(time, dtype=float)))
    if not np.isfinite(model.named_steps['cox'].coef_).all():
        raise ValueError('Nonfinite Cox coefficients')
    return model


def make_inner_splits(event):
    return list(StratifiedKFold(3, shuffle=True, random_state=123).split(np.zeros(len(event)), event))


def select_penalty(x, time, event, penalties=(.01, .1, 1., 10., 100., 1000.)):
    records, candidates = [], []
    for penalty in penalties:
        scores = []
        for inner, (train, val) in enumerate(make_inner_splits(event)):
            record = dict(penalty=float(penalty), inner_fold=inner, cindex=np.nan, error='')
            try:
                model = fit_cox(x.iloc[train], time[train], event[train], penalty)
                score = float(concordance_index_censored(event[val].astype(bool), time[val],
                                                        model.predict(x.iloc[val]))[0])
                if not np.isfinite(score):
                    raise ValueError('Nonfinite inner C-index')
                record.update(cindex=score, n_selected=len(model.named_steps['filter'].selected_))
                scores.append(score)
            except (ValueError, ArithmeticError, ConvergenceWarning, np.linalg.LinAlgError) as exc:
                record['error'] = f'{type(exc).__name__}: {exc}'
            records.append(record)
        if len(scores) == 3:
            candidates.append((float(np.mean(scores)), float(penalty)))
    if not candidates:
        raise RuntimeError(f'No valid inner Cox candidate: {records}')
    return max(candidates)[1], pd.DataFrame(records)


def fit_folds(data, root):
    # Shared evaluator only; no imports from other test experiments or model code.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    import torch
    from loss import cox_loss
    from utils import save_cv_results
    from visualization.survival_plots import analyze_fold

    if root.exists() and any(root.iterdir()):
        raise FileExistsError('Results already exist; choose a new --output_dir to avoid overwriting')
    root.mkdir(parents=True, exist_ok=True)
    rows, oof = [], []
    columns = feature_names()
    for fold in range(5):
        train = data.loc[data.split != fold].reset_index(drop=True)
        val = data.loc[data.split == fold].reset_index(drop=True)
        if set(train.ct_id) & set(val.ct_id):
            raise ValueError('Outer patient overlap')
        print(f'\nFold {fold} | Train {len(train)} ({int(train["dfs.status"].sum())} events)'
              f' | Val {len(val)} ({int(val["dfs.status"].sum())} events)', flush=True)
        penalty, tuning = select_penalty(train[columns], train['dfs.month'].to_numpy(),
                                         train['dfs.status'].to_numpy())
        model = fit_cox(train[columns], train['dfs.month'], train['dfs.status'], penalty)
        target = root / f'fold_{fold}'
        target.mkdir()
        tuning.to_csv(target / 'inner_cv.csv', index=False)
        model.named_steps['filter'].audit_.to_csv(target / 'feature_selection.csv', index=False)
        joblib.dump(model, target / 'cox_model.joblib')
        selected = model.named_steps['filter'].selected_
        pd.DataFrame({'feature': selected, 'standardized_coefficient': model.named_steps['cox'].coef_}
                     ).to_csv(target / 'coefficients.csv', index=False)
        predictions = {}
        for name, frame in [('train', train), ('val', val)]:
            prediction = frame[['case_id', 'pa_id', 'ct_id', 'dfs.month', 'dfs.status', 'split']].copy()
            prediction['risk_score'] = model.predict(frame[columns])
            if not np.isfinite(prediction.risk_score).all():
                raise ValueError('Nonfinite risk predictions')
            predictions[name] = prediction
        predictions['train'].to_csv(target / 'train_predictions.csv', index=False)
        metrics = analyze_fold(predictions['train'], predictions['val'], target)
        for name, prediction in predictions.items():
            metrics[name + '_loss'] = float(cox_loss(*[torch.tensor(prediction[key].to_numpy(), dtype=torch.float64)
                for key in ['risk_score', 'dfs.month', 'dfs.status']]))
        rows.append(dict(fold=fold, ridge=penalty, n_train=len(train), n_features=len(selected), **metrics))
        oof.append(pd.read_csv(target / 'val_predictions.csv', dtype={'case_id': str, 'ct_id': str, 'pa_id': str}))
        print(f'Ridge={penalty:g} | Features={len(selected)}/107 | '
              f'Train C-index={metrics["train_cindex"]:.4f} | Val C-index={metrics["val_cindex"]:.4f} | '
              f'Train loss={metrics["train_loss"]:.4f} | Val loss={metrics["val_loss"]:.4f}', flush=True)
    combined = pd.concat(oof, ignore_index=True)
    if len(combined) != len(data) or combined.ct_id.duplicated().any():
        raise ValueError('Incomplete or repeated outer validation patients')
    combined.to_csv(root / 'oof_predictions.csv', index=False)
    save_cv_results(rows, root)
    for key in ['auc_36m', 'auc_60m', 'gap']:
        values = pd.DataFrame(rows)[key]
        print(f'{key}: {values.mean(skipna=False):.4f} ± {values.std(skipna=False):.4f}', flush=True)


def run(args):
    if args.workers < 1:
        raise ValueError('workers must be positive')
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
    logging.getLogger('radiomics').setLevel(logging.ERROR)
    root = Path(args.output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    cohort = load_cohort(args.cohort_csv, args.ct_root)
    print(f'Experiment 16: CT-only Original Radiomics | {len(cohort)} patients | '
          f'{int(cohort["dfs.status"].sum())} events | five existing folds', flush=True)
    config = dict(cohort_csv=str(Path(args.cohort_csv).resolve()), ct_root=str(Path(args.ct_root).resolve()),
                  output_dir=str(root), extraction=extraction_settings(), image_types=['Original'],
                  n_features=107, feature_names=feature_names(), correlation='absolute Spearman > 0.90',
                  correlation_order='alphabetical greedy; keep the first representative',
                  scaling='training-only Z-score; no patient-vector L2', model='Ridge-Cox, Breslow',
                  ridge_grid=[.01, .1, 1., 10., 100., 1000.], inner_folds=3, seed=123,
                  selection='mean inner C-index; exact ties favor stronger ridge',
                  evaluation='outer validation once; AUC36/60 IPCW; KM cutoff is training risk median',
                  intensity_caveat='Doctor-provided NIfTI; upstream processing and original HU calibration unknown. '
                                   'binWidth is 25 input units; no additional clipping or normalization.',
                  stability_caveat='No repeat-scan/reader ICC filtering without repeated measurements',
                  scope='CT-only comparison baseline; no pathology features, fusion or neural checkpoint use')
    config_path = root / 'run_config.yaml'
    if config_path.exists() and yaml.safe_load(config_path.read_text()) != config:
        raise ValueError('Existing run configuration differs; use a new output directory')
    config_path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True))
    data = prepare_features(cohort, args.cohort_csv, root, args.workers, args.stage != 'fit')
    if args.stage != 'extract':
        fit_folds(data, root / 'results')
