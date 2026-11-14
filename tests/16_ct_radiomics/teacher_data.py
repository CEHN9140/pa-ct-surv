"""Paired Top25/Radiomics data and FIT-only preprocessing for experiment16."""

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import joblib
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

from dataset import load_path_feature
from radiomics_ct import (RadiomicsFilter, feature_names, load_cohort, load_features,
                          extraction_signature)


def radiomics_groups():
    return ['shape', 'firstorder', 'glcm', 'glrlm', 'glszm', 'gldm', 'ngtdm']


class RadiomicsPreprocessor:
    def fit(self, frame, fit_ids):
        if len(frame) != len(fit_ids) or len(set(fit_ids)) != len(fit_ids):
            raise ValueError('Preprocessor needs unique FIT identities for every training row')
        self.input_features_ = list(frame.columns)
        self.selector = RadiomicsFilter().fit(frame)
        self.selected_ = list(self.selector.selected_)
        self.scaler = StandardScaler().fit(self.selector.transform(frame))
        self.group_indices = [[i for i, name in enumerate(self.selected_) if name.split('_')[1] == group]
                              for group in radiomics_groups()]
        if sorted(i for group in self.group_indices for i in group) != list(range(len(self.selected_))):
            raise ValueError('Unknown or ambiguous Radiomics feature family')
        if sum(bool(group) for group in self.group_indices) < 2:
            raise ValueError('Cross-attention requires at least two nonempty CT feature groups')
        self.fit_n_ = len(frame)
        self.fit_ids_sha256_ = hashlib.sha256('\n'.join(map(str, fit_ids)).encode()).hexdigest()
        return self

    def transform(self, frame):
        values = self.scaler.transform(self.selector.transform(frame[self.input_features_]))
        if not np.isfinite(values).all():
            raise ValueError('Nonfinite standardized Radiomics')
        return np.asarray(values, dtype=np.float32)

    def metadata(self):
        return dict(fit_n=self.fit_n_, fit_ids_sha256=self.fit_ids_sha256_,
                    selected_features=self.selected_, group_names=radiomics_groups(),
                    group_indices=self.group_indices,
                    nonempty_groups=[bool(group) for group in self.group_indices],
                    mean=self.scaler.mean_.tolist(), scale=self.scaler.scale_.tolist(),
                    filtering='FIT-only constant removal and greedy abs Spearman >0.90',
                    scaling='FIT-only column Z-score; no CT vector L2')


def join_teacher_tables(samples, features):
    required = ['pa_id', 'ct_id', 'pa_path', 'time', 'event', 'split']
    if not set(required).issubset(samples) or samples[required].isna().any().any():
        raise ValueError('Missing paired patient identity/labels/pathology paths')
    if (samples.ct_id.duplicated().any() or samples.pa_id.duplicated().any()
            or features.ct_id.duplicated().any() or set(samples.ct_id) != set(features.ct_id)):
        raise ValueError('Paired cohort identity differs or patients are duplicated')
    aligned = features.set_index('ct_id').loc[samples.ct_id].reset_index()
    for cached, actual in [('pa_id', 'pa_id'), ('dfs.month', 'time'),
                            ('dfs.status', 'event'), ('split', 'split')]:
        if not np.array_equal(aligned[cached].to_numpy(), samples[actual].to_numpy()):
            raise ValueError(f'Radiomics/pathology identity or labels differ: {cached}')
    if any(Path(path).stem != identity for path, identity in zip(samples.pa_path, samples.pa_id)):
        raise ValueError('Pathology feature filename does not match PA identity')
    return aligned


class PaRadiomicsDataset(Dataset):
    def __init__(self, samples, pathology, radiomics, values=None):
        if len(samples) != len(pathology) or len(samples) != len(radiomics):
            raise ValueError('Pathology and Radiomics patient counts differ')
        if pathology.shape[1:] != (25, 1280) or not torch.isfinite(pathology).all():
            raise ValueError('Require finite EAGLE Top25 features [N,25,1280]')
        self.samples = samples.reset_index(drop=True)
        self.pathology = pathology
        self.radiomics = radiomics.reset_index(drop=True)
        self.values = None if values is None else torch.as_tensor(values, dtype=torch.float32)

    def with_preprocessor(self, preprocessor):
        return PaRadiomicsDataset(self.samples, self.pathology, self.radiomics,
                                  preprocessor.transform(self.radiomics))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        if self.values is None:
            raise ValueError('Fit a training-only preprocessor before building a loader')
        row = self.samples.iloc[index]
        return (self.values[index], self.pathology[index], torch.tensor(row.event, dtype=torch.long),
                torch.tensor(row.time, dtype=torch.float32), str(row.case_id))


def teacher_partitions(samples, outer_train, seed=123):
    indices = np.asarray(outer_train, dtype=int)
    if len(set(indices)) != len(indices):
        raise ValueError('Repeated outer training patients')
    splitter = StratifiedKFold(3, shuffle=True, random_state=seed)
    result = []
    for remaining, held in splitter.split(indices, samples.iloc[indices].event):
        remaining, held = indices[remaining], indices[held]
        fit, stop = train_test_split(remaining, test_size=.15, random_state=seed,
                                    stratify=samples.iloc[remaining].event)
        result.append((fit, stop, held))
    return result


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load_teacher_inputs(cohort_csv, features_csv):
    cohort_csv, features_csv = Path(cohort_csv), Path(features_csv)
    source_config = yaml.safe_load((features_csv.parent / 'run_config.yaml').read_text())
    manifest = json.loads((features_csv.parent / 'extraction_manifest.json').read_text())
    if file_digest(cohort_csv) != manifest['cohort_sha256']:
        raise ValueError('Cohort differs from the complete CT extraction source')
    cohort = load_cohort(cohort_csv, source_config['ct_root'])
    if manifest != extraction_signature(cohort, cohort_csv):
        raise ValueError('Radiomics extraction settings or source CT/Mask metadata changed')
    cached = load_features(features_csv, cohort)
    samples = pd.read_csv(cohort_csv, dtype={'pa_id': str, 'ct_id': str})
    aligned = join_teacher_tables(samples, cached)
    samples['case_id'] = samples.pa_id + '|' + samples.ct_id
    pathology, sources = [], {}
    for row in samples.itertuples():
        features = load_path_feature(row.pa_path, 1280)
        if features.shape != (25, 1280):
            raise ValueError(f'Expected exactly25 EAGLE patches: {row.pa_id}')
        pathology.append(features)
        sources[row.pa_path] = file_digest(row.pa_path)
    dataset = PaRadiomicsDataset(samples, torch.stack(pathology), aligned[feature_names()])
    sources.update({str(path.resolve()): file_digest(path) for path in
                    [cohort_csv, features_csv, features_csv.parent/'run_config.yaml',
                     features_csv.parent/'extraction_manifest.json']})
    print(f'Paired teacher inputs verified: {len(dataset)} patients, {int(samples.event.sum())} events; '
          'Top25 [25,1280], full Original107; no CT image or checkpoint loading.', flush=True)
    return dataset, sources


def save_preprocessor(preprocessor, directory):
    directory = Path(directory)
    joblib.dump(preprocessor, directory / 'preprocessor.joblib')
    (directory / 'preprocessing.json').write_text(json.dumps(preprocessor.metadata(), indent=2))
    preprocessor.selector.audit_.to_csv(directory / 'feature_selection.csv', index=False)
