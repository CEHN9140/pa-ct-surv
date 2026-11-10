"""GL-only cohort preparation and variable-length TRIDENT feature loading."""
import hashlib
import json
from pathlib import Path
import sys

import h5py
import numpy as np
import pandas as pd
import torch
from torch.nn.functional import normalize
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from dataset import read_samples


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def feature_signature(paths):
    records = [(str(Path(p).resolve()), Path(p).stat().st_size,
                Path(p).stat().st_mtime_ns) for p in paths]
    return hashlib.sha256(json.dumps(records).encode()).hexdigest()


def validate_cohort(data_dir):
    directory = Path(data_dir)
    metadata = json.loads((directory/'preparation.json').read_text())
    if file_hash(metadata['labels_csv']) != metadata['labels_sha256']:
        raise ValueError('Original label source changed after cohort preparation')
    if file_hash(directory/'all_label.csv') != metadata['cohort_sha256']:
        raise ValueError('Prepared cohort or saved folds changed after preparation')
    if ('reference_cohort_csv' in metadata and
            file_hash(metadata['reference_cohort_csv']) != metadata['reference_cohort_sha256']):
        raise ValueError('Frozen reference cohort or folds changed after KEEP preparation')
    return metadata


def add_event_time_splits(samples):
    """Shuffle within event/time strata; rotate fold slots to balance all counts."""
    result = samples.copy().reset_index(drop=True)
    counts = result.event.value_counts().reindex([0, 1], fill_value=0)
    if not result.event.isin([0, 1]).all() or (counts < 5).any():
        raise ValueError('Five-fold CV requires at least five patients in each event group')
    result['time_group'], result['split'] = -1, -1
    cutpoints, offset = {}, 0
    for event in [0, 1]:
        indices = result.index[result.event == event].to_numpy()
        cuts = np.unique(np.quantile(result.loc[indices, 'time'], [.25, .5, .75]))
        cutpoints[str(event)] = cuts.tolist()
        groups = np.searchsorted(cuts, result.loc[indices, 'time'], side='left')
        result.loc[indices, 'time_group'] = groups
        for group in np.unique(groups):
            shuffled = np.random.permutation(indices[groups == group])
            result.loc[shuffled, 'split'] = (np.arange(len(shuffled))+offset) % 5
            offset += len(shuffled)
    return result, cutpoints


def prepare_cohort(labels_csv, feature_dir, output_dir):
    labels_csv, feature_dir, output_dir = map(lambda p: Path(p).resolve(),
                                             [labels_csv, feature_dir, output_dir])
    destination = output_dir/'all_label.csv'
    if destination.is_file():
        metadata = validate_cohort(output_dir)
        if metadata['labels_csv'] != str(labels_csv) or metadata['feature_dir'] != str(feature_dir):
            raise ValueError('Existing prepared cohort uses a different source or feature directory')
        dataset = GLDataset(output_dir)
        print(f'Reusing existing saved five-fold assignments: {destination}', flush=True)
        return dataset.samples
    source = pd.read_csv(labels_csv, dtype={'patient_id': str}, encoding='utf-8-sig')
    if not {'patient_id', 'status', 'survival_time'}.issubset(source):
        raise ValueError('Label source requires patient_id, status, survival_time')
    identifiers = source.patient_id
    if (identifiers.isna().any() or identifiers.str.strip().eq('').any()
            or identifiers.duplicated().any() or not identifiers.map(lambda v: Path(v).name == v).all()):
        raise ValueError('Source patient IDs must be unique, nonempty file stems')
    audit = source.copy()
    audit['source_row'] = np.arange(len(source))+2
    audit['pa_id'] = identifiers
    audit['pa_path'] = identifiers.map(lambda v: str(feature_dir/f'{v}.h5'))
    status = pd.to_numeric(source.status, errors='coerce')
    audit['reason'] = np.where(status.isin([0, 1]), 'included', 'status_not_0_or_1')
    missing = status.isin([0, 1]) & ~audit.pa_path.map(lambda p: Path(p).is_file())
    audit.loc[missing, 'reason'] = 'missing_feature'
    samples = audit.loc[audit.reason == 'included'].drop(columns='reason').copy()
    if samples.empty:
        raise ValueError('No binary-status patients with matching features')
    samples['event'] = status.loc[samples.index].astype(int)
    samples['time'] = pd.to_numeric(samples.survival_time, errors='coerce')
    if not np.isfinite(samples.time).all() or (samples.time < 0).any():
        raise ValueError('Included source survival_time must be finite and nonnegative')
    shapes = []
    for row in samples.itertuples(index=False):
        with h5py.File(row.pa_path, 'r') as f:
            if 'features' not in f:
                raise ValueError(f'Missing features dataset: {row.pa_path}')
            shape = f['features'].shape
        if len(shape) != 2 or min(shape) < 1:
            raise ValueError(f'Invalid feature shape {shape}: {row.pa_path}')
        shapes.append(shape)
    if len({shape[1] for shape in shapes}) != 1:
        raise ValueError('Inconsistent feature dimensions across patients')
    samples['n_patches'] = [shape[0] for shape in shapes]
    samples['feature_dim'] = shapes[0][1]
    samples, cutpoints = add_event_time_splits(samples)
    output_dir.mkdir(parents=True, exist_ok=True)
    samples.to_csv(destination, index=False)
    audit.to_csv(output_dir/'data_audit.csv', index=False)
    audit.loc[audit.reason != 'included'].to_csv(output_dir/'excluded_cases.csv', index=False)
    metadata = dict(labels_csv=str(labels_csv), labels_sha256=file_hash(labels_csv),
                    feature_dir=str(feature_dir), feature_encoder=feature_dir.name.removeprefix('features_'),
                    feature_dim=shapes[0][1], feature_signature=feature_signature(samples.pa_path),
                    cohort_sha256=file_hash(destination), n_source=len(source), n_included=len(samples),
                    n_events=int(samples.event.sum()), time_unit='original_unconfirmed',
                    time_conversion='none', event_mapping='status 1 -> event; status 0 -> censoring',
                    cohort_tag=f'gl{len(samples)}_eventtime',
                    stratification_time_cutpoints=cutpoints,
                    split_method='Unseeded shuffle within event/time-quartile strata; rotating fold slots; saved once')
    (output_dir/'preparation.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2)+'\n')
    distributions = []
    for fold, group in samples.groupby('split'):
        row = dict(fold=int(fold), n=len(group), events=int(group.event.sum()),
                   event_rate=float(group.event.mean()))
        for event, name in [(1, 'event'), (0, 'censor')]:
            times = group.loc[group.event == event, 'time']
            for label, value in zip(['q25', 'median', 'q75'], times.quantile([.25, .5, .75])):
                row[f'{name}_time_{label}'] = float(value)
        distributions.append(row)
    pd.DataFrame(distributions).to_csv(output_dir/'fold_distributions.csv', index=False)
    samples.groupby(['event', 'time_group', 'split']).size().unstack('split', fill_value=0).to_csv(
        output_dir/'time_stratum_counts.csv')
    if 'stage' in samples:
        pd.crosstab(samples.split, samples.stage, dropna=False).to_csv(output_dir/'stage_counts.csv')
    print(f'Prepared {len(samples)}/{len(source)} patients | Events: {int(samples.event.sum())} | '
          f'Feature dimension: {shapes[0][1]} | Original time units retained', flush=True)
    print(samples.groupby('split').agg(n=('pa_id', 'size'), events=('event', 'sum')).to_string(), flush=True)
    print(f'Excluded reasons: {audit.loc[audit.reason != "included", "reason"].value_counts().to_dict()}', flush=True)
    return samples


class GLDataset(Dataset):
    def __init__(self, data_dir, l2_normalize=False):
        self.l2_normalize = l2_normalize
        self.metadata = validate_cohort(data_dir)
        self.samples = read_samples(data_dir, None,
                                    ['pa_id', 'pa_path', 'event', 'time', 'split', 'n_patches', 'feature_dim'])
        if self.samples.pa_id.duplicated().any() or set(self.samples.split) != set(range(5)):
            raise ValueError('Patients must be unique and have all five saved folds')
        dimensions = self.samples.feature_dim.unique()
        if len(dimensions) != 1 or dimensions[0] != self.metadata['feature_dim']:
            raise ValueError('Saved feature dimension differs from cohort metadata')
        self.feature_dim = int(dimensions[0])
        for row in self.samples.itertuples(index=False):
            with h5py.File(row.pa_path, 'r') as f:
                if 'features' not in f or f['features'].shape != (row.n_patches, self.feature_dim):
                    raise ValueError(f'Feature shape/dimension changed: {row.pa_path}')
        if feature_signature(self.samples.pa_path) != self.metadata['feature_signature']:
            raise ValueError('Feature files changed after cohort preparation')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        row = self.samples.iloc[index]
        with h5py.File(row.pa_path, 'r') as f:
            features = np.asarray(f['features'], dtype=np.float32)
        if not np.isfinite(features).all():
            raise ValueError(f'Non-finite pathology features: {row.pa_path}')
        features = torch.from_numpy(features)
        if self.l2_normalize:
            features = normalize(features, p=2, dim=-1, eps=1e-12)
        return (features, torch.tensor(row.event, dtype=torch.long),
                torch.tensor(row.time, dtype=torch.float32), row.pa_id)
