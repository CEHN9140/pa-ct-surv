from itertools import islice
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import BatchSampler, DataLoader, Dataset, RandomSampler, Subset


def resolve_data_dir(data_dir, roi_size):
    directory = Path(data_dir) if data_dir else Path(__file__).resolve().parent / "data/trident/features_uni_v1"
    directory = directory.resolve()
    label_file = directory / f"all_label_roi{roi_size}.csv"
    if not label_file.is_file():
        raise FileNotFoundError(f"Dataset CSV not found: {label_file}")
    return str(directory)


def cv_splits(samples):
    """Read the saved CV assignment; training never regenerates it."""
    development, test = locked_split_indices(samples)
    print(f"CV cohort: {len(development)} | Held-out test: {len(test)}")
    print("Test set is not accessed during CV" if len(test)
          else "5-fold CV uses all samples; no held-out test set")
    return [cv_fold_indices(samples, fold) for fold in range(5)]


def load_path_feature(feature_path):
    """Read native CLAM PT or TRIDENT H5 as CPU float32 features [N, 1024]."""
    path = Path(feature_path)
    if path.suffix.lower() == '.pt':
        features = torch.load(path, map_location='cpu', weights_only=True)
        if not isinstance(features, torch.Tensor):
            raise ValueError(f'Expected a feature tensor in {path}')
        features = features.float()
    elif path.suffix.lower() == '.h5':
        # Open per sample; do not share HDF5 handles between loader workers.
        with h5py.File(path, 'r') as file:
            if 'features' not in file:
                raise ValueError(f"Missing 'features' dataset in {path}")
            features = torch.from_numpy(np.asarray(file['features'], dtype=np.float32))
    else:
        raise ValueError(f'Expected pathology .pt or .h5 features: {path}')
    if features.ndim != 2 or features.shape[0] == 0 or features.shape[1] != 1024:
        raise ValueError(f'Expected nonempty features [N, 1024], got {tuple(features.shape)}: {path}')
    if not torch.isfinite(features).all():
        raise ValueError(f'Non-finite pathology features: {path}')
    return features.contiguous()


def load_ct_npy(ct_path, expected_roi_size=None):
    ct = np.load(ct_path).astype(np.float32)
    if ct.ndim == 3:
        ct = ct[np.newaxis, ...]
    if ct.ndim != 4 or ct.shape[0] != 1:
        raise ValueError(f'Invalid CT shape {ct.shape} for {ct_path}')
    if expected_roi_size is not None:
        expected_shape = (1, expected_roi_size, expected_roi_size, expected_roi_size)
        if tuple(ct.shape) != expected_shape:
            raise ValueError(f'Invalid CT shape {ct.shape}; expected {expected_shape} for {ct_path}')
    if not np.isfinite(ct).all():
        raise ValueError(f'CT values must be finite for {ct_path}')
    if ct.min() < 0 or ct.max() > 1:
        raise ValueError(f'CT values must be in [0, 1] for {ct_path}')
    return np.ascontiguousarray(ct, dtype=np.float32)


def read_samples(data_dir, roi_size, columns):
    filename = 'all_label.csv' if roi_size is None else f'all_label_roi{roi_size}.csv'
    label_file = Path(data_dir) / filename
    samples = pd.read_csv(label_file, dtype={'pa_id': str, 'ct_id': str})
    if 'h5_path' in samples:
        if 'coord_path' not in samples:
            samples.rename(columns={'h5_path': 'coord_path'}, inplace=True)
        else:
            samples.drop(columns='h5_path', inplace=True)
    missing = set(columns) - set(samples.columns)
    if missing:
        raise ValueError(f'Missing columns in {label_file}: {sorted(missing)}')
    if samples.empty:
        raise ValueError(f'No samples in {label_file}')
    for column in columns:
        if samples[column].isna().any():
            raise ValueError(f'Missing {column} values in {label_file}')
        if column.endswith('_id') or column.endswith('_path'):
            samples[column] = samples[column].astype(str).str.strip()
            if samples[column].eq('').any():
                raise ValueError(f'Empty {column} values in {label_file}')
        if column.endswith('_path'):
            missing_paths = samples.loc[~samples[column].map(lambda p: Path(p).is_file()), column]
            if len(missing_paths):
                raise FileNotFoundError(f'Missing {len(missing_paths)} {column} files: {missing_paths.head(10).tolist()}')
    events = pd.to_numeric(samples['event'], errors='coerce')
    times = pd.to_numeric(samples['time'], errors='coerce')
    if not events.isin([0, 1]).all():
        raise ValueError(f'Events must be 0 or 1 in {label_file}')
    if not np.isfinite(times).all() or (times < 0).any():
        raise ValueError(f'Survival times must be finite and nonnegative in {label_file}')
    samples['event'], samples['time'] = events.astype(int), times.astype(float)
    return samples


class Path_Dataset(Dataset):
    def __init__(self, data_dir, roi_size=64):
        self.samples = read_samples(data_dir, roi_size, ['pa_id', 'pa_path', 'event', 'time'])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        row = self.samples.iloc[idx]
        return (load_path_feature(row['pa_path']),
                torch.tensor(row['event'], dtype=torch.long),
                torch.tensor(row['time'], dtype=torch.float32), row['pa_id'])


class CT_Dataset(Dataset):
    def __init__(self, data_dir, roi_size=64, augment=False):
        if augment:
            raise ValueError('CT augmentation is currently disabled')
        self.roi_size = roi_size
        self.samples = read_samples(data_dir, roi_size, ['ct_id', 'ct_path', 'event', 'time'])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        row = self.samples.iloc[idx]
        return (torch.from_numpy(load_ct_npy(row['ct_path'], self.roi_size)),
                torch.tensor(row['event'], dtype=torch.long),
                torch.tensor(row['time'], dtype=torch.float32), row['ct_id'])


class Pa_CT_Dataset(Dataset):
    def __init__(self, data_dir, roi_size=64, augment=False):
        if augment:
            raise ValueError('CT augmentation is currently disabled')
        self.roi_size = roi_size
        self.samples = read_samples(data_dir, roi_size,
                                    ['pa_id', 'pa_path', 'ct_id', 'ct_path', 'event', 'time'])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        row = self.samples.iloc[idx]
        return (torch.from_numpy(load_ct_npy(row['ct_path'], self.roi_size)),
                load_path_feature(row['pa_path']),
                torch.tensor(row['event'], dtype=torch.long),
                torch.tensor(row['time'], dtype=torch.float32),
                f"{row['pa_id']}|{row['ct_id']}")


class CT_Student_Dataset(Pa_CT_Dataset):
    @staticmethod
    def collate(batch):
        """Stack CT volumes and keep variable-length pathology bags as a list."""
        cts, pas, events, times, case_ids = zip(*batch)
        return (torch.stack(cts), list(pas), torch.stack(events).view(-1),
                torch.stack(times).view(-1), list(case_ids))


def build_dataloader(dataset, indices=None, batch_size=1, shuffle=False,
                     num_workers=0, merge_tail=False, pin_memory=True):
    """Shared loader creation for training, validation and attention statistics."""
    if indices is not None:
        dataset = Subset(dataset, indices)
    source = dataset
    while isinstance(source, Subset):
        source = source.dataset
    collate = source.collate if isinstance(source, CT_Student_Dataset) else None
    options = dict(num_workers=num_workers, pin_memory=pin_memory, collate_fn=collate)
    if merge_tail:
        if not shuffle:
            raise ValueError('Tail merging is only used for shuffled training loaders')
        return DataLoader(dataset, batch_sampler=CoxBatchSampler(RandomSampler(dataset), batch_size), **options)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, **options)


def cox_batch_sizes(num_samples, batch_size):
    """Merge an incomplete final risk set into the preceding full batch."""
    if num_samples < 0 or batch_size < 1:
        raise ValueError("num_samples must be nonnegative and batch_size positive")
    full, tail = divmod(num_samples, batch_size)
    sizes = [batch_size] * full
    if tail:
        if sizes:
            sizes[-1] += tail
        else:
            sizes.append(tail)
    return sizes


class CoxBatchSampler(BatchSampler):
    """Preserve sampler order and use every draw once, merging the last tail."""

    def __init__(self, sampler, batch_size):
        super().__init__(sampler, batch_size, drop_last=False)

    def __iter__(self):
        indices = iter(self.sampler)
        for size in cox_batch_sizes(len(self.sampler), self.batch_size):
            batch = list(islice(indices, size))
            if len(batch) != size:
                raise ValueError("Sampler yielded fewer indices than its declared length")
            yield batch
        # Finish the sampler iterator too: RandomSampler performs final RNG work
        # when exhausted, which must match ordinary DataLoader shuffling.
        if next(indices, None) is not None:
            raise ValueError("Sampler yielded more indices than its declared length")

    def __len__(self):
        return len(cox_batch_sizes(len(self.sampler), self.batch_size))


def locked_split_indices(samples):
    if "split" not in samples.columns:
        raise ValueError("Dataset CSV missing split column; run prepare_data.py")
    raw_split = samples["split"]
    legacy_values = set(raw_split.astype(str))
    if legacy_values.issubset({"train", "test"}):
        train_mask = raw_split.astype(str).to_numpy() == "train"
        test_mask = raw_split.astype(str).to_numpy() == "test"
        if not train_mask.any() or not test_mask.any():
            raise ValueError("Dataset split must contain both train and test samples")
        return np.flatnonzero(train_mask), np.flatnonzero(test_mask)

    split = pd.to_numeric(raw_split, errors="coerce").to_numpy()
    valid = np.isfinite(split) & np.isin(split, [-1, 0, 1, 2, 3, 4])
    if not valid.all():
        raise ValueError("Dataset split values must be integers -1 or 0..4")
    train_mask = split >= 0
    test_mask = split == -1
    if not train_mask.any():
        raise ValueError("Dataset split must contain CV samples (0..4)")
    return np.flatnonzero(train_mask), np.flatnonzero(test_mask)


def cv_fold_indices(samples, fold, n_splits=5):
    """Return (train, validation) row indices for a preassigned CV fold."""
    if not isinstance(fold, (int, np.integer)) or not 0 <= int(fold) < n_splits:
        raise ValueError(f"fold must be an integer in [0, {n_splits - 1}]")
    train_indices, _ = locked_split_indices(samples)
    split = pd.to_numeric(samples["split"], errors="coerce").to_numpy()
    validation_mask = split == int(fold)
    fold_train_mask = (split >= 0) & (split != int(fold))
    validation_indices = np.flatnonzero(validation_mask)
    fold_train_indices = np.flatnonzero(fold_train_mask)
    if not len(validation_indices) or not len(fold_train_indices):
        raise ValueError(f"CV fold {fold} must have non-empty train and validation sets")
    if len(fold_train_indices) + len(validation_indices) != len(train_indices):
        raise ValueError("CV folds must cover exactly the development set")
    return fold_train_indices, validation_indices
