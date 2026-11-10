"""KEEP feature loading isolated from the production 1024-D datasets."""

from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from dataset import read_samples


class KeepDataset(Dataset):
    def __init__(self, data_dir, feature_dir, roi_size=64):
        self.samples = read_samples(data_dir, roi_size, ['pa_id', 'event', 'time'])
        if self.samples.pa_id.duplicated().any():
            raise ValueError('Reference CSV contains duplicate pathology IDs')
        feature_dir = Path(feature_dir).resolve()
        self.samples['pa_path'] = self.samples.pa_id.map(
            lambda slide_id: str(feature_dir / f'{slide_id}.h5')
        )
        self.feature_dim = None
        for row in self.samples.itertuples(index=False):
            with h5py.File(row.pa_path, 'r') as file:
                shape = file['features'].shape
                if len(shape) != 2 or min(shape) == 0:
                    raise ValueError(f'Invalid KEEP features {shape}: {row.pa_path}')
                if self.feature_dim is None:
                    self.feature_dim = shape[1]
                if shape[1] != self.feature_dim:
                    raise ValueError(f'Inconsistent KEEP feature dimension: {row.pa_path}')
                if file['coords'].shape != (shape[0], 2):
                    raise ValueError(f'KEEP patch/coordinate counts differ: {row.pa_id}')
                if 'coord_path' in self.samples.columns:
                    with h5py.File(row.coord_path, 'r') as coordinates:
                        if not np.array_equal(file['coords'][:], coordinates['coords'][:]):
                            raise ValueError(f'KEEP/reference coordinates differ: {row.pa_id}')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        row = self.samples.iloc[index]
        with h5py.File(row.pa_path, 'r') as file:
            features = np.asarray(file['features'], dtype=np.float32)
        if not np.isfinite(features).all():
            raise ValueError(f'Non-finite KEEP features: {row.pa_path}')
        return (torch.from_numpy(features),
                torch.tensor(row.event, dtype=torch.long),
                torch.tensor(row.time, dtype=torch.float32), row.pa_id)
