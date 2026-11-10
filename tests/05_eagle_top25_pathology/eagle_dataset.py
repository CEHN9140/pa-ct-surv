"""Fixed-size EAGLE bags for ordinary patient batching in experiment 05."""

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from dataset import read_samples


class EagleDataset(Dataset):
    def __init__(self, data_dir, roi_size=64):
        self.samples = read_samples(data_dir, roi_size, ['pa_id', 'pa_path', 'event', 'time'])
        self.feature_dim = 1280
        self.patch_count = 25

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        row = self.samples.iloc[index]
        with h5py.File(row['pa_path'], 'r') as handle:
            if 'features' not in handle:
                raise ValueError(f"Missing features in {row['pa_path']}")
            features = np.asarray(handle['features'], dtype=np.float32)
        if features.shape != (self.patch_count, self.feature_dim):
            raise ValueError(f"Expected features [25, 1280], got {features.shape}: {row['pa_path']}")
        if not np.isfinite(features).all():
            raise ValueError(f"Non-finite features: {row['pa_path']}")
        return (torch.from_numpy(np.ascontiguousarray(features)),
                torch.tensor(row['event'], dtype=torch.long),
                torch.tensor(row['time'], dtype=torch.float32), row['pa_id'])
