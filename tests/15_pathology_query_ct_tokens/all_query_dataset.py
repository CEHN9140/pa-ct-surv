"""Full Virchow2 query bags, with the exact original Top25 in the first rows."""

import hashlib
from pathlib import Path

import h5py
import numpy as np
import torch

from dataset import Pa_CT_Dataset, load_ct_npy, load_path_feature


class AllPatchQueryDataset(Pa_CT_Dataset):
    def __init__(self, data_dir, all_feature_dir, roi_size=64):
        super().__init__(data_dir, roi_size=roi_size, feature_dim=1280)
        self.all_feature_dir = Path(all_feature_dir).resolve()
        if not self.all_feature_dir.is_dir():
            raise FileNotFoundError(self.all_feature_dir)
        if self.samples.pa_id.duplicated().any():
            raise ValueError('Expected one WSI per patient in experiment 15')
        self.full_paths, self.query_orders, counts = [], [], []
        for i, row in enumerate(self.samples.itertuples(index=False)):
            path = self.all_feature_dir / f'{row.pa_id}.h5'
            with h5py.File(row.pa_path, 'r') as top, h5py.File(path, 'r') as full:
                if not {'features', 'coords', 'selected_indices'} <= set(top):
                    raise ValueError(f'Missing Top25 features/coordinates/indices: {row.pa_id}')
                if not {'features', 'coords'} <= set(full):
                    raise ValueError(f'Missing full features/coordinates: {row.pa_id}')
                n = full['features'].shape[0]
                if top['features'].shape != (25, 1280) or full['features'].shape != (n, 1280) or n < 25:
                    raise ValueError(f'Expected Top25/full Virchow2 CLS [N,1280]: {row.pa_id}')
                if full['coords'].shape != (n, 2) or top['coords'].shape != (25, 2):
                    raise ValueError(f'Invalid coordinate shapes: {row.pa_id}')
                indices = top['selected_indices'][:]
                if (indices.shape != (25,) or not np.issubdtype(indices.dtype, np.integer)
                        or len(np.unique(indices)) != 25 or (indices < 0).any() or (indices >= n).any()):
                    raise ValueError(f'Invalid selected_indices: {row.pa_id}')
                for key in ['downstream_encoder', 'source_stamp', 'input_signature']:
                    if top.attrs.get(key) != full.attrs.get(key):
                        raise ValueError(f'Top25/full {key} mismatch: {row.pa_id}')
                features = np.asarray(full['features'], dtype=np.float32)
                if not np.isfinite(features).all():
                    raise ValueError(f'Nonfinite full features: {row.pa_id}')
                if not np.array_equal(features[indices], top['features'][:]):
                    raise ValueError(f'Top25 features mismatch: {row.pa_id}')
                if not np.array_equal(full['coords'][:][indices], top['coords'][:]):
                    raise ValueError(f'Top25 coordinates mismatch: {row.pa_id}')
                remaining = np.ones(n, dtype=bool)
                remaining[indices] = False
                self.query_orders.append(np.concatenate([indices, np.flatnonzero(remaining)]))
                self.full_paths.append(path)
                counts.append(n)
            if (i + 1) % 100 == 0 or i + 1 == len(self.samples):
                print(f'All-query input audit: {i + 1}/{len(self.samples)} exact Top25/full mappings', flush=True)
        self.audit = dict(patients=len(counts), total_query_patches=int(sum(counts)),
                          min_query_patches=int(min(counts)), max_query_patches=int(max(counts)),
                          median_query_patches=float(np.median(counts)),
                          mean_query_patches=float(np.mean(counts)), top25_feature_coordinate_checks='exact',
                          cohort_sha256=hashlib.sha256(
                              (Path(data_dir) / f'all_label_roi{roi_size}.csv').read_bytes()).hexdigest(),
                          query_order='original Top25 ranking first; remaining source rows once, in source order')

    def __getitem__(self, idx):
        row = self.samples.iloc[idx]
        features = load_path_feature(self.full_paths[idx], 1280)[self.query_orders[idx]]
        return (torch.from_numpy(load_ct_npy(row.ct_path, self.roi_size)), features,
                torch.tensor(row.event, dtype=torch.long), torch.tensor(row.time, dtype=torch.float32),
                f'{row.pa_id}|{row.ct_id}')
