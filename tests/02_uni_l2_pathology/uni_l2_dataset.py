"""Normalize each raw UNI patch independently without changing stored features."""

from torch.nn.functional import normalize

from dataset import Path_Dataset


class UniL2Dataset(Path_Dataset):
    def __getitem__(self, index):
        features, event, time, case_id = super().__getitem__(index)
        return normalize(features, p=2, dim=-1, eps=1e-12), event, time, case_id
