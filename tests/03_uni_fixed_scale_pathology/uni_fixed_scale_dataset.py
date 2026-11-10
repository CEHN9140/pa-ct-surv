"""Scale raw UNI features by one fixed constant, preserving norm differences."""

from dataset import Path_Dataset


class UniFixedScaleDataset(Path_Dataset):
    def __getitem__(self, index):
        features, event, time, case_id = super().__getitem__(index)
        return features / 40.0, event, time, case_id
