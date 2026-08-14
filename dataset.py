import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from monai.transforms import (
    Compose,
    RandFlip,
    RandRotate,
    RandZoom,
    RandGaussianNoise,
    RandScaleIntensity,
    RandShiftIntensity,
    Lambda,
)


def load_ct_npy(ct_path):
    ct = np.load(ct_path).astype(np.float32)
    if ct.ndim == 3:
        ct = ct[np.newaxis, ...]
    if ct.shape[0] != 1:
        raise ValueError(f"Invalid CT shape {ct.shape} for {ct_path}")
    return np.ascontiguousarray(ct, dtype=np.float32)


def _clip01(x):
    return torch.clamp(x, 0, 1) if torch.is_tensor(x) else np.clip(x, 0, 1).astype(np.float32)


def _build_ct_augmentation():
    return Compose([
        RandFlip(prob=0.5, spatial_axis=2),
        RandRotate(prob=0.3, range_x=0.1, range_y=0.1, range_z=0.1),
        RandZoom(prob=0.3),
        RandGaussianNoise(prob=0.2, std=0.01),
        RandScaleIntensity(prob=0.2, factors=0.1),
        RandShiftIntensity(prob=0.2, offsets=0.05),
        Lambda(_clip01),
    ])


def _get_label_file(data_dir, roi_size=None):
    if roi_size is not None:
        fname = f"all_label_roi{roi_size}.csv"
    else:
        fname = "all_label.csv"
    return os.path.join(os.fspath(data_dir), fname)


class Path_Dataset(Dataset):
    """Pathology patch feature dataset for survival analysis."""
    def __init__(self, data_dir, roi_size=64):
        label_file = _get_label_file(data_dir, roi_size=roi_size)
        df = pd.read_csv(label_file)
        cols = ["pa_id", "pa_path", "event", "time"]
        missing = set(cols) - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns in {label_file}: {sorted(missing)}")
        optional_cols = [c for c in ["ct_id", "h5_path", "split"] if c in df.columns]
        self.samples = df[cols + optional_cols].copy()
        self.samples["pa_id"] = self.samples["pa_id"].astype(str)
        self.samples["pa_path"] = self.samples["pa_path"].astype(str)
        self.samples["event"] = self.samples["event"].astype(int)
        self.samples["time"] = self.samples["time"].astype(float)
        before = len(self.samples)
        self.samples = self.samples[
            self.samples["pa_path"].apply(lambda p: os.path.exists(p))
        ].reset_index(drop=True)
        dropped = before - len(self.samples)
        if dropped:
            print(f"[Warning] Dropped {dropped} samples with missing PT files")
        if len(self.samples) == 0:
            raise ValueError("No valid pathology samples found")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        row = self.samples.iloc[idx]
        feat = torch.load(row["pa_path"], map_location="cpu").float()
        label = torch.tensor(row["event"], dtype=torch.long)
        time = torch.tensor(row["time"], dtype=torch.float32)
        return feat, label, time, row["pa_id"]


class CT_Dataset(Dataset):
    """CT image dataset for survival analysis."""

    def __init__(self, data_dir, roi_size=64, augment=False):
        label_file = _get_label_file(data_dir, roi_size)
        df = pd.read_csv(label_file)
        cols = ["ct_id", "ct_path", "event", "time"]
        missing = set(cols) - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns in {label_file}: {sorted(missing)}")
        optional_cols = [c for c in ["pa_id", "h5_path", "split"] if c in df.columns]
        self.samples = df[cols + optional_cols].copy()
        self.samples["ct_id"] = self.samples["ct_id"].astype(str)
        self.samples["pa_id"] = self.samples["pa_id"].astype(str)
        self.samples["ct_path"] = self.samples["ct_path"].astype(str)
        self.samples["event"] = self.samples["event"].astype(int)
        self.samples["time"] = self.samples["time"].astype(float)
        before = len(self.samples)
        self.samples = self.samples[
            self.samples["ct_path"].apply(lambda p: os.path.exists(p))
        ].reset_index(drop=True)
        dropped = before - len(self.samples)
        if dropped:
            print(f"[Warning] Dropped {dropped} samples with missing CT files")
        if len(self.samples) == 0:
            raise ValueError("No valid CT samples found")
        self.ct_aug = _build_ct_augmentation() if augment else None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        row = self.samples.iloc[idx]
        ct = load_ct_npy(row["ct_path"])
        if self.ct_aug is not None:
            ct = self.ct_aug(ct)
        return (
            torch.as_tensor(ct, dtype=torch.float32),
            torch.tensor(row["event"], dtype=torch.long),
            torch.tensor(row["time"], dtype=torch.float32),
            row["ct_id"],
        )


class Pa_CT_Dataset(Dataset):
    """Paired PA+CT dataset for bimodal survival analysis."""

    def __init__(self, data_dir, roi_size=64, augment=False):
        label_file = _get_label_file(data_dir, roi_size)
        df = pd.read_csv(label_file)
        cols = ["pa_id", "pa_path", "h5_path", "ct_id", "ct_path", "event", "time"]
        missing = set(cols) - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns in {label_file}: {sorted(missing)}")
        optional_cols = [c for c in ["split"] if c in df.columns]
        self.samples = df[cols + optional_cols].copy()
        self.samples["pa_id"] = self.samples["pa_id"].astype(str)
        self.samples["ct_id"] = self.samples["ct_id"].astype(str)
        self.samples["pa_path"] = self.samples["pa_path"].astype(str)
        self.samples["ct_path"] = self.samples["ct_path"].astype(str)
        self.samples["event"] = self.samples["event"].astype(int)
        self.samples["time"] = self.samples["time"].astype(float)
        before = len(self.samples)
        self.samples = self.samples[
            self.samples["pa_path"].apply(lambda p: os.path.exists(p))
            & self.samples["ct_path"].apply(lambda p: os.path.exists(p))
        ].reset_index(drop=True)
        dropped = before - len(self.samples)
        if dropped:
            print(f"[Warning] Dropped {dropped} samples with missing CT/PT files")
        if len(self.samples) == 0:
            raise ValueError("No valid paired pathology-CT samples found")
        self.ct_aug = _build_ct_augmentation() if augment else None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        row = self.samples.iloc[idx]
        ct_img = load_ct_npy(row["ct_path"])
        if self.ct_aug is not None:
            ct_img = self.ct_aug(ct_img)
        pa_fea = torch.load(row["pa_path"], map_location="cpu").float()
        case_id = f"{row['pa_id']}|{row['ct_id']}"
        return (
            torch.as_tensor(ct_img, dtype=torch.float32),
            pa_fea,
            torch.tensor(row["event"], dtype=torch.long),
            torch.tensor(row["time"], dtype=torch.float32),
            case_id,
        )
