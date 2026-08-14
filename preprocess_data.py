import os
import warnings

import pandas as pd
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# ================= 配置路径 =================
survival_file = "/home/gly001/cqj/data/dz/HE预后.xlsx"
pt_dir = "/home/gly001/cqj/data/lung_cancer/pathology/clam/uni_features/pt_files"
ct_base_dir = "/home/gly001/cqj/data/lung_cancer/ct"
save_root = "/home/gly001/cqj/pa_ct_surv/data"

# CT ROI 尺寸列表
CT_SIZES = [64, 96, 128]
SPLIT_SEED = 42
TEST_SIZE = 0.20

slide_id_col = "病理号"
ct_id_col = "ID"
time_col = "DFS.months"
event_col = "DFS.status"


def clean_id(x):
    if pd.isna(x):
        return ""
    x = str(x).strip()
    if x.lower() in {"nan", "none", ""}:
        return ""
    if x.endswith(".0"):
        x = x[:-2]
    for s in [".nii.gz", ".nii", ".svs", ".pt", ".h5", ".npy"]:
        if x.endswith(s):
            x = x[: -len(s)]
    return x


def prepare_survival_dataframe(source):
    required = [slide_id_col, ct_id_col, time_col, event_col]
    missing = set(required) - set(source.columns)
    if missing:
        raise ValueError(f"HE预后.xlsx 缺少列: {sorted(missing)}")

    df = pd.DataFrame(
        {
            "slide_id": source[slide_id_col].apply(clean_id),
            "ct_id": source[ct_id_col].apply(clean_id),
            "label": pd.to_numeric(source[event_col], errors="coerce"),
            "time": pd.to_numeric(source[time_col], errors="coerce"),
        }
    )

    invalid_event = df["label"].isna() | ~df["label"].isin([0, 1])
    if invalid_event.any():
        values = source.loc[invalid_event, event_col].unique().tolist()
        raise ValueError(f"DFS.status 必须为 0 或 1，发现无效值: {values}")

    df = df[
        (df["slide_id"] != "") & (df["ct_id"] != "") & df["time"].notna()
    ].copy()
    df["label"] = df["label"].astype(int)
    df["time"] = df["time"].astype(float)
    return df.drop_duplicates("slide_id", keep="first").reset_index(drop=True)


def add_train_test_split(samples):
    out = samples.copy()
    event_counts = out["label"].value_counts()
    if (event_counts < 5).any():
        raise ValueError(
            "Cannot create 5-fold survival split; event groups with fewer than 5 "
            f"samples: {event_counts[event_counts < 5].to_dict()}"
        )

    train_idx, test_idx = train_test_split(
        out.index.to_numpy(),
        test_size=TEST_SIZE,
        random_state=SPLIT_SEED,
        shuffle=True,
        stratify=out["label"],
    )

    out["split"] = "test"
    out.loc[train_idx, "split"] = "train"
    print(pd.crosstab(out["split"], out["label"], margins=True))
    return out


if __name__ == "__main__":
    print("Reading HE预后.xlsx...")
    source = pd.read_excel(survival_file)
    df = prepare_survival_dataframe(source)
    df = add_train_test_split(df)
    print(f"Valid survival records: {len(df)}")

    # 病理路径
    df["pt_path"] = df["slide_id"].apply(lambda x: os.path.join(pt_dir, x + ".pt"))
    df["has_pt"] = df["pt_path"].apply(os.path.exists)

    os.makedirs(save_root, exist_ok=True)

    for size in CT_SIZES:
        print(f"\n{'='*40}\nProcessing CT ROI size: {size}\n{'='*40}")
        ct_image_dir = os.path.join(ct_base_dir, f"processed_ct_{size}", "image")

        df[f"ct_image_path_{size}"] = df["ct_id"].apply(
            lambda x: os.path.join(ct_image_dir, x + ".npy")
        )
        df[f"has_ct_{size}"] = df[f"ct_image_path_{size}"].apply(os.path.exists)

        sub_df = df[df["has_pt"] & df[f"has_ct_{size}"]].copy()
        missing_pt = (~df["has_pt"]).sum()
        missing_ct = (~df[f"has_ct_{size}"]).sum()
        print(f"Total HE records: {len(df)}, Missing PT: {missing_pt}, Missing CT (roi{size}): {missing_ct}")
        print(f"Valid paired samples: {len(sub_df)}")

        if sub_df.empty:
            print(f"  [Skip] No valid samples for ROI {size}")
            continue

        out_df = sub_df[
            [
                "slide_id",
                "pt_path",
                "ct_id",
                f"ct_image_path_{size}",
                "label",
                "time",
                "split",
            ]
        ].copy()
        out_df.rename(columns={f"ct_image_path_{size}": "ct_image_path"}, inplace=True)

        csv_path = os.path.join(save_root, f"all_label_roi{size}.csv")
        out_df.to_csv(csv_path, index=False)

        print(f"\nSaved {len(out_df)} samples to {csv_path}")
        print("Label distribution:")
        print(out_df["label"].value_counts().sort_index())
        print(out_df["label"].value_counts(normalize=True).sort_index())

    print("\nDone! Generated files:")
    for size in CT_SIZES:
        p = os.path.join(save_root, f"all_label_roi{size}.csv")
        if os.path.exists(p):
            print(f"  {p} ({len(pd.read_csv(p))} samples)")
