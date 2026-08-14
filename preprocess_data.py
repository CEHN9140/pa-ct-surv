import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

CT_SIZES = [64, 96, 128]


def prepare_survival_dataframe(source):
    """Clean clinical records and enforce one-to-one path/CT identifiers."""
    columns = {
        "pa_id": "病理号",
        "ct_id": "ID",
        "time": "DFS.months",
        "event": "DFS.status",
    }
    missing = set(columns.values()) - set(source.columns)
    if missing:
        raise ValueError(f"HE预后.xlsx 缺少列: {sorted(missing)}")

    data = pd.DataFrame(
        {
            "pa_id": source[columns["pa_id"]].astype(str).str.strip(),
            "ct_id": source[columns["ct_id"]].astype(str).str.strip(),
            "event": pd.to_numeric(source[columns["event"]], errors="coerce"),
            "time": pd.to_numeric(source[columns["time"]], errors="coerce"),
        }
    )

    invalid_event = data["event"].isna() | ~data["event"].isin([0, 1])
    if invalid_event.any():
        values = source.loc[invalid_event, columns["event"]].unique().tolist()
        raise ValueError(f"DFS.status 必须为 0 或 1，发现无效值: {values}")

    data = data[
        data["pa_id"].ne("") & data["ct_id"].ne("") & data["time"].notna()
    ].copy()
    data["event"] = data["event"].astype(int)
    data["time"] = data["time"].astype(float)
    data.reset_index(drop=True, inplace=True)

    for column in ("pa_id", "ct_id"):
        duplicated = data[column].duplicated(keep=False)
        if duplicated.any():
            values = data.loc[duplicated, column].tolist()
            raise ValueError(f"{column} 必须唯一，发现重复值: {values}")
    return data


def add_locked_splits(
    samples,
    seed=42,
):
    """Assign -1 to locked test rows and 0..4 to train CV folds."""
    test_size = 0.20
    n_splits = 5
    insufficient = samples["event"].value_counts()
    insufficient = insufficient[insufficient < n_splits]
    if not insufficient.empty:
        raise ValueError(
            f"Cannot create {n_splits}-fold split; event groups have fewer than "
            f"{n_splits} samples: {insufficient.to_dict()}"
        )

    train_indices, _ = train_test_split(
        samples.index.to_numpy(),
        test_size=test_size,
        random_state=seed,
        shuffle=True,
        stratify=samples["event"],
    )
    result = samples.copy()
    result["split"] = -1

    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    train_events = result.loc[train_indices, "event"].to_numpy()
    for fold, (_, validation_positions) in enumerate(
        splitter.split(train_indices, train_events)
    ):
        result.loc[train_indices[validation_positions], "split"] = fold

    if (result.loc[train_indices, "split"] < 0).any():
        raise RuntimeError("Failed to assign every train sample to a CV fold")
    return result


def build_roi_cohort(records, roi_size, pa_dir, h5_dir, ct_base_dir):
    """Match pathology features and one CT ROI into the final paired cohort."""
    pa_dir = Path(pa_dir)
    h5_dir = Path(h5_dir)
    ct_dir = Path(ct_base_dir) / f"processed_ct_{roi_size}" / "image"

    data = records.copy()
    data["pa_path"] = data["pa_id"].map(lambda value: str(pa_dir / f"{value}.pt"))
    data["h5_path"] = data["pa_id"].map(lambda value: str(h5_dir / f"{value}.h5"))
    data["ct_path"] = data["ct_id"].map(lambda value: str(ct_dir / f"{value}.npy"))

    has_pa = data["pa_path"].map(lambda path: Path(path).exists())
    has_ct = data["ct_path"].map(lambda path: Path(path).exists())
    has_h5 = data["h5_path"].map(lambda path: Path(path).exists())
    paired = data.loc[
        has_pa & has_ct,
        ["pa_id", "pa_path", "h5_path", "ct_id", "ct_path", "event", "time"],
    ].copy()
    return (
        paired,
        int((~has_pa).sum()),
        int((~has_ct).sum()),
        int((~has_h5).sum()),
    )


def write_roi_csv(records, roi_size, seed, output_dir, pa_dir, h5_dir, ct_base_dir):
    cohort, missing_pa, missing_ct, missing_h5 = build_roi_cohort(
        records, roi_size, pa_dir, h5_dir, ct_base_dir
    )
    print(
        f"\nROI {roi_size}: total={len(records)}, paired={len(cohort)}, "
        f"missing PA={missing_pa}, missing CT={missing_ct}, missing H5={missing_h5}"
    )
    if cohort.empty:
        print("  [Skip] No valid paired samples")
        return None

    output = add_locked_splits(cohort, seed=seed)[
        ["pa_id", "pa_path", "h5_path", "ct_id", "ct_path", "event", "time", "split"]
    ]
    csv_path = output_dir / f"all_label_roi{roi_size}.csv"
    output.to_csv(csv_path, index=False)

    print(f"  Saved: {csv_path} ({len(output)} samples)")
    print(f"  Split counts: {output['split'].value_counts().sort_index().to_dict()}")
    print(f"  Event counts: {output['event'].value_counts().sort_index().to_dict()}")
    return csv_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build paired pathology-CT survival CSVs with locked splits."
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--data-root",
        default="/home/gly001/cqj/pa_ct_surv/data",
        help="Root directory for generated seed-specific CSVs.",
    )
    return parser.parse_args()


def main(args):
    survival_file = "/home/gly001/cqj/data/dz/HE预后.xlsx"
    pa_dir = "/home/gly001/cqj/data/lung_cancer/pathology/clam/uni_features/pt_files"
    h5_dir = "/home/gly001/cqj/data/lung_cancer/pathology/clam/uni_features/h5_files"
    ct_base_dir = "/home/gly001/cqj/data/lung_cancer/ct"
    output_dir = Path(args.data_root) / f"seed_{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading {survival_file}")
    records = prepare_survival_dataframe(pd.read_excel(survival_file))
    print(f"Valid clinical records before modality matching: {len(records)}")

    generated = [
        write_roi_csv(
            records,
            roi_size,
            args.seed,
            output_dir,
            pa_dir=pa_dir,
            h5_dir=h5_dir,
            ct_base_dir=ct_base_dir,
        )
        for roi_size in CT_SIZES
    ]
    generated = [path for path in generated if path is not None]
    print(f"Done. Generated {len(generated)} CSV file(s) in {output_dir}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
