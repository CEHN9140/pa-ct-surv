import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold

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


def add_cv_splits(samples):
    """Randomly assign every patient to five stratified folds without a fixed seed."""
    n_splits = 5
    result = samples.copy().reset_index(drop=True)
    result["split"] = -1
    events = result["event"].to_numpy()

    insufficient = pd.Series(events).value_counts().reindex([0, 1], fill_value=0)
    insufficient = insufficient[insufficient < n_splits]
    if not insufficient.empty:
        raise ValueError(
            f"Event groups must contain at least {n_splits} samples "
            f"for {n_splits}-fold CV: {insufficient.to_dict()}"
        )

    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True)
    for fold, (_, validation_positions) in enumerate(
        splitter.split(result, events)
    ):
        result.loc[validation_positions, "split"] = fold

    if (result["split"] < 0).any():
        raise RuntimeError("Failed to assign every sample to a CV fold")
    return result


def build_roi_cohort(records, roi_size, pa_dir, coord_dir, ct_base_dir, pa_source="clam"):
    """Match pathology features and one CT ROI into the final paired cohort."""
    if pa_source not in {"clam", "trident"}:
        raise ValueError("pa_source must be 'clam' or 'trident'")
    pa_dir = Path(pa_dir)
    coord_dir = Path(coord_dir)
    ct_dir = Path(ct_base_dir) / f"processed_ct_{roi_size}" / "image"

    data = records.copy()
    extension = ".pt" if pa_source == "clam" else ".h5"
    data["pa_path"] = data["pa_id"].map(lambda value: str(pa_dir / f"{value}{extension}"))
    coord_suffix = ".h5" if pa_source == "clam" else "_patches.h5"
    data["coord_path"] = data["pa_id"].map(
        lambda value: str(coord_dir / f"{value}{coord_suffix}")
    )
    data["ct_path"] = data["ct_id"].map(lambda value: str(ct_dir / f"{value}.npy"))

    has_pa = data["pa_path"].map(lambda path: Path(path).is_file())
    has_ct = data["ct_path"].map(lambda path: Path(path).is_file())
    has_coords = data["coord_path"].map(lambda path: Path(path).is_file())
    paired = data.loc[
        has_pa & has_ct,
        ["pa_id", "pa_path", "coord_path", "ct_id", "ct_path", "event", "time"],
    ].copy()
    return (
        paired,
        int((~has_pa).sum()),
        int((~has_ct).sum()),
        int((~has_coords).sum()),
    )


def write_roi_csv(records, roi_size, output_dir, pa_dir, coord_dir, ct_base_dir,
                  pa_source="clam"):
    cohort, missing_pa, missing_ct, missing_coords = build_roi_cohort(
        records, roi_size, pa_dir, coord_dir, ct_base_dir, pa_source=pa_source
    )
    print(
        f"\nROI {roi_size}: total={len(records)}, paired={len(cohort)}, "
        f"missing PA={missing_pa}, missing CT={missing_ct}, missing coordinates={missing_coords}"
    )
    if cohort.empty:
        print("  [Skip] No valid paired samples")
        return None

    output = add_cv_splits(cohort)[
        ["pa_id", "pa_path", "coord_path", "ct_id", "ct_path", "event", "time", "split"]
    ]
    csv_path = output_dir / f"all_label_roi{roi_size}.csv"
    output.to_csv(csv_path, index=False)

    print(f"  Saved: {csv_path} ({len(output)} samples)")
    print(f"  Split counts: {output['split'].value_counts().sort_index().to_dict()}")
    print(f"  Event counts: {output['event'].value_counts().sort_index().to_dict()}")
    return csv_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build CLAM/TRIDENT paired survival CSVs with five-fold CV (no test set)."
    )
    parser.add_argument("--pa-source", choices=["clam", "trident"], default="clam")
    parser.add_argument("--pa-feature-dir", default=None,
                        help="CLAM pt_files or TRIDENT features_<encoder> directory.")
    parser.add_argument("--pa-coords-dir", default=None,
                        help="Coordinate H5 directory (CLAM: <id>.h5; TRIDENT: <id>_patches.h5).")
    parser.add_argument(
        "--data-root",
        default=None,
        help="Output CSV directory; default: data/<source>/<feature-directory>.",
    )
    return parser.parse_args()


def main(args):
    survival_file = "/home/gly001/cqj/data/dz/HE预后.xlsx"
    pathology_dir = Path("/home/gly001/cqj/data/lung_cancer/pathology")
    if args.pa_source == "clam":
        pa_dir = Path(args.pa_feature_dir or pathology_dir / "clam/uni_features/pt_files")
        coord_dir = Path(args.pa_coords_dir or pathology_dir / "clam/patches")
        feature_name = pa_dir.parent.name if pa_dir.name == "pt_files" else pa_dir.name
    else:
        pa_dir = Path(args.pa_feature_dir or pathology_dir / "trident/20x_256px_0px_overlap/features_uni_v1")
        coord_dir = Path(args.pa_coords_dir) if args.pa_coords_dir else pa_dir.parent / "patches"
        feature_name = pa_dir.name
    pa_dir, coord_dir = pa_dir.resolve(), coord_dir.resolve()
    for directory in (pa_dir, coord_dir):
        if not directory.is_dir():
            raise FileNotFoundError(f"Input directory not found: {directory}")
    ct_base_dir = "/home/gly001/cqj/data/lung_cancer/ct"
    output_dir = (Path(args.data_root) if args.data_root else
                  Path(__file__).resolve().parent / "data" / args.pa_source / feature_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading {survival_file}")
    print(f"Pathology source: {args.pa_source} | Features: {pa_dir}")
    print(f"Coordinates: {coord_dir}")
    print("Split: random stratified 5-fold CV without a fixed seed; no held-out test set")
    records = prepare_survival_dataframe(pd.read_excel(survival_file))
    print(f"Valid clinical records before modality matching: {len(records)}")

    generated = [
        write_roi_csv(
            records,
            roi_size,
            output_dir,
            pa_dir=pa_dir,
            coord_dir=coord_dir,
            ct_base_dir=ct_base_dir,
            pa_source=args.pa_source,
        )
        for roi_size in (64, 96, 128)
    ]
    generated = [path for path in generated if path is not None]
    print(f"Done. Generated {len(generated)} CSV file(s) in {output_dir}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
