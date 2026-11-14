"""Run extraction and/or nested CT-only Radiomics survival modelling."""

import argparse
from pathlib import Path

from radiomics_ct import run


if __name__ == '__main__':
    experiment = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cohort_csv', default=str(experiment.parents[1] / 'data/eagle/top25/all_label_roi64.csv'))
    parser.add_argument('--ct_root', default='/home/gly001/cqj/data/lung_cancer/ct')
    parser.add_argument('--output_dir', default=str(experiment / 'output'))
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--stage', choices=['all', 'extract', 'fit'], default='all')
    run(parser.parse_args())
