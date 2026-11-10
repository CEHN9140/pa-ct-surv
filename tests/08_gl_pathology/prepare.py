"""Prepare and freeze the GL binary-status pathology cohort; no training."""
import argparse
from pathlib import Path

from gl_dataset import prepare_cohort


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--labels_csv', default='/home/gly001/cqj/data/yixianai/GL_病理_CT_生存.csv')
    parser.add_argument('--feature_dir', default='/home/gly001/cqj/data/yixianai/GL/pathology/trident/20x_256px_0px_overlap/features_uni_v1')
    parser.add_argument('--output_dir', default=str(Path(__file__).resolve().parent/'data_154_eventtime'))
    args = parser.parse_args()
    prepare_cohort(args.labels_csv, args.feature_dir, args.output_dir)


if __name__ == '__main__':
    main()
