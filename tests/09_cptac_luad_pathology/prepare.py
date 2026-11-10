"""Select one WSI per CPTAC patient and freeze the existing patient-level folds."""
import argparse
from pathlib import Path
from cptac_dataset import prepare_cohort


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source_root',default='/home/gly001/cqj/data/CPTAC/LUAD')
    parser.add_argument('--output_dir',default=str(Path(__file__).resolve().parent/'data'))
    args=parser.parse_args()
    prepare_cohort(args.source_root,args.output_dir)


if __name__=='__main__':main()
