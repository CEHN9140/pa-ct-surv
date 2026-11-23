"""Local reuse of the immutable three-arm backend for two-arm seed replication."""
from pathlib import Path
from contextlib import contextmanager
import importlib.util,sys
from unittest.mock import patch

BASE=Path(__file__).resolve().parent;ORIGINAL=BASE.parent/'module_freeze';ROOT=BASE.parents[2]
sys.path[:0]=[str(ORIGINAL),str(ROOT)]


def load_backend(name,filename):
    spec=importlib.util.spec_from_file_location(name,ORIGINAL/filename);module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module);return module


trainer=load_backend('freeze_repeat_backend','train.py')


@contextmanager
def training_context(seed,directory):
    original_seed=trainer.seed_everything
    with patch.object(trainer,'BASE',Path(directory)),patch.object(trainer,'ARMS',['baseline','freeze_c']),patch.object(trainer,'seed_everything',lambda ignored:original_seed(seed)):
        yield


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--seed',type=int,choices=[7,42,2026],required=True);parser.add_argument('--fold',type=int,choices=range(5),required=True);args=parser.parse_args()
    with training_context(args.seed,BASE/f'seed_{args.seed}'):trainer.run_fold(args.fold)
