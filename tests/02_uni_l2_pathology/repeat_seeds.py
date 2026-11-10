"""Repeat a fixed experiment configuration with two additional training seeds."""

import subprocess
import sys
from pathlib import Path


def main():
    experiment = Path(__file__).resolve().parent
    arguments = sys.argv[1:]
    if any(arg == '--seed' or arg.startswith('--seed=') for arg in arguments):
        raise ValueError('Training seeds are fixed here to 42 and 2024; do not pass --seed')
    logs = experiment / 'logs'
    logs.mkdir(parents=True, exist_ok=True)
    for seed in (42, 2024):
        log = logs / f'stability_seed{seed}.log'
        print(f'Starting 5-fold training: seed={seed} | Log: {log}', flush=True)
        with log.open('w') as handle:
            subprocess.run(
                [sys.executable, '-u', str(experiment / 'train.py'),
                 *arguments, '--seed', str(seed)],
                stdout=handle, stderr=subprocess.STDOUT, check=True)
        print(f'Completed seed={seed}', flush=True)
    print('Completed both additional seeds; existing seed123 results were not rerun.', flush=True)


if __name__ == '__main__':
    main()
