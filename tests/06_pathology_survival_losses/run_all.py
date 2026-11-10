"""Run all six experiment06 combinations sequentially, with a separate log each."""

from pathlib import Path
import subprocess
import sys


def main():
    experiment = Path(__file__).resolve().parent
    logs = experiment / 'logs'
    logs.mkdir(parents=True, exist_ok=True)
    for source in ['uni_l2', 'keep', 'eagle']:
        for loss in ['nll', 'cox_rank']:
            log = logs / f'{source}_{loss}.log'
            command = [sys.executable, '-u', str(experiment / 'train.py'), *sys.argv[1:],
                       '--feature_source', source, '--survival_loss', loss]
            print(f'Starting {source} / {loss} | Log: {log}', flush=True)
            with log.open('w') as output:
                result = subprocess.run(command, stdout=output, stderr=subprocess.STDOUT)
            if result.returncode:
                print(f'Failed: {source} / {loss} | Exit: {result.returncode} | See {log}', flush=True)
                raise SystemExit(result.returncode)
            print(f'Completed {source} / {loss}', flush=True)
    print('All six experiment06 runs completed.', flush=True)


if __name__ == '__main__':
    main()
