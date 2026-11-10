"""Run the three fixed-selection CPTAC configurations concurrently on the visible GPU."""
from concurrent.futures import ThreadPoolExecutor,as_completed
from contextlib import ExitStack
from pathlib import Path
import subprocess
import sys
from train import parse_args,run_tag


def main():
    args=parse_args()
    experiment=Path(__file__).resolve().parent
    logs=Path(args.output_dir)/'logs';logs.mkdir(parents=True,exist_ok=True)
    failures=[]
    with ExitStack()as stack:
        processes=[]
        for source in ['uni','uni_l2','keep']:
            args.feature_source=source
            log=logs/f'{run_tag(args)}.log'
            command=[sys.executable,'-u',str(experiment/'train.py'),*sys.argv[1:],'--feature_source',source]
            output=stack.enter_context(log.open('w'))
            process=subprocess.Popen(command,stdout=output,stderr=subprocess.STDOUT)
            processes.append((source,process,log))
            print(f'Started {source} | PID: {process.pid} | Log: {log}',flush=True)
        print('All three configurations launched concurrently.',flush=True)
        with ThreadPoolExecutor(max_workers=3)as pool:
            futures={pool.submit(process.wait):(source,log)for source,process,log in processes}
            for future in as_completed(futures):
                source,log=futures[future]
                code=future.result()
                if code:
                    failures.append(code)
                    print(f'Failed {source}: exit {code} | Other configurations continue | See {log}',flush=True)
                else:print(f'Completed {source}',flush=True)
    if failures:raise SystemExit(failures[0])
    print('Completed UNI, UNI L2 and KEEP.',flush=True)


if __name__=='__main__':main()
