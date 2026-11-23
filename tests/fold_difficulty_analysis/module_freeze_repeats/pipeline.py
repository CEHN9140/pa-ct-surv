"""Fifteen fold jobs, at most three concurrent; only paired baseline/Freeze-C."""
from pathlib import Path
import hashlib,json,os,subprocess,sys,time
BASE=Path(__file__).resolve().parent;ROOT=BASE.parents[2]


def main():
    assert json.loads((BASE/'validation/verification.json').read_text())['passed']
    sources=json.loads((BASE/'validation/source_sha256.json').read_text())
    def check():
        for path,digest in sources.items():
            if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=digest:raise ValueError(f'Source changed:{path}')
    state=dict(status='running',pid=os.getpid(),jobs={});waiting=[(seed,fold) for seed in [7,42,2026] for fold in range(5)];running={};children=[]
    def save():
        path=BASE/'status.json';temp=path.with_suffix('.tmp');temp.write_text(json.dumps(state,indent=2));temp.replace(path)
    try:
        check()
        if any((BASE/f'seed_{seed}').exists() for seed in [7,42,2026]):raise FileExistsError('Existing seed results')
        while waiting or running:
            while waiting and len(running)<3:
                check();seed,fold=waiting.pop(0);directory=BASE/f'seed_{seed}/fold_{fold}';directory.mkdir(parents=True);stream=(directory/'training.log').open('w')
                key=f'{seed}_{fold}';process=subprocess.Popen([sys.executable,'-u',str(BASE/'repeat.py'),'--seed',str(seed),'--fold',str(fold)],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
                running[key]=(process,stream);children.append(process);state['jobs'][key]=dict(pid=process.pid,status='running')
            for key,(process,stream) in list(running.items()):
                code=process.poll()
                if code is not None:
                    stream.close();del running[key];state['jobs'][key].update(status='completed' if code==0 else 'failed',exit_code=code)
                    if code!=0:raise RuntimeError(f'Job {key} failed')
            save()
            if running:time.sleep(5)
        check();state['status']='verifying';save()
        from report import summarize
        summarize();check();state['status']='completed';save()
    except Exception as error:
        for process in children:
            if process.poll() is None:process.terminate()
        state.update(status='failed',error=f'{type(error).__name__}: {error}');save();raise
    finally:
        for process,stream in running.values():stream.close()


if __name__=='__main__':main()
