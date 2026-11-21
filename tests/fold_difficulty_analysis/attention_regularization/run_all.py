"""Three concurrent outer jobs, audited nested comparison, then narrow-scope push."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

BASE=Path(__file__).resolve().parent;ROOT=BASE.parents[2]
sys.path.insert(0,str(BASE.parent/'uniform128_epoch_training'))
from train_uniform import sha256


def main():
    proof=json.loads((BASE/'validation/verification.json').read_text())
    if not proof['passed'] or not proof['gpu_passed']:raise ValueError('Verification gate failed')
    sources=json.loads((BASE/'validation/source_sha256.json').read_text())
    def check():
        for filename,digest in sources.items():
            if sha256(filename)!=digest:raise RuntimeError(f'Source changed: {filename}')
    check()
    if any((BASE/f'fold_{i}').exists() for i in range(5)):raise FileExistsError('Existing fold output')
    state=dict(status='running',pid=os.getpid(),folds={});waiting=list(range(5));running={};children=[]
    def save():
        path=BASE/'validation/pipeline_status.json';temporary=path.with_suffix('.tmp')
        temporary.write_text(json.dumps(state,indent=2));temporary.replace(path)
    try:
        while waiting or running:
            while waiting and len(running)<3:
                check();fold=waiting.pop(0);directory=BASE/f'fold_{fold}';directory.mkdir()
                stream=(directory/'training.log').open('w')
                child=subprocess.Popen([sys.executable,'-u',str(BASE/'train.py'),'--fold',str(fold)],cwd=ROOT,
                    stdout=stream,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
                running[fold]=(child,stream);children.append(child);state['folds'][str(fold)]=dict(pid=child.pid,status='running')
            for fold,(child,stream) in list(running.items()):
                marker=BASE/f'fold_{fold}/status.json'
                if marker.exists():
                    try:state['folds'][str(fold)].update(json.loads(marker.read_text()))
                    except json.JSONDecodeError:pass
                code=child.poll()
                if code is not None:
                    stream.close();del running[fold];state['folds'][str(fold)].update(status='completed' if code==0 else 'failed',exit_code=code)
                    if code!=0:raise RuntimeError(f'Outer job {fold} failed; no publication')
            save()
            if running:time.sleep(5)
        check();state['status']='analyzing';save()
        from report import summarize
        summarize();check()
        (BASE/'status.json').write_text(json.dumps(dict(status='completed',outer_folds=5,inner_fits=35,outer_fits=15,neural_fits=50),indent=2))
        state['status']='publishing';save()
        from publish import publish
        commit=publish();state.update(status='completed_pushed',commit=commit);save()
        print('Completed50 fits, validated and pushed',commit,flush=True)
    except Exception as exc:
        for child in children:
            if child.poll() is None:child.terminate()
        state.update(status='publication_pending' if (BASE/'results/validation.json').exists() else 'failed',error=f'{type(exc).__name__}: {exc}');save();raise
    finally:
        for child,stream in running.values():stream.close()


if __name__=='__main__':main()
