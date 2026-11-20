"""Parallel diagnostics → verified aggregate report → experiment-only push."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

BASE=Path(__file__).resolve().parent;ROOT=BASE.parents[2]
sys.path.insert(0,str(BASE.parent/'uniform128_epoch_training'))
from train_uniform import sha256


def main():
    proof=json.loads((BASE/'validation/verification.json').read_text())
    if not proof['passed'] or not proof['gpu_passed']:raise ValueError('Verification gate not passed')
    sources=json.loads((BASE/'validation/source_sha256.json').read_text())
    def check_sources():
        for filename,digest in sources.items():
            if sha256(filename)!=digest:raise RuntimeError(f'Source changed: {filename}')
    check_sources()
    if any((BASE/f'bs{b}').exists() for b in [32,64,128]):raise FileExistsError('Existing condition; refusing overwrite')
    state=dict(status='running',pid=os.getpid(),conditions={});children=[];streams=[]
    def save():
        (BASE/'validation/pipeline_status.json').write_text(json.dumps(state,indent=2))
    try:
        for batch in [32,64,128]:
            condition=BASE/f'bs{batch}';condition.mkdir()
            stream=(condition/'training.log').open('w');streams.append(stream)
            child=subprocess.Popen([sys.executable,'-u',str(BASE/'train.py'),'--batch',str(batch)],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
            children.append((batch,child));state['conditions'][str(batch)]=dict(pid=child.pid,status='running')
        save()
        while any(child.poll() is None for _,child in children):
            for batch,child in children:
                condition=state['conditions'][str(batch)];code=child.poll()
                if code is not None:condition.update(status='completed' if code==0 else 'failed',exit_code=code)
            save();time.sleep(5)
        for batch,child in children:state['conditions'][str(batch)].update(status='completed' if child.returncode==0 else 'failed',exit_code=child.returncode)
        save()
        if any(child.returncode!=0 for _,child in children):raise RuntimeError('Condition failed; no publication')
        check_sources();state['status']='analyzing';save()
        from report import summarize
        summarize()
        check_sources()
        (BASE/'status.json').write_text(json.dumps(dict(status='completed',conditions=3,completed_folds=15,report='results/report.md'),indent=2))
        state['status']='publishing';save()
        from publish import publish
        commit=publish();state.update(status='completed_pushed',commit=commit);save()
        print(f'Completed15 folds, verified diagnostics and pushed {commit}',flush=True)
    except Exception as exc:
        state.update(status='publication_pending' if (BASE/'results/validation.json').exists() else 'failed',error=f'{type(exc).__name__}: {exc}');save();raise
    finally:
        for stream in streams:stream.close()


if __name__=='__main__':main()
