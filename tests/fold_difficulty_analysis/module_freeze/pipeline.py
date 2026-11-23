"""At most three independent outer-fold workers; verify all completed interventions."""
from pathlib import Path
import hashlib,json,os,subprocess,sys,time

BASE=Path(__file__).resolve().parent;ROOT=BASE.parents[2]


def main():
    assert json.loads((BASE/'validation/verification.json').read_text())['passed']
    sources=json.loads((BASE/'validation/source_sha256.json').read_text())
    def check():
        for filename,expected in sources.items():
            if hashlib.sha256(Path(filename).read_bytes()).hexdigest()!=expected:raise ValueError(f'Source drift:{filename}')
    state=dict(status='running',pid=os.getpid(),folds={});waiting=list(range(5));running={};children=[]
    def save():
        path=BASE/'status.json';temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(state,indent=2));temporary.replace(path)
    try:
        check()
        if any((BASE/f'fold_{i}').exists() for i in range(5)):raise FileExistsError('Existing fold output')
        while waiting or running:
            while waiting and len(running)<3:
                check();fold=waiting.pop(0);folder=BASE/f'fold_{fold}';folder.mkdir();stream=(folder/'training.log').open('w')
                process=subprocess.Popen([sys.executable,'-u',str(BASE/'train.py'),'--fold',str(fold)],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
                running[fold]=(process,stream);children.append(process);state['folds'][str(fold)]=dict(pid=process.pid,status='running')
            for fold,(process,stream) in list(running.items()):
                code=process.poll()
                if code is not None:
                    stream.close();del running[fold];state['folds'][str(fold)].update(status='completed' if code==0 else 'failed',exit_code=code)
                    if code!=0:raise RuntimeError(f'Fold {fold} failed')
            save()
            if running:time.sleep(5)
        check();state['status']='verifying';save()
        from report import summarize
        summarize();check();state['status']='completed';save();print('Completed and verified three-arm fivefold experiment',flush=True)
    except Exception as error:
        for process in children:
            if process.poll() is None:process.terminate()
        state.update(status='failed',error=f'{type(error).__name__}: {error}');save();raise
    finally:
        for process,stream in running.values():stream.close()


if __name__=='__main__':main()
