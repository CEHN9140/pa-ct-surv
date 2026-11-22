"""Run attribution, native-patch QC and report; never train or publish anything."""
from pathlib import Path
import json,os,subprocess,sys
from run import sha256

BASE=Path(__file__).resolve().parent;ROOT=BASE.parents[2]


def main():
    proof=json.loads((BASE/'validation/verification.json').read_text())
    if not proof['passed']:raise ValueError('Verification gate failed')
    manifest=json.loads((BASE/'validation/source_sha256.json').read_text())
    state=dict(status='running',pid=os.getpid(),neural_training=False)
    def save():
        p=BASE/'status.json';temporary=p.with_suffix('.tmp');temporary.write_text(json.dumps(state,indent=2));temporary.replace(p)
    def check():
        for filename,digest in manifest.items():
            if sha256(filename)!=digest:raise ValueError(f'Source drift: {filename}')
    try:
        for script in ['run.py','quality.py','report.py']:
            check();state['stage']=script;save()
            print('Starting',script,flush=True)
            subprocess.run([sys.executable,'-u',str(BASE/script)],cwd=ROOT,check=True)
        check()
        if not json.loads((BASE/'results/validation.json').read_text())['passed']:raise ValueError('Final verification failed')
        state.update(status='completed',stage='all',patients=721,outer_folds=5)
    except Exception as error:state.update(status='failed',error=f'{type(error).__name__}: {error}');raise
    finally:save()


if __name__=='__main__':main()
