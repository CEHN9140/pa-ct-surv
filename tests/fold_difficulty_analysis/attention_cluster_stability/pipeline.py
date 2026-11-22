"""Train-only unsupervised dictionaries, fixed attribution and report; no ABMIL training."""
from pathlib import Path
import json,os,subprocess,sys

BASE=Path(__file__).resolve().parent;ROOT=BASE.parents[2]
sys.path.insert(0,str(BASE.parent/'attention_risk_attribution'))
from run import sha256


def main():
    if not json.loads((BASE/'validation/verification.json').read_text())['passed']:raise ValueError('Verification gate failed')
    sources=json.loads((BASE/'validation/source_sha256.json').read_text());state=dict(status='running',pid=os.getpid(),neural_training=False)
    def check():
        for filename,digest in sources.items():
            if sha256(filename)!=digest:raise ValueError(f'Source drift: {filename}')
    def save():
        p=BASE/'status.json';temporary=p.with_suffix('.tmp');temporary.write_text(json.dumps(state,indent=2));temporary.replace(p)
    try:
        for script in ['analyze.py','visualize_qc.py','report.py']:
            check();state['stage']=script;save();print('Starting',script,flush=True)
            subprocess.run([sys.executable,'-u',str(BASE/script)],cwd=ROOT,check=True)
        check();assert json.loads((BASE/'results/validation.json').read_text())['passed']
        state.update(status='completed',stage='all',fold_patient_observations=3605)
    except Exception as error:state.update(status='failed',error=f'{type(error).__name__}: {error}');raise
    finally:save()


if __name__=='__main__':main()
