"""Publish this completed, validated experiment only; never weights or patient data."""
import json
import os
from pathlib import Path
import subprocess
import time

BASE=Path(__file__).resolve().parent;ROOT=BASE.parents[2]


def git(*args):
    env=dict(os.environ,GIT_TERMINAL_PROMPT='0')
    return subprocess.check_output(['git',*args],cwd=ROOT,env=env,text=True).strip()


def network(*args):
    for attempt in range(3):
        try:return git('-c','http.version=HTTP/1.1',*args)
        except subprocess.CalledProcessError:
            if attempt==2:raise
            time.sleep(2)


def publish():
    if not json.loads((BASE/'results/validation.json').read_text())['passed']:raise ValueError('Scientific verification failed')
    if git('diff','--cached','--name-only'):raise RuntimeError('Other staged changes; publication needs manual follow-up')
    if git('branch','--show-current')!='main':raise RuntimeError('Branch changed; no automatic publication')
    if git('config','--get','remote.origin.url')!='https://github.com/CEHN9140/pa-ct-surv.git':raise RuntimeError('GitHub remote changed')
    network('fetch','origin','main')
    before=git('rev-parse','HEAD')
    if before!=git('rev-parse','origin/main'):raise RuntimeError('Branch divergence; no unrelated commits will be pushed')
    files=[BASE/p for p in ['.gitignore','README.md','PLAN.md','regularization.py','protocol.py','train.py','report.py','publish.py',
        'run_all.py','run.sh','test_regularization.py','experiment_config.json','status.json','validation/verification.json','validation/source_sha256.json']]
    files.extend(BASE/'results'/name for name in ['fold_metrics.csv','selected_parameters.csv','inner_candidate_curves.csv','selected_inner_curves.csv',
        'fit_audit.csv','summary.csv','paired_comparisons.csv','inner_stability.png','report.md','validation.json'])
    from protocol import CANDIDATES
    for fold in range(5):
        directory=BASE/f'fold_{fold}'
        files.extend(directory/name for name in ['training.log','status.json','role_counts.csv','selection.csv'])
        for config in CANDIDATES:
            files.extend(directory/'inner'/config['key']/name for name in ['audit.json','epoch_metrics.csv','checkpoint_metadata.yaml'])
        for family in ['baseline','entropy','neff']:
            files.extend(directory/'outer'/family/name for name in ['run_config.yaml','audit.json','epoch_metrics.csv','metrics.csv','completed.json',
                'checkpoint_metadata.yaml','results/km_high_vs_low.png'])
    paths=[]
    for p in files:
        relative=p.relative_to(ROOT)
        if not p.is_file() or p.stat().st_size>5*1024*1024:raise ValueError(f'Invalid publication file: {relative}')
        if 'private' in p.parts or p.suffix in ['.pth','.npz','.npy'] or 'predictions' in p.name or 'private_cohort' in p.name:
            raise ValueError('Patient data/weights cannot be published')
        paths.append(str(relative))
    paths=sorted(set(paths))
    git('add','-f','--',*paths)
    if set(git('diff','--cached','--name-only').splitlines())!=set(paths):raise RuntimeError('Staged scope differs')
    git('diff','--cached','--check')
    git('commit','--only','-m','Add nested attention entropy and effective-patch regularization experiments','--',*paths)
    commit=git('rev-parse','HEAD')
    if git('rev-parse',f'{commit}^')!=before:raise RuntimeError('Concurrent commit; publication aborted')
    changed=git('diff-tree','--no-commit-id','--name-only','-r',commit).splitlines()
    if set(changed)!=set(paths):raise RuntimeError('Commit scope differs')
    network('push','origin',f'{commit}:refs/heads/main')
    return commit


if __name__=='__main__':print(publish())
