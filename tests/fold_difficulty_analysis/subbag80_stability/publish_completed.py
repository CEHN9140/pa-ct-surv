"""User-authorized post-completion publication; independent of pinned scientific pipeline."""
from pathlib import Path
import hashlib,json,os,subprocess,time

BASE=Path(__file__).resolve().parent;ROOT=BASE.parents[2]


def publication_paths(base):
    names=['.gitignore','README.md','PLAN.md','subbag.py','run.py','test_subbag.py','pipeline.py','run.sh','publish_completed.py',
        'validation/verification.json','validation/source_sha256.json','validation/publish_authorization.json',
        'results/fold_metrics.csv','results/summary.csv','results/late_minus_best.csv','results/report.md','results/validation.json']
    return [Path(base)/p for p in names]


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def git(*args):
    return subprocess.check_output(['git',*args],cwd=ROOT,env=dict(os.environ,GIT_TERMINAL_PROMPT='0'),text=True).strip()


def network(*args):
    for attempt in range(3):
        try:return git('-c','http.version=HTTP/1.1',*args)
        except subprocess.CalledProcessError:
            if attempt==2:raise
            time.sleep(2)


def publish():
    if not json.loads((BASE/'validation/publish_authorization.json').read_text())['user_requested_push_after_completion']:raise ValueError('Missing user authorization')
    status=json.loads((BASE/'status.json').read_text());proof=json.loads((BASE/'results/validation.json').read_text())
    if status['status']!='completed' or proof['training'] or proof['patient_phase_observations']!=7210 or proof['folds']!=5 or proof['repeats']!=30:raise ValueError('Full diagnostic incomplete')
    if not all(proof[key] for key in ['passed','paired_subsets','full_risks_reproduced','direct_subset_checks','saved_risks_metrics_independently_recomputed']):raise ValueError('Scientific verification failed')
    for filename,expected in json.loads((BASE/'validation/source_sha256.json').read_text()).items():
        if digest(filename)!=expected:raise ValueError(f'Pinned source drift: {filename}')
    if git('diff','--cached','--name-only'):raise RuntimeError('Other staged files; no automatic commit')
    if git('branch','--show-current')!='main' or git('config','--get','remote.origin.url')!='https://github.com/CEHN9140/pa-ct-surv.git':raise RuntimeError('Branch/remote changed')
    network('fetch','origin','main');before=git('rev-parse','HEAD')
    if before!=git('rev-parse','origin/main'):raise RuntimeError('Branch divergence; no unrelated commits will be pushed')
    files=publication_paths(BASE);hashes={}
    for p in files:
        if not p.is_file() or p.stat().st_size>5*1024*1024:raise ValueError(f'Invalid publication artifact: {p.name}')
        if 'private' in p.relative_to(BASE).parts or p.suffix in ['.pth','.npy','.npz','.joblib'] or 'predictions' in p.name:raise ValueError('Private data/weights excluded')
        if p.suffix=='.csv':
            with p.open() as stream:header=stream.readline().strip()
            if any(column in header.split(',') for column in ['pa_id','case_id','event_pa_id','other_pa_id']):raise ValueError('Patient-level table excluded')
        hashes[p]=digest(p)
    paths=sorted(str(p.relative_to(ROOT)) for p in files)
    git('add','-f','--',*paths)
    if set(git('diff','--cached','--name-only').splitlines())!=set(paths):raise RuntimeError('Staged scope differs')
    git('diff','--cached','--check')
    if any(digest(p)!=expected for p,expected in hashes.items()):raise RuntimeError('Publication artifacts changed while staging')
    git('commit','--only','-m','Add paired80-percent subbag stability diagnostic for frozen ABMIL','--',*paths)
    commit=git('rev-parse','HEAD')
    if git('rev-parse',commit+'^')!=before or set(git('diff-tree','--no-commit-id','--name-only','-r',commit).splitlines())!=set(paths):raise RuntimeError('Commit scope/concurrency check failed')
    network('push','origin',commit+':refs/heads/main')
    return commit


def main():
    state=dict(status='waiting_for_scientific_completion',pid=os.getpid());self_hash=digest(__file__)
    path=BASE/'validation/publish_status.json'
    def save():
        temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(state,indent=2));temporary.replace(path)
    try:
        while True:
            current=json.loads((BASE/'status.json').read_text())
            if current['status']=='failed':raise RuntimeError('Scientific pipeline failed; no publication')
            if current['status']=='completed':break
            try:os.kill(int(current['pid']),0)
            except ProcessLookupError:raise RuntimeError('Scientific process no longer exists; no publication')
            save();time.sleep(30)
        if digest(__file__)!=self_hash:raise RuntimeError('Publisher source changed')
        state['status']='publishing';save();commit=publish();state.update(status='completed_pushed',commit=commit);save()
        print('Validated and pushed',commit,flush=True)
    except Exception as error:
        state.update(status='publication_pending',error=f'{type(error).__name__}: {error}');save();raise


if __name__=='__main__':main()
