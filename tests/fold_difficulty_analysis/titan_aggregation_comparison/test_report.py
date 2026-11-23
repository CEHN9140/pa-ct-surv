from pathlib import Path
import sys,json
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).parent))


def test_paired_report_replays_scores_and_keeps_patient_output_private(tmp_path,monkeypatch):
    import report
    from sksurv.metrics import concordance_index_censored
    here=tmp_path/'titan';here.mkdir();monkeypatch.setattr(report,'HERE',here)
    (here/'config.json').write_text(json.dumps(dict(alphas=[1000],revision='test',bootstrap_repeats=30,bootstrap_seed=123)))
    old=tmp_path/'encoder_ridge_comparison/results';old.mkdir(parents=True)
    pd.DataFrame([dict(encoder='uni_v1',val_cindex=.72)]).to_csv(old/'summary.csv',index=False)
    for arm in ['conch_mean','titan']:
        d=here/'results'/arm;d.mkdir(parents=True);rows=[]
        for fold in range(5):
            folder=d/f'private/fold_{fold}';folder.mkdir(parents=True)
            t=np.array([1.,10.,35.,60.,90.]);e=np.array([1,1,1,0,0],bool)
            r=np.array([5.,4.,3.,2.,1.]) if arm=='titan' else np.array([4.,5.,3.,2.,1.])
            ci=concordance_index_censored(e,t,r)[0]
            p=pd.DataFrame({'case_id':[f'{fold}_{i}' for i in range(5)],'dfs.month':t,'dfs.status':e.astype(int),'risk_score':r})
            for role in ['train','val']:p.to_csv(folder/f'{role}_predictions.csv',index=False)
            rows.append(dict(arm=arm,fold=fold,alpha=1000,train_cindex=ci,val_cindex=ci,gap=0,auc_36m=.7,auc_60m=.7))
        pd.DataFrame(rows).to_csv(d/'metrics.csv',index=False)
        pd.DataFrame(dict(converged=[True]*150)).to_csv(d/'inner_cv.csv',index=False)
    report.build_report()
    root=here/'results';p=json.loads((root/'paired_comparison.json').read_text())
    assert p['delta_cindex']>0 and p['positive_folds']==5
    for f in root.glob('*.csv'):assert 'case_id' not in pd.read_csv(f,nrows=0).columns
    assert json.loads((root/'validation.json').read_text())['prediction_replays']==20


def test_report_imports_from_outside_repository(tmp_path):
    import subprocess,sys
    script=Path(__file__).with_name('report.py').resolve()
    result=subprocess.run([sys.executable,'-I','-c',f'import runpy;runpy.run_path({str(script)!r},run_name="import_test")'],
                          cwd=tmp_path,capture_output=True,text=True)
    assert result.returncode==0,result.stderr
