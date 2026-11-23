from pathlib import Path
import sys
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).parent))
from repeat import trainer,training_context


def test_training_seed_redirect_and_scope_restore(tmp_path,monkeypatch):
    seen=[];original_base=trainer.BASE;original_arms=trainer.ARMS
    monkeypatch.setattr(trainer,'seed_everything',lambda seed:seen.append(seed))
    with training_context(7,tmp_path):
        trainer.seed_everything(123)
        assert trainer.ARMS==['baseline','freeze_c'] and trainer.BASE==tmp_path
    with training_context(42,tmp_path):trainer.seed_everything(123)
    assert seen==[7,42] and trainer.BASE==original_base and trainer.ARMS==original_arms


def test_partition_fixed_across_training_seeds(tmp_path):
    frame=pd.DataFrame(dict(split=np.repeat(np.arange(5),10),event=[0,1]*25))
    expected=trainer.partition(frame,0)
    for seed in [7,42,2026]:
        with training_context(seed,tmp_path):
            trainer.seed_everything(123)
            for a,b in zip(expected,trainer.partition(frame,0)):np.testing.assert_array_equal(a,b)


def test_initialization_varies_only_with_training_seed(tmp_path):
    from freeze import weight_hash
    values=[]
    for seed in [7,42,7]:
        with training_context(seed,tmp_path):
            trainer.seed_everything(123);values.append(weight_hash(trainer.Pa_Model(feature_dim=1024,abmil_dropout=.25)))
    assert values[0]==values[2] and values[0]!=values[1]


def test_report_two_arm_metadata_and_seed_aggregation(tmp_path,monkeypatch):
    import importlib.util,json
    from types import SimpleNamespace
    spec=importlib.util.spec_from_file_location('repeat_report_test',Path(__file__).with_name('report.py'));report=importlib.util.module_from_spec(spec);spec.loader.exec_module(report)
    base=tmp_path/'new';base.mkdir();original=tmp_path/'old';(original/'results').mkdir(parents=True)
    rows=[dict(fold=f,arm=a,train_cindex=.8,val_cindex=.7+(.01 if a=='freeze_c' else 0.)) for f in range(5) for a in ['baseline','freeze_c']]
    pd.DataFrame(rows).to_csv(original/'results/fold_metrics.csv',index=False)
    backend=SimpleNamespace(BASE=None,ARMS=None)
    def run_report():
        assert backend.ARMS==['baseline','freeze_c']
        out=backend.BASE/'results';out.mkdir(parents=True)
        pd.DataFrame(rows).to_csv(out/'fold_metrics.csv',index=False)
        (out/'validation.json').write_text(json.dumps(dict(passed=True,arms=3)))
        (out/'report.md').write_text('three branches;30final;two comparisons')
    backend.summarize=run_report
    monkeypatch.setattr(report,'BASE',base);monkeypatch.setattr(report,'ORIGINAL',original);monkeypatch.setattr(report,'load_backend',lambda *a:backend)
    report.summarize();result=pd.read_csv(base/'results/seed_summary.csv')
    assert result.seed.tolist()==[7,42,2026];np.testing.assert_allclose(result.mean_delta,.01)
    for seed in [7,42,2026]:assert json.loads((base/f'seed_{seed}/results/validation.json').read_text())['arms']==2
