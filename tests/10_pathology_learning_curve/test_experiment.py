"""Temporary CPU data only; never run real learning-curve training."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np
import pandas as pd
import torch
from sksurv.metrics import concordance_index_censored

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
sys.path.insert(0,str(Path(__file__).resolve().parent))


def module(name,filename):
    path=Path(__file__).with_name(filename)
    assert path.is_file(),f'Experiment10 implementation missing: {filename}'
    spec=importlib.util.spec_from_file_location(name,path)
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result)
    return result


class LearningCurveTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temporary=tempfile.TemporaryDirectory(prefix='pact10_')
        self.addCleanup(self.temporary.cleanup)
        self.root=Path(self.temporary.name);source=self.root/'source';source.mkdir()
        rng=np.random.default_rng(19);rows=[]
        for i in range(20):
            path=source/f'case{i}.h5'
            with h5py.File(path,'w')as f:f['features']=rng.normal(size=(4,1024)).astype(np.float32)
            rows.append(dict(pa_id=f'case{i}',pa_path=str(path),event=(i//5)%2,time=10.+i*7,split=i%5))
        self.source=source/'all_label_roi64.csv';pd.DataFrame(rows).to_csv(self.source,index=False)

    def test_nested_subsets_frozen_and_no_validation_leakage(self):
        data=module('curve_data_test','learning_data.py')
        dataset,selected,validation,metadata=data.prepare_subsets(self.source,self.root/'data',[4,8],2)
        self.assertEqual(len(selected.groupby(['fold','size_label','repeat'])),25)
        for fold in range(5):
            val=set(validation.loc[validation.fold.eq(fold),'pa_id'])
            self.assertEqual(val,set(dataset.samples.loc[dataset.samples.split.eq(fold),'pa_id']))
            for repeat in range(2):
                smaller=set(selected.loc[selected.fold.eq(fold)&selected['repeat'].eq(repeat)&selected.size_label.eq('n4'),'pa_id'])
                larger=set(selected.loc[selected.fold.eq(fold)&selected['repeat'].eq(repeat)&selected.size_label.eq('n8'),'pa_id'])
                self.assertEqual(len(smaller),4);self.assertEqual(len(larger),8)
                self.assertTrue(smaller<larger)
                self.assertTrue(val.isdisjoint(larger))
        before={p.name:p.read_bytes()for p in (self.root/'data').iterdir()}
        data.prepare_subsets(self.source,self.root/'data',[4,8],2)
        self.assertEqual(before,{p.name:p.read_bytes()for p in (self.root/'data').iterdir()})
        with self.assertRaises(ValueError):data.prepare_subsets(self.source,self.root/'data',[4,12],2)
        changed=pd.read_csv(self.source);changed.loc[0,'event']=1-changed.loc[0,'event'];changed.to_csv(self.source,index=False)
        with self.assertRaises(ValueError):data.prepare_subsets(self.source,self.root/'data',[4,8],2)

    def test_fixed_update_budget_and_all_fold_outputs(self):
        train=module('curve_training_test','train.py')
        argv=['train.py','--label_csv',str(self.source),'--output_dir',str(self.root/'experiment'),
              '--train_sizes','4','8','--repeats','2','--cox_batch_size','4',
              '--max_updates','4','--eval_every','2','--num_workers','0']
        with patch.object(sys,'argv',argv),patch.object(torch.cuda,'is_available',return_value=False), \
             contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):train.main()
        result=next((self.root/'experiment/results').iterdir())
        metrics=pd.read_csv(result/'metrics.csv')
        self.assertEqual(len(metrics),25)
        self.assertTrue(metrics.optimizer_steps.eq(4).all())
        self.assertTrue(metrics.validation_checks.eq(2).all())
        self.assertEqual(metrics.initial_state_sha256.nunique(),1)
        self.assertEqual(metrics.groupby('size_label').size().to_dict(),{'full':5,'n4':10,'n8':10})
        self.assertTrue((result/'learning_curve.png').is_file())
        for row in metrics.itertuples():
            directory=result/row.run_name/f'fold_{row.fold}'
            history=pd.read_csv(directory/'update_metrics.csv')
            self.assertEqual(history['step'].tolist(),[2,4])
            for prefix in ['', 'last/']:
                val=pd.read_csv(directory/prefix/'val_predictions.csv')
                self.assertEqual(set(val.case_id),set(pd.read_csv(self.source).loc[lambda x:x.split.eq(row.fold),'pa_id']))
                ci=concordance_index_censored(val['dfs.status'].astype(bool),val['dfs.month'],val.risk_score)[0]
                self.assertAlmostEqual(ci,getattr(row,'last_val_cindex'if prefix else'val_cindex'))
            train_predictions=pd.read_csv(directory/'train_predictions.csv')
            self.assertEqual(len(train_predictions),row.n_train)
            self.assertTrue(set(train_predictions.case_id).isdisjoint(val.case_id))
            checkpoint=self.root/'experiment/checkpoints'/result.name/row.run_name/f'fold_{row.fold}'
            self.assertTrue((checkpoint/'best_model.pth').is_file())
            self.assertTrue((checkpoint/'last_model.pth').is_file())

    def test_summary_averages_draws_within_each_fold_first(self):
        analysis=module('curve_analysis_test','analysis.py')
        records=[]
        for fold,values in [(0,[.6,.8]),(1,[.2,.4])]:
            for repeat,value in enumerate(values):
                records.append(dict(fold=fold,repeat=repeat,size_label='n4',n_train=4,
                                    val_cindex=value,last_val_cindex=value,last_train_cindex=.9,
                                    last_gap=.9-value,last_auc_36m=value,last_auc_60m=value))
        summary=analysis.curve_summary(pd.DataFrame(records)).iloc[0]
        self.assertAlmostEqual(summary.last_val_cindex_mean,.5)
        self.assertAlmostEqual(summary.last_val_cindex_std,np.std([.7,.3],ddof=1))
        self.assertEqual(summary.n_folds,2)
        records[1]['last_auc_36m']=np.nan
        incomplete=analysis.curve_summary(pd.DataFrame(records)).iloc[0]
        self.assertTrue(np.isnan(incomplete.last_auc_36m_mean))

    def test_existing_run_cannot_be_overwritten(self):
        train=module('curve_overwrite_test','train.py')
        output=self.root/'experiment'
        tag='uni_abmil_cox_dropout0.25_updates4_eval2_coxbs4_lr0.0001_wd0.0005_seed123'
        previous=output/'results'/tag;previous.mkdir(parents=True)
        (previous/'run_config.yaml').write_text('existing: true\n')
        argv=['train.py','--label_csv',str(self.source),'--output_dir',str(output),
              '--train_sizes','4','8','--repeats','2','--cox_batch_size','4','--max_updates','4','--eval_every','2']
        with patch.object(sys,'argv',argv),contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(FileExistsError):train.main()
        self.assertEqual((previous/'run_config.yaml').read_text(),'existing: true\n')

    def test_same_output_directory_cannot_run_concurrently(self):
        import fcntl
        train=module('curve_lock_test','train.py');output=self.root/'experiment';output.mkdir()
        with (output/'.run.lock').open('a')as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            argv=['train.py','--label_csv',str(self.source),'--output_dir',str(output),'--prepare_only',
                  '--train_sizes','4','8','--repeats','2','--cox_batch_size','4']
            with patch.object(sys,'argv',argv):
                with self.assertRaisesRegex(RuntimeError,'already running'):train.main()


if __name__=='__main__':unittest.main()
