"""Synthetic CPU regression checks; never train on the real CPTAC cohort."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
import warnings
from unittest.mock import patch

import h5py
import numpy as np
import pandas as pd
import torch
import yaml
from sksurv.metrics import concordance_index_censored


def module(name, filename):
    path = Path(__file__).with_name(filename)
    assert path.is_file(), f'Experiment09 implementation missing: {filename}'
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


class Experiment09Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        warnings.filterwarnings('ignore', category=DeprecationWarning, module=r'(lifelines\.plotting|sksurv\.metrics)')
        self.temporary = tempfile.TemporaryDirectory(prefix='pact09_')
        self.root = Path(self.temporary.name)
        self.source = self.root/'source'
        survival = self.source/'survival'; survival.mkdir(parents=True)
        base = self.source/'pathology/trident/20x_256px_0px_overlap'
        for folder in ['features_uni_v1', 'features_keep', 'patches']:
            (base/folder).mkdir(parents=True)
        rng = np.random.default_rng(43)
        patients, slides, benchmark = [], [], []
        for i in range(40):
            case, event, time, fold = f'case{i:03d}', (i//5)%2, 50+i*67, i%5
            patients.append(dict(case_id=case, OS_event=event, OS_days=time, time_unit='days'))
            for j in range(1+i%3):
                slide = f'{case}_slide{j}'; count = 3+j
                coords = np.arange(count*2).reshape(count, 2)
                for encoder, dimension in [('uni_v1',1024),('keep',768)]:
                    features = rng.normal(size=(count, dimension)).astype('float32')
                    if i==0: features[0] = 0
                    with h5py.File(base/f'features_{encoder}'/f'{slide}.h5','w') as f:
                        f['features'], f['coords'] = features, coords
                with h5py.File(base/'patches'/f'{slide}_patches.h5','w') as f:
                    f['coords'] = coords
                row = dict(case_id=case, slide_id=slide, file_name=f'{slide}.svs',
                           pathology_path=f'/unused/{slide}.svs', OS_event=event, OS_days=time)
                slides.append(row)
                benchmark.append({k:row[k] for k in ['case_id','slide_id','OS_event','OS_days']} |
                                 {'OS':7, **{f'fold_{k}':'test' if k==fold else 'train' for k in range(5)}})
        pd.DataFrame(patients).to_csv(survival/'patients.csv',index=False)
        pd.DataFrame(slides).to_csv(survival/'slides.csv',index=False)
        pd.DataFrame(benchmark).to_csv(survival/'PathoBench_CPTAC_LUAD_OS.tsv',sep='\t',index=False)

    def tearDown(self):
        self.temporary.cleanup()

    def prepare(self):
        m=module('cptac_dataset_test','cptac_dataset.py')
        with contextlib.redirect_stdout(io.StringIO()):
            data=m.prepare_cohort(self.source,self.root/'data')
        return m,data

    def test_selection_frozen_patient_level_and_three_inputs(self):
        m,data=self.prepare()
        self.assertEqual(len(data),40)
        self.assertEqual(data.case_id.nunique(),40)
        self.assertEqual(data.slide_id.nunique(),40)
        self.assertEqual(data.event.sum(),20)
        expected=pd.read_csv(self.source/'survival/patients.csv').set_index('case_id')
        np.testing.assert_array_equal(data.time,expected.loc[data.case_id,'OS_days'])
        self.assertEqual(data.split.tolist(),[i%5 for i in range(40)])
        before=(self.root/'data/all_label.csv').read_bytes()
        with contextlib.redirect_stdout(io.StringIO()):m.prepare_cohort(self.source,self.root/'data')
        self.assertEqual(before,(self.root/'data/all_label.csv').read_bytes())
        for feature_source,dimension in [('uni',1024),('uni_l2',1024),('keep',768)]:
            dataset=m.CPTACDataset(self.root/'data',feature_source)
            pd.testing.assert_frame_equal(dataset.samples,data)
            self.assertEqual(dataset.feature_dim,dimension)
            for i in [0,7,21]:
                features,event,time,case=dataset[i]
                self.assertEqual(features.shape[1],dimension)
                path=data.iloc[i].keep_path if feature_source=='keep' else data.iloc[i].uni_path
                with h5py.File(path) as f:raw=torch.from_numpy(f['features'][:])
                if feature_source=='uni_l2':
                    torch.testing.assert_close(features,raw/raw.norm(dim=-1,keepdim=True).clamp_min(1e-12))
                else:torch.testing.assert_close(features,raw,rtol=0,atol=0)
                self.assertEqual(case,data.iloc[i].case_id)

    def test_conflicting_slide_labels_and_fold_leakage_rejected(self):
        m=module('cptac_dataset_invalid','cptac_dataset.py')
        path=self.source/'survival/PathoBench_CPTAC_LUAD_OS.tsv'
        original=path.read_bytes(); frame=pd.read_csv(path,sep='\t')
        frame.loc[0,'OS_event']=1-frame.loc[0,'OS_event'];frame.to_csv(path,sep='\t',index=False)
        with self.assertRaises(ValueError):m.prepare_cohort(self.source,self.root/'bad_labels')
        path.write_bytes(original)
        frame=pd.read_csv(path,sep='\t'); index=frame.index[frame.case_id=='case001'][1]
        frame.loc[index,'fold_0']='test';frame.to_csv(path,sep='\t',index=False)
        with self.assertRaises(ValueError):m.prepare_cohort(self.source,self.root/'bad_folds')

    def test_selected_features_and_saved_assignment_cannot_silently_change(self):
        m,data=self.prepare()
        path=self.root/'data/all_label.csv'; original=path.read_bytes()
        changed=data.copy();changed.loc[0,'split']=(int(changed.loc[0,'split'])+1)%5
        changed.to_csv(path,index=False)
        with self.assertRaises(ValueError):m.CPTACDataset(self.root/'data','uni')
        path.write_bytes(original)
        with h5py.File(data.uni_path.iloc[0],'r+') as f:f['features'][0,0]=np.nan
        with self.assertRaises(ValueError):m.CPTACDataset(self.root/'data','uni')

    def test_three_fivefold_runs_and_checkpoint_reload(self):
        self.prepare();train=module('cptac_train_test','train.py')
        for source in ['uni','uni_l2','keep']:
            argv=['train.py','--data_dir',str(self.root/'data'),'--output_dir',str(self.root/'runs'),
                  '--feature_source',source,'--num_epochs','1','--num_workers','0','--cox_batch_size','8']
            with self.subTest(source=source),patch.object(sys,'argv',argv), \
                 patch.object(torch.cuda,'is_available',return_value=False),contextlib.redirect_stdout(io.StringIO()):
                train.main()
            result=next(p for p in (self.root/'runs/results').iterdir() if f'single_{source}_abmil' in p.name)
            config=yaml.safe_load((result/'run_config.yaml').read_text())
            self.assertEqual(config['feature_dim'],768 if source=='keep' else 1024)
            self.assertEqual(config['evaluation_protocol'],'internal_cv_with_validation_early_stopping')
            self.assertEqual(config['endpoint'],'OS');self.assertEqual(config['time_unit'],'days')
            self.assertEqual(config['normalization'],'per_patch_l2' if source=='uni_l2' else 'encoder_native' if source=='keep' else 'none')
            metric=pd.read_csv(result/'metrics.csv')
            self.assertEqual(metric.fold.astype(str).tolist(),['0','1','2','3','4','mean','std'])
            ids=[]
            for fold in range(5):
                val=pd.read_csv(result/f'fold_{fold}/val_predictions.csv')
                self.assertIn('OS_days',val);self.assertIn('OS_event',val)
                self.assertFalse(any(c.startswith('dfs.') for c in val))
                labels=pd.read_csv(self.source/'survival/patients.csv').set_index('case_id')
                np.testing.assert_array_equal(val.OS_days.to_numpy(np.float32),labels.loc[val.case_id,'OS_days'].to_numpy(np.float32))
                np.testing.assert_array_equal(val.OS_event.to_numpy(),labels.loc[val.case_id,'OS_event'].to_numpy())
                ci=concordance_index_censored(val.OS_event.to_numpy(bool),val.OS_days,val.risk_score)[0]
                self.assertAlmostEqual(ci,float(metric.loc[fold,'val_cindex']),places=12)
                training=pd.read_csv(result/f'fold_{fold}/train_predictions.csv')
                self.assertTrue(set(training.case_id).isdisjoint(val.case_id))
                ids.extend(val.case_id)
                self.assertTrue((result/f'fold_{fold}/km_high_vs_low.png').is_file())
            self.assertEqual(len(set(ids)),40)
            files=[result/'run_config.yaml',result/'data.csv']; before=[p.read_bytes()for p in files]
            with patch.object(sys,'argv',[*argv,'--eval_only']),patch.object(torch.cuda,'is_available',return_value=False),contextlib.redirect_stdout(io.StringIO()):
                train.main()
            self.assertEqual(before,[p.read_bytes()for p in files])
            checkpoint=self.root/'runs/checkpoints'/result.name/'fold_4/best_model.pth';checkpoint.unlink()
            with patch.object(sys,'argv',[*argv,'--eval_only']),patch.object(torch.cuda,'is_available',return_value=False):
                with self.assertRaises(FileNotFoundError):train.main()

    def test_patience5_has_separate_tag_and_stops_after_five_nonimprovements(self):
        train=module('cptac_train_patience_test','train.py')
        with patch.object(sys,'argv',['train.py']):args=train.parse_args()
        original=train.run_tag(args)
        args.patience=5
        self.assertNotEqual(train.run_tag(args),original)
        self.assertTrue(train.run_tag(args).endswith('_patience5'))
        args.survival_loss='cox';args.num_epochs=30;args.cox_batch_size=4
        directory=self.root/'early_stop';directory.mkdir()

        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__();self.head=torch.nn.Linear(16,1)
            def forward(self,x):
                return self.head(x.mean(dim=1)).flatten()

        loader=torch.utils.data.DataLoader(torch.utils.data.TensorDataset(
            torch.randn(8,3,16),torch.tensor([1,0]*4),torch.arange(1,9).float(),torch.arange(8)),batch_size=1)
        model=TinyModel();optimizer=torch.optim.Adam(model.parameters(),lr=1e-4)
        with patch('path_train.concordance_index_censored',return_value=(.6,0,0,0,0)),contextlib.redirect_stdout(io.StringIO()):
            train.train_path(model,loader,loader,optimizer,args,torch.device('cpu'),0,directory)
        saved=yaml.safe_load((directory/'checkpoint_metadata.yaml').read_text())
        self.assertEqual(saved['best_epoch'],1)
        self.assertEqual(saved['last_epoch'],6)

    def test_runner_starts_three_processes_before_waiting_and_keeps_other_runs(self):
        runner=module('cptac_runner_test','run_all.py')
        argv=['run_all.py','--output_dir',str(self.root/'runner'),'--patience','5']
        commands=[]

        class Process:
            def __init__(self,command,**kwargs):
                commands.append(command)
                self.pid=100+len(commands)
                self.source=command[-1]
            def wait(process):
                self.assertEqual(len(commands),3,'All three sources must start before waiting')
                return 0

        with patch.object(sys,'argv',argv),patch.object(runner.subprocess,'Popen',side_effect=Process),contextlib.redirect_stdout(io.StringIO()):
            runner.main()
        self.assertEqual([c[-1]for c in commands],['uni','uni_l2','keep'])
        self.assertEqual(len(list((self.root/'runner/logs').glob('*.log'))),3)
        self.assertTrue(all('_patience5' in p.name for p in (self.root/'runner/logs').glob('*.log')))
        self.assertTrue(all(c[c.index('--patience')+1]=='5'for c in commands))
        commands=[]
        finished=[]

        class FailedProcess(Process):
            def wait(process):
                self.assertEqual(len(commands),3)
                finished.append(process.source)
                return 7 if process.source=='uni_l2' else 0

        with patch.object(sys,'argv',argv),patch.object(runner.subprocess,'Popen',side_effect=FailedProcess),contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit)as caught:runner.main()
        self.assertEqual(caught.exception.code,7)
        self.assertEqual(set(finished),{'uni','uni_l2','keep'})

    def test_real_subprocess_barrier_proves_parallel_launch_without_training(self):
        runner=module('cptac_parallel_subprocess_test','run_all.py')
        fake=self.root/'fake_runner';fake.mkdir()
        (fake/'train.py').write_text('''import argparse,time\nfrom pathlib import Path\np=argparse.ArgumentParser()\np.add_argument('--output_dir')\np.add_argument('--feature_source')\na,_=p.parse_known_args()\nr=Path(a.output_dir);r.mkdir(exist_ok=True,parents=True)\n(r/(a.feature_source+'.started')).write_text('started')\nlimit=time.monotonic()+5\nwhile not all((r/(s+'.started')).exists() for s in ['uni','uni_l2','keep']):\n if time.monotonic()>limit:raise SystemExit(9)\n time.sleep(.02)\n(r/(a.feature_source+'.finished')).write_text('finished')\nprint('parallel barrier passed',flush=True)\nraise SystemExit(7 if a.feature_source=='uni_l2' and (r/'fail_uni_l2').exists() else 0)\n''')
        for fail in [False,True]:
            output=self.root/('parallel_failure'if fail else'parallel_success');output.mkdir()
            if fail:(output/'fail_uni_l2').touch()
            argv=['run_all.py','--output_dir',str(output),'--patience','5']
            with patch.object(runner,'__file__',str(fake/'run_all.py')),patch.object(sys,'argv',argv),contextlib.redirect_stdout(io.StringIO()):
                if fail:
                    with self.assertRaises(SystemExit)as caught:runner.main()
                    self.assertEqual(caught.exception.code,7)
                else:runner.main()
            self.assertEqual(len(list(output.glob('*.finished'))),3)
            self.assertTrue(all('parallel barrier passed'in p.read_text()for p in (output/'logs').glob('*.log')))


if __name__=='__main__':
    unittest.main()
