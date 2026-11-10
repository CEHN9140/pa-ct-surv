"""CPU checks for normalizing Cox by actual patient count in experiment09."""
import contextlib
import io
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
import yaml
from sksurv.metrics import concordance_index_censored

import test_experiment as existing


class PatientMeanTests(unittest.TestCase):
    def test_patient_loss_matches_dense_breslow_and_gradients(self):
        loss_module=existing.module('patient_cox_unit','patient_cox_loss.py')
        for count in [16,20,84]:
            risk=torch.linspace(-1,2,count,dtype=torch.float64,requires_grad=True)
            time=(torch.arange(count)//2).double()+1
            event=(torch.arange(count)%3==0)
            matrix=risk[None,:].expand(count,-1).masked_fill(time[None,:]<time[:,None],-torch.inf)
            expected=(torch.logsumexp(matrix,dim=1)-risk)[event].sum()/count
            actual=loss_module.cox_patient_loss(risk,time,event)
            torch.testing.assert_close(actual,expected)
            actual_gradient=torch.autograd.grad(actual,risk,retain_graph=True)[0]
            expected_gradient=torch.autograd.grad(expected,risk)[0]
            torch.testing.assert_close(actual_gradient,expected_gradient)
        risk=torch.randn(16,requires_grad=True)
        zero=loss_module.cox_patient_loss(risk,torch.arange(16).float(),torch.zeros(16))
        self.assertEqual(zero.item(),0)
        zero.backward()
        torch.testing.assert_close(risk.grad,torch.zeros_like(risk))

    def test_three_sources_train_with_patient_reduction_and_reload(self):
        fixture=existing.Experiment09Tests();fixture.setUp();self.addCleanup(fixture.tearDown)
        fixture.prepare();root=fixture.root
        train=existing.module('patient_cox_training_test','train.py')
        original=train.cox_patient_loss;calls=[]

        def tracked(risk,time,event):
            calls.append((risk.numel(),risk.requires_grad))
            return original(risk,time,event)

        for source in ['uni','uni_l2','keep']:
            argv=['train.py','--data_dir',str(root/'data'),'--output_dir',str(root/'runs'),
                  '--feature_source',source,'--cox_reduction','patients','--num_epochs','1',
                  '--num_workers','0','--cox_batch_size','12']
            with patch.object(sys,'argv',argv),patch.object(torch.cuda,'is_available',return_value=False), \
                 patch.object(train,'cox_patient_loss',side_effect=tracked),contextlib.redirect_stdout(io.StringIO()):
                train.main()
            result=next(p for p in (root/'runs/results').iterdir()if f'single_{source}_'in p.name)
            self.assertIn('_abmil_cox_patientmean_',result.name)
            config=yaml.safe_load((result/'run_config.yaml').read_text())
            self.assertEqual(config['cox_reduction'],'patients')
            self.assertEqual(config['cox_batch_size'],12)
            self.assertEqual(config['pa_model'],'abmil')
            metrics=pd.read_csv(result/'metrics.csv');seen=[]
            for fold in range(5):
                for part in ['train','val']:
                    predictions=pd.read_csv(result/f'fold_{fold}/{part}_predictions.csv')
                    risk=torch.tensor(predictions.risk_score.to_numpy(),dtype=torch.float32)
                    time=torch.tensor(predictions.OS_days.to_numpy(),dtype=torch.float32)
                    event=torch.tensor(predictions.OS_event.to_numpy(),dtype=torch.float32)
                    self.assertAlmostEqual(original(risk,time,event).item(),metrics.loc[fold,f'{part}_loss'],places=5)
                    ci=concordance_index_censored(event.numpy().astype(bool),time.numpy(),risk.numpy())[0]
                    self.assertAlmostEqual(ci,metrics.loc[fold,f'{part}_cindex'])
                    if part=='val':seen.extend(predictions.case_id)
            self.assertEqual(len(seen),len(set(seen)))
            before=[(result/n).read_bytes()for n in ['data.csv','run_config.yaml']]
            with patch.object(sys,'argv',[*argv,'--eval_only']),patch.object(torch.cuda,'is_available',return_value=False), \
                 contextlib.redirect_stdout(io.StringIO()):train.main()
            self.assertEqual(before,[(result/n).read_bytes()for n in ['data.csv','run_config.yaml']])
        training_sizes=[n for n,grad in calls if grad]
        self.assertEqual(len(training_sizes),30)
        self.assertEqual(set(training_sizes),{12,20})
        self.assertEqual(len(list((root/'runs/results').iterdir())),3)

    def test_runner_forwards_reduction_to_all_three_processes(self):
        runner=existing.module('patient_cox_runner_test','run_all.py')
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            commands=[]

            class Process:
                def __init__(self,command,**kwargs):
                    commands.append(command);self.pid=len(commands)
                def wait(process):
                    self.assertEqual(len(commands),3)
                    return 0

            argv=['run_all.py','--output_dir',directory,'--cox_reduction','patients']
            with patch.object(sys,'argv',argv),patch.object(runner.subprocess,'Popen',side_effect=Process), \
                 contextlib.redirect_stdout(io.StringIO()):runner.main()
            self.assertEqual([c[-1]for c in commands],['uni','uni_l2','keep'])
            self.assertTrue(all(c[c.index('--cox_reduction')+1]=='patients'for c in commands))
            self.assertTrue(all('cox_patientmean_'in p.name for p in (Path(directory)/'logs').glob('*.log')))


if __name__=='__main__':unittest.main()
