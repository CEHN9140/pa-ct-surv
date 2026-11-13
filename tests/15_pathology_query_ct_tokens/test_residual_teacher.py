import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import torch
from sksurv.metrics import concordance_index_censored


class ResidualTeacherTests(unittest.TestCase):
    def api(self):
        from residual_teacher import ResidualTeacher, residual_objective, pair_changes
        return ResidualTeacher, residual_objective, pair_changes

    def test_zero_initialization_matches_own_pathology_risk_and_bn_stays_fixed(self):
        Model,_,_=self.api()
        model=Model(dropout=0.)
        model.train()
        before={n:v.clone() for n,v in model.ct_backbone.named_buffers()}
        total,delta,path=model(torch.rand(3,1,16,16,16),torch.randn(3,25,1280))
        torch.testing.assert_close(total,path,rtol=0,atol=0)
        torch.testing.assert_close(delta,torch.zeros(3),rtol=0,atol=0)
        self.assertTrue(all(p.requires_grad for p in model.parameters()))
        self.assertTrue(model.pa_branch.training)
        for n,v in model.ct_backbone.named_buffers():
            torch.testing.assert_close(v,before[n],rtol=0,atol=0)

    def test_exact_objective_and_batch_mean_penalty(self):
        _,objective,_=self.api()
        from loss import cox_loss
        p=torch.tensor([.8,.2,-.4],requires_grad=True)
        d=torch.tensor([.2,-.3,.1],requires_grad=True)
        t=torch.tensor([1.,2.,3.]);e=torch.tensor([1,0,1])
        values=objective(p+d,d,p,t,e,.3,.01)
        expected=cox_loss(p+d,t,e)+.3*cox_loss(p,t,e)+.01*d.square().mean()
        torch.testing.assert_close(values['total'],expected)
        torch.testing.assert_close(values['correction_l2'],d.square().mean())
        values['total'].backward()
        self.assertGreater(p.grad.abs().sum(),0)
        self.assertGreater(d.grad.abs().sum(),0)

    def test_both_encoders_can_update_after_zero_head_first_step(self):
        Model,objective,_=self.api()
        torch.manual_seed(123)
        model=Model(dropout=0.).train()
        optimizer=torch.optim.Adam(model.parameters(),lr=1e-3)
        ct=torch.rand(4,1,16,16,16);pa=torch.randn(4,25,1280)
        t=torch.arange(1.,5.);e=torch.tensor([1,1,0,1])
        before=model.ct_backbone.conv1.weight.detach().clone()
        for step in range(2):
            optimizer.zero_grad()
            r,d,p=model(ct,pa)
            objective(r,d,p,t,e,.3,.01)['total'].backward()
            ct_grad=model.ct_backbone.conv1.weight.grad.abs().sum().item()
            if step==0:self.assertEqual(ct_grad,0.)
            else:self.assertGreater(ct_grad,0.)
            self.assertGreater(model.pa_branch.compression[0].weight.grad.abs().sum(),0)
            optimizer.step()
        self.assertFalse(torch.equal(before,model.ct_backbone.conv1.weight))

    def test_pair_counts_follow_event_censor_and_tie_rules(self):
        _,_,changes=self.api()
        t=np.array([1.,1.,2.,3.]);e=np.array([1,0,1,1])
        p=np.array([0.,1.,2.,3.]);r=np.array([4.,1.,2.,3.])
        result=changes(t,e,p,r)
        self.assertEqual(result['comparable_pairs'],4)
        self.assertEqual(result['corrected_pairs'],3)
        self.assertEqual(result['broken_pairs'],0)
        expected=concordance_index_censored(e.astype(bool),t,r)[0]-concordance_index_censored(e.astype(bool),t,p)[0]
        self.assertAlmostEqual(result['delta_cindex'],expected)
        reverse=changes(t,e,r,p)
        self.assertEqual(reverse['broken_pairs'],3)
        tied=changes(t,e,p,np.zeros(4))
        self.assertEqual(tied['improved_pairs'],4)
        self.assertAlmostEqual(tied['delta_cindex'],.5)

    def test_penalty_selection_uses_inner_scores_and_median_epochs(self):
        from residual_train import select_residual_config
        records=[]
        for name,weight,scores,epochs in [('residual_r0',0.,[.6,.7,.8],[1,5,3]),
                                         ('residual_r0p01',.01,[.7,.7,.8],[4,6,8])]:
            for fold,score in enumerate(scores):
                records.append(dict(name=name,lambda_r=weight,inner_fold=fold,
                                    held_cindex=score,selected_epoch=epochs[fold]))
        choice=select_residual_config(records)
        self.assertEqual(choice['name'],'residual_r0p01')
        self.assertEqual(choice['epochs'],6)
        with self.assertRaises(ValueError):select_residual_config(records[:-1])

    def test_source_roles_exclude_outer_validation_and_cover_inner_oof_once(self):
        from residual_train import source_partitions
        samples=pd.DataFrame(dict(pa_id=[str(i) for i in range(7)],ct_id=[str(i+10) for i in range(7)]))
        rows=[]
        for inner in range(3):
            for role,indices in [('oof',[2*inner,2*inner+1]),
                                 ('stop',[(2*inner+2)%6,(2*inner+3)%6]),
                                 ('fit',[(2*inner+4)%6,(2*inner+5)%6])]:
                for i in indices:rows.append(dict(inner_fold=inner,role=role,case_id=f'{i}|{i+10}'))
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'roles.csv'
            pd.DataFrame(rows).to_csv(path,index=False)
            result=source_partitions(samples,np.arange(6),np.array([6]),path)
            self.assertEqual(len(result),3)
            self.assertEqual(set(np.concatenate([x[3] for x in result])),set(range(6)))
            rows[0]['case_id']='6|16';pd.DataFrame(rows).to_csv(path,index=False)
            with self.assertRaises(ValueError):source_partitions(samples,np.arange(6),np.array([6]),path)

    def test_training_consumes_tail_and_fixed_refit_saves_reloadable_model(self):
        from residual_train import fit_teacher,predict_teacher
        from torch.utils.data import DataLoader,TensorDataset
        Model,_,_=self.api()
        rng=torch.Generator().manual_seed(123)
        data=TensorDataset(torch.rand(7,1,16,16,16,generator=rng),
            torch.randn(7,25,1280,generator=rng),torch.ones(7,dtype=torch.long),
            torch.arange(1.,8.),torch.arange(7))
        args=SimpleNamespace(seed=123,ct_pretrained_path=None,dropout=0.,lr=1e-3,
            ct_backbone_lr=1e-4,weight_decay=0.,num_workers=0,cox_batch_size=4,
            num_epochs=1,patience=1,lambda_pa=.3)
        with tempfile.TemporaryDirectory() as folder:
            model,metadata=fit_teacher(data,np.arange(7),None,args,torch.device('cpu'),
                                       folder,.01,fixed_epochs=1)
            self.assertEqual(metadata['cox_group_sizes'],[7])
            history=pd.read_csv(Path(folder)/'training_history.csv')
            self.assertEqual(history.optimizer_updates.tolist(),[1])
            frame=predict_teacher(model,DataLoader(data,batch_size=1),'cpu')
            restored=Model(dropout=0.)
            restored.load_state_dict(torch.load(Path(folder)/'best_model.pth',weights_only=True),strict=True)
            again=predict_teacher(restored,DataLoader(data,batch_size=1),'cpu')
            np.testing.assert_allclose(frame.risk_score,again.risk_score,rtol=0,atol=0)
            with self.assertRaises(ValueError):
                fit_teacher(data,np.arange(7),None,args,torch.device('cpu'),folder,.01)

    def test_outer_export_keeps_existing_inner_artifacts(self):
        from residual_train import export_teacher
        def frame(prefix,n):
            t=np.arange(1.,n+1)*6
            return pd.DataFrame(dict(case_id=[f'{prefix}{i}' for i in range(n)],
                **{'dfs.month':t,'dfs.status':np.ones(n,dtype=int)},
                risk_score=-t+.1,pathology_risk=-t,ct_correction=np.full(n,.1)))
        frames={'train':frame('t',16),'val':frame('v',8)}
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            reference=root/'source/pathology/fold_0';reference.mkdir(parents=True)
            frames['val'].assign(risk_score=frames['val'].pathology_risk).to_csv(reference/'val_predictions.csv',index=False)
            inner=root/'results/residual_r0/fold_0/inner_0';inner.mkdir(parents=True)
            (inner/'marker.txt').write_text('preserve')
            args=SimpleNamespace(results_root=str(root/'results'),source_run=str(root/'source'),lambda_pa=.3)
            row,prediction=export_teacher('residual_r0',0,frames,0.,args)
            self.assertEqual(row['val_cindex'],1.)
            self.assertEqual((inner/'marker.txt').read_text(),'preserve')
            self.assertEqual(len(prediction),8)
            self.assertTrue((inner.parent/'pair_changes.json').is_file())


if __name__=='__main__':
    torch.set_num_threads(2)
    unittest.main()
