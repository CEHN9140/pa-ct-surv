import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))


class PhenotypeFollowupTests(unittest.TestCase):
    def data(self):
        rng=np.random.default_rng(123)
        x=rng.normal(size=(120,10)).astype(np.float32)
        y=np.column_stack([2*x[:,0]+3,100*x[:,1]-50])
        return x,y

    def test_probe_recovers_phenotypes_without_fitting_validation_scaler(self):
        from phenotype_probe import fit_probe, evaluate_probe
        x,y=self.data();head=fit_probe(x[:80],y[:80],[.01,1.])
        self.assertEqual(head.named_steps['ridge'].solver,'svd')
        self.assertEqual(head.named_steps['ridge'].coef_.dtype,np.dtype('float64'))
        np.testing.assert_allclose(head.named_steps['scale'].mean_,x[:80].astype(float).mean(0))
        before=head.named_steps['scale'].mean_.copy()
        pred=head.predict(x[80:])
        scores=evaluate_probe(y[:80],y[80:],pred,['a','b'])
        self.assertGreater(scores[0]['r2'],.999)
        self.assertGreater(scores[1]['r2'],.99)
        self.assertAlmostEqual(scores[0]['mae'],mean_absolute_error(y[80:,0],pred[:,0]))
        head.predict(x[80:]+1e5)
        np.testing.assert_array_equal(before,head.named_steps['scale'].mean_)

    def test_targetwise_tuning_and_frozen_target_mean_baseline(self):
        from phenotype_probe import select_probe, evaluate_probe
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import Ridge
        x,y=self.data()
        context=dict(train_ids=list(range(80)),val_ids=list(range(80,120)),
                     train_ct=x[:80],val_ct=x[80:],train_targets=y[:80],val_targets=y[80:])
        selected=select_probe([context],[.01,1.,100.],['a','b'])
        manual=[]
        for penalty in [.01,1.,100.]:
            model=make_pipeline(StandardScaler(),Ridge(alpha=penalty,solver='svd')).fit(x[:80].astype(float),y[:80])
            manual.append(np.mean((model.predict(x[80:])-y[80:])**2,axis=0))
        expected=np.array([.01,1.,100.])[np.argmin(manual,axis=0)]
        np.testing.assert_array_equal(selected['alphas'],expected)
        truth=np.array([[100.,200.],[101.,202.]])
        stats=evaluate_probe(y[:80],truth,np.zeros_like(truth),['a','b'])
        self.assertAlmostEqual(stats[0]['baseline_mae'],np.mean(abs(truth[:,0]-y[:80,0].astype(float).mean())))
        self.assertAlmostEqual(stats[0]['r2'],r2_score(truth[:,0],[0.,0.]))
        with self.assertRaisesRegex(ValueError,'overlap'):
            select_probe([dict(context,val_ids=[0,81])],[1.],['a','b'])

    def test_target_columns_are_exact_shape_plus_firstorder(self):
        from phenotype_followup import phenotype_targets
        targets,units=phenotype_targets()
        self.assertEqual(len(targets),9)
        self.assertEqual(targets[:3],['log_volume_mm3','diameter_mm','sphericity'])
        self.assertEqual(targets[3:],['tumor_mean','tumor_std','tumor_p10','tumor_median','tumor_p90','tumor_entropy'])
        self.assertEqual(len(units),9)

    def test_cached_features_are_identity_checked_and_never_recomputed(self):
        from phenotype_followup import read_feature_cache
        import hashlib
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);cache=root/'ct.npz';checkpoint=root/'public.pth';checkpoint.write_bytes(b'public')
            metadata=dict(checkpoint=str(checkpoint),checkpoint_sha256=hashlib.sha256(b'public').hexdigest(),
                          cohort_sha256='cohort',patients=3,feature_dim=512,source='public only',
                          computation='layer4 global average; before risk head; eval; no L2; BN unchanged')
            cache.with_suffix('.json').write_text(json.dumps(metadata))
            np.savez(cache,case_id=['a','b','c'],features=np.ones((3,512)))
            frame=read_feature_cache(cache,['a','b','c'],'cohort',True,{})
            self.assertEqual(frame.shape,(3,512))
            with self.assertRaisesRegex(ValueError,'identity'):
                read_feature_cache(cache,['a','b','d'],'cohort',True,{})
            np.savez(cache,case_id=['a','b','c'],features=np.full((3,512),np.nan))
            with self.assertRaisesRegex(ValueError,'finite'):
                read_feature_cache(cache,['a','b','c'],'cohort',True,{})
            with self.assertRaises(FileNotFoundError):
                read_feature_cache(root/'missing.npz',['a'],'cohort',True,{})
            checkpoint.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'checkpoint hash'):
                read_feature_cache(cache,['a','b','c'],'cohort',True,{})

    def test_ct_encoder_roles_must_exclude_held_and_outer_patients(self):
        from phenotype_followup import validate_ct_roles
        training=[f'c{i}' for i in range(12)]
        oof=pd.DataFrame({'inner_fold':np.arange(12)%3},index=training)
        bundle={'oof':oof,'val':pd.DataFrame(index=['outer'])}
        contexts=[];roles=[]
        for inner in range(3):
            held=oof.index[oof.inner_fold.eq(inner)].tolist()
            train=oof.index[~oof.inner_fold.eq(inner)].tolist()
            contexts.append(dict(train_ids=train,val_ids=held))
            for role,ids in [('fit',train[:6]),('stop',train[6:]),('oof',held)]:
                roles.extend(dict(inner_fold=inner,case_id=x,role=role) for x in ids)
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory)/'fold_0';folder.mkdir()
            file=folder/'inner_split_roles.csv'
            pd.DataFrame(roles).to_csv(file,index=False)
            self.assertEqual(validate_ct_roles(0,bundle,contexts,Path(directory)),file)
            broken=pd.DataFrame(roles);broken.loc[0,'case_id']='outer';broken.to_csv(file,index=False)
            with self.assertRaisesRegex(ValueError,'isolation'):
                validate_ct_roles(0,bundle,contexts,Path(directory))


if __name__=='__main__':unittest.main()
