import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sksurv.metrics import concordance_index_censored

sys.path.insert(0, str(Path(__file__).parent))


def test_pair_scores_match_sksurv_including_time_ties_and_risk_tolerance():
    from pairs import pair_changes
    t = np.array([2., 2., 2., 4., 7., 9.])
    e = np.array([1, 1, 0, 1, 0, 0])
    a = np.array([1., 1.+1e-9, 1., .5, .1, .2])
    b = np.array([.1, .2, .3, .5, 1., 2.])
    pairs, patients, summary = pair_changes(t, e, a, b)
    ca = concordance_index_censored(e.astype(bool), t, a)
    cb = concordance_index_censored(e.astype(bool), t, b)
    assert summary['best_cindex'] == ca[0]
    assert summary['last_cindex'] == cb[0]
    assert summary['pairs'] == ca[1] + ca[2] + ca[3]
    assert summary['delta_cindex'] == pytest.approx(cb[0]-ca[0])
    assert summary['net_credit'] / len(pairs) == pytest.approx(cb[0]-ca[0])
    assert not ((pairs.i.eq(0) & pairs.j.eq(1)) | (pairs.i.eq(1) & pairs.j.eq(0))).any()
    assert len(pairs[pairs.pair_type.eq('event_same_time_censor')]) == 2
    assert (patients.comparable_pairs.sum() == 2 * len(pairs))
    assert patients.error_credit_best.sum() == pytest.approx(2*(1-ca[0])*len(pairs))


def test_corrected_broken_and_half_credit_are_separate():
    from pairs import pair_changes
    p, patients, row = pair_changes(np.array([1.,2.,3.]), np.array([1,0,0]),
                                   np.array([1.,0.,2.]), np.array([0.,1.,0.]))
    assert row['broken_full'] == 1
    assert row['corrected_full'] == 0
    assert row['wrong_to_tie'] == 1
    assert row['net_credit'] == -.5
    assert row['delta_cindex'] == -.25
    assert np.isfinite(p.best_gap_sd).all()


def test_time_bins_include_boundaries_and_empty_patient_error_is_nan():
    from pairs import pair_changes
    p, patients, row = pair_changes(np.array([12.,24.,36.,40.,60.,0.]),
                                   np.array([1,1,1,1,0,0]), np.arange(6.), -np.arange(6.))
    assert set(p.event_time_bin) == {'0-12','12-24','24-36','>36'}
    assert patients.loc[5,'comparable_pairs'] == 0
    assert np.isnan(patients.loc[5,'error_rate_best'])


def test_concentration_counts_union_of_pairs_once():
    from pairs import pair_changes, concentration
    p, patients, row = pair_changes(np.array([1.,2.,3.,4.]), np.array([1,1,1,0]),
                                   np.array([4.,3.,2.,1.]), np.array([1.,2.,3.,4.]))
    result = concentration(p, patients, fractions=(.5, 1.))
    assert result.iloc[-1].broken_pair_coverage == 1.
    assert result.iloc[-1].patients_selected == 4
    assert result.iloc[0].broken_pairs_covered <= row['broken_full']
    assert result.iloc[0].broken_pairs_covered == 5


def test_prediction_alignment_uses_float32_labels_and_rejects_mismatch(tmp_path):
    from diagnose_late import read_predictions
    cohort = pd.DataFrame(dict(pa_id=['a','b'], time=[1/3,7.], event=[1,0]))
    saved = pd.DataFrame({'case_id':['b','a'], 'dfs.month':np.array([7.,1/3],np.float32),
                          'dfs.status':[0,1], 'risk_score':[.1,.5]})
    path=tmp_path/'pred.csv';saved.to_csv(path,index=False)
    a = read_predictions(path, cohort)
    assert a.case_id.tolist() == ['a','b']
    saved.loc[0,'dfs.status']=1;saved.to_csv(path,index=False)
    with pytest.raises(ValueError):
        read_predictions(path,cohort)


def test_learning_summary_averages_repeats_within_fold_first():
    from diagnose_late import learning_summary
    rows=[]
    for fold in range(5):
        for repeat in range(3):
            rows.append(dict(size_label='n128',fold=fold,repeat=repeat,n_train=128,
                             val_cindex=.5+fold*.01+repeat*.1,train_cindex=.8,
                             last_val_cindex=.4+fold*.01,last_train_cindex=.95,
                             training_patient_exposures=11520,best_step=9+repeat*9,
                             n_events_train=44))
    summary, folds = learning_summary(pd.DataFrame(rows))
    assert len(folds)==5
    assert summary.iloc[0].best_val_cindex_mean == pytest.approx(.62)
    assert summary.iloc[0].best_val_cindex_sd == pytest.approx(np.std([.6,.61,.62,.63,.64],ddof=1))
    assert summary.iloc[0].patient_exposure_epochs_mean == 90.
    np.testing.assert_allclose(folds.within_repeat_best_val_sd,.1)
    assert folds.n_repeats.eq(3).all()


def test_randomized_ties_and_patient_reordering_preserve_cindex():
    from pairs import pair_changes
    rng=np.random.default_rng(123)
    for _ in range(100):
        t=rng.integers(1,10,20).astype(float)
        e=rng.integers(0,2,20);e[0]=1;t[0]=0
        a=rng.integers(0,5,20).astype(float)+rng.choice([0,1e-9],20)
        b=rng.integers(0,5,20).astype(float)
        _,_,row=pair_changes(t,e,a,b)
        index=rng.permutation(20)
        _,_,other=pair_changes(t[index],e[index],a[index],b[index])
        assert row==other


def test_refuse_overwrite_before_reading_sources(tmp_path):
    from diagnose_late import run_analysis
    (tmp_path/'report.md').write_text('Existing output')
    with pytest.raises(FileExistsError):
        run_analysis('/does/not/exist',tmp_path)
