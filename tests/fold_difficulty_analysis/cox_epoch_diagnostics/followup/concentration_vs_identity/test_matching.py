import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
from matching import effective,rank_preserving_profile,match_effective,patch_stats


def test_rank_and_profile_are_independently_controlled():
    late=np.array([.7,.2,.08,.02]);mixed=np.array([.35,.1,.5,.05])
    output=rank_preserving_profile(late,mixed)
    np.testing.assert_array_equal(np.sort(output),np.sort(mixed))
    np.testing.assert_array_equal(np.argsort(output),np.argsort(late))
    assert abs(effective(output)-effective(mixed))<1e-12


def test_temperature_matches_neff_without_rank_change():
    late=np.array([.7,.2,.08,.02]);target=effective(np.array([.35,.1,.5,.05]))
    output,temperature=match_effective(late,target)
    assert temperature>0 and abs(effective(output)/target-1)<1e-7
    np.testing.assert_array_equal(np.argsort(output),np.argsort(late))
    np.testing.assert_allclose(output.sum(),1.)
    peak,_=match_effective(late,1.1)
    assert abs(effective(peak)-1.1)<1e-7


def test_uniform_and_ratio_of_means_are_not_confused():
    uniform,t=match_effective(np.ones(5)/5,5.)
    np.testing.assert_allclose(uniform,np.ones(5)/5)
    rows=patch_stats(np.array([.5,.5,1.,0.,0.,0.]),np.array([0,2,6]))
    assert abs(np.mean(rows['effective_ratio'])-.625)<1e-12
    assert np.mean(rows['effective'])/np.mean(rows['patches'])==.5
