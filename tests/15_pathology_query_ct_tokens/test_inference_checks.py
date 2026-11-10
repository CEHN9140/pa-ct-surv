import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch

from path_query_ct import PathQueryCTFusion, PathQueryCTModel
from inference_checks import cache_fusion_inputs, patient_permutations, permuted_risks, summarize


class InferenceCheckTests(unittest.TestCase):
    def test_derangements_are_reproducible_and_fold_specific(self):
        orders = patient_permutations(12, 100, 123, 0)
        np.testing.assert_array_equal(orders, patient_permutations(12, 100, 123, 0))
        self.assertFalse(np.array_equal(orders, patient_permutations(12, 100, 123, 1)))
        self.assertFalse((orders == np.arange(12)).any())
        for order in orders:
            np.testing.assert_array_equal(np.sort(order), np.arange(12))
        with self.assertRaises(ValueError):
            patient_permutations(1, 100, 123, 0)

    def test_only_intended_fusion_input_changes(self):
        torch.manual_seed(123)
        fusion = PathQueryCTFusion().eval()
        features = dict(pa_patches=torch.randn(3, 25, 512), ct_tokens=torch.randn(3, 8, 512),
                        pa_patient=torch.randn(3, 256))
        identity, donor = np.arange(3), np.array([2, 0, 1])
        with torch.inference_mode():
            for ct_order, query_order in [(identity, identity), (donor, identity), (identity, donor)]:
                captured = []
                handle = fusion.register_forward_pre_hook(lambda m, args: captured.append(args))
                actual = permuted_risks(fusion, features, ct_order, query_order, 'cpu', 2)
                handle.remove()
                for i, key in enumerate(['pa_patches', 'ct_tokens', 'pa_patient']):
                    expected_order = query_order if i == 0 else ct_order if i == 1 else identity
                    torch.testing.assert_close(torch.cat([x[i] for x in captured]),
                                               features[key][expected_order])
                expected = fusion(features['pa_patches'][query_order], features['ct_tokens'][ct_order],
                                  features['pa_patient'])[0].numpy()
                np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-5)

    def test_cache_matches_full_forward_and_keeps_patient_order(self):
        torch.manual_seed(123)
        model = PathQueryCTModel().eval()
        ct, pa = torch.rand(2, 1, 16, 16, 16), torch.randn(2, 25, 1280)
        loader = [(ct, pa, torch.tensor([1, 0]), torch.tensor([3., 9.]), ['p1', 'p0'])]
        before = {k: v.clone() for k, v in model.state_dict().items()}
        cached = cache_fusion_inputs(model, loader, torch.device('cpu'))
        self.assertEqual(cached['case_id'], ['p1', 'p0'])
        risk = permuted_risks(model.fusion, cached, np.arange(2), np.arange(2), 'cpu', 1)
        np.testing.assert_allclose(risk, cached['risk'].numpy(), rtol=1e-5, atol=1e-6)
        self.assertEqual(tuple(cached['pa_patient'].shape), (2, 256))
        for k, v in model.state_dict().items():
            torch.testing.assert_close(v, before[k])

    def test_summary_averages_repeats_before_folds(self):
        import tempfile
        import pandas as pd
        rows = [dict(fold=fold, condition='shuffle_query', repeat=repeat,
                     cindex=value, auc_36m=value, auc_60m=value, drop_cindex=0.,
                     drop_auc_36m=0., drop_auc_60m=0., risk_rmse=0., risk_rmse_over_baseline_sd=0.,
                     risk_spearman=1.)
                for fold, values in [(0, [.4, .6]), (1, [.7, .9])]
                for repeat, value in enumerate(values)]
        with tempfile.TemporaryDirectory() as directory:
            summarize(rows, Path(directory))
            result = pd.read_csv(Path(directory) / 'summary.csv').iloc[0]
            self.assertAlmostEqual(result.cindex_mean, .65)
            self.assertAlmostEqual(result.cindex_std, np.std([.5, .8], ddof=1))


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
