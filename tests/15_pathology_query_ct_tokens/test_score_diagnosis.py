import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from path_query_ct import PathQueryCTFusion
from diagnose_scores import attention_parts, feature_stats


class ScoreDiagnosisTests(unittest.TestCase):
    def test_manual_attention_matches_pytorch_outputs_and_input_gradients(self):
        torch.manual_seed(123)
        fusion = PathQueryCTFusion().double().eval()
        q = torch.randn(2, 25, 256, dtype=torch.float64, requires_grad=True)
        k = torch.randn(2, 17, 256, dtype=torch.float64, requires_grad=True)
        parts = attention_parts(fusion.coattn, q, k)
        actual, weights = fusion.coattn(q, k, k)
        torch.testing.assert_close(parts['weights'], weights)
        torch.testing.assert_close(parts['retrieved'], actual)
        a = torch.autograd.grad(parts['retrieved'].square().sum(), (q, k), retain_graph=True)
        b = torch.autograd.grad(actual.square().sum(), (q, k))
        for x, y in zip(a, b):
            torch.testing.assert_close(x, y)

    def test_spatially_identical_keys_give_uniform_attention(self):
        fusion = PathQueryCTFusion().eval()
        q = torch.randn(2, 25, 256)
        k = torch.randn(2, 1, 256).expand(-1, 17, -1)
        parts = attention_parts(fusion.coattn, q, k)
        torch.testing.assert_close(parts['weights'], torch.full((2, 25, 17), 1 / 17))
        torch.testing.assert_close(parts['logits'].std(-1, unbiased=False), torch.zeros(2, 25), atol=1e-6, rtol=0)

    def test_feature_statistics_separate_scale_from_spatial_similarity(self):
        x = torch.randn(2, 1, 32).expand(-1, 17, -1)
        a, b = feature_stats(x), feature_stats(x * 10)
        self.assertAlmostEqual(a['within_token_relative_rms'], 0., places=6)
        self.assertAlmostEqual(a['mean_pairwise_token_cosine'], 1., places=6)
        self.assertAlmostEqual(b['rms'] / a['rms'], 10., places=5)
        self.assertAlmostEqual(b['mean_pairwise_token_cosine'], a['mean_pairwise_token_cosine'], places=6)


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
