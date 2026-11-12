import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
import torch.nn.functional as F

from inference_checks import cache_fusion_inputs, permuted_risks
from loss import cox_loss
from pact_train import build_pact_optimizer
from path_query_ct import PathQueryCTFusion, PathQueryCTModel


class QKNormTests(unittest.TestCase):
    def test_attention_matches_internal_normalized_qk_and_unchanged_value(self):
        torch.manual_seed(123)
        fusion = PathQueryCTFusion(dropout=0., qk_norm=True).eval()
        attention = fusion.coattn.double()
        query, context = torch.randn(2, 25, 256, dtype=torch.float64), torch.randn(2, 37, 256, dtype=torch.float64)
        output, actual = attention(query, context, context)
        weight, bias = attention.in_proj_weight, attention.in_proj_bias
        q = F.linear(query, weight[:256], bias[:256])
        k = F.linear(context, weight[256:512], bias[256:512])
        v = F.linear(context, weight[512:], bias[512:])
        expected = (attention.scale * (F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1))).softmax(-1)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(output, F.linear(expected @ v, attention.out_proj.weight, attention.out_proj.bias))
        torch.testing.assert_close(actual.sum(-1), torch.ones(2, 25, dtype=torch.float64))
        self.assertAlmostEqual(attention.scale.item(), math.log2(512**2-512), places=5)

    def test_positive_qk_rescaling_leaves_attention_unchanged_and_zero_is_finite(self):
        attention = PathQueryCTFusion(qk_norm=True).coattn.double()
        q, k, v = [torch.randn(2, n, 256, dtype=torch.float64) for n in [25, 37, 37]]
        with torch.no_grad():
            original = attention(q, k, v)
            # Initial projection biases are zero; scaling each input rescales its internal Q/K only.
            scaled = attention(q*1e-4, k*1e4, v)
            for a, b in zip(original, scaled):
                torch.testing.assert_close(a, b, rtol=1e-10, atol=1e-10)
            output, weights = attention(torch.zeros_like(q), torch.zeros_like(k), v)
            self.assertTrue(torch.isfinite(output).all())
            torch.testing.assert_close(weights, torch.full_like(weights, 1/37))

    def test_fused_cox_trains_scale_and_both_branches_and_checkpoint_roundtrip(self):
        torch.manual_seed(123)
        baseline = PathQueryCTModel()
        torch.manual_seed(123)
        model = PathQueryCTModel(qk_norm=True).train()
        self.assertEqual(set(model.state_dict())-set(baseline.state_dict()), {'fusion.coattn.scale'})
        for key, value in baseline.state_dict().items():
            torch.testing.assert_close(value, model.state_dict()[key], rtol=0, atol=0)
        ct, pa = torch.rand(2, 1, 16, 16, 16), torch.randn(2, 25, 1280)
        output = model(ct, pa)
        self.assertEqual(output[6].shape, (2, 25, 8))
        self.assertEqual(len(output), 7)
        cox_loss(output[0], torch.tensor([3., 12.]), torch.tensor([1, 0])).backward()
        for parameter in [model.ct_backbone.conv1.weight, model.pa_branch.compression[0].weight,
                          model.fusion.coattn.in_proj_weight, model.fusion.coattn.scale]:
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(parameter.grad.abs().sum().item(), 0)
        optimizer = build_pact_optimizer(model, 1e-4, 5e-4, 1e-5)
        scale = model.fusion.coattn.scale
        self.assertEqual(sum(p is scale for g in optimizer.param_groups for p in g['params']), 1)
        optimizer.step()
        restored = PathQueryCTModel(qk_norm=True).eval()
        restored.load_state_dict(model.state_dict(), strict=True)
        model.eval()
        with torch.no_grad():
            for actual, expected in zip(restored(ct, pa), model(ct, pa)):
                torch.testing.assert_close(actual, expected)

    def test_qknorm_cached_inference_and_recipient_patient_feature(self):
        model = PathQueryCTModel(qk_norm=True).eval()
        loader = [(torch.rand(2, 1, 16, 16, 16), torch.randn(2, 25, 1280),
                   torch.tensor([1, 0]), torch.tensor([3., 12.]), ['a', 'b'])]
        features = cache_fusion_inputs(model, loader, 'cpu')
        identity, swap = np.arange(2), np.array([1, 0])
        risks = permuted_risks(model.fusion, features, identity, identity, 'cpu', 1)
        np.testing.assert_allclose(risks, features['risk'], rtol=1e-4, atol=1e-6)
        actual = permuted_risks(model.fusion, features, identity, swap, 'cpu', 1)
        with torch.no_grad():
            expected = model.fusion(features['pa_patches'][swap], features['ct_tokens'], features['pa_patient'])[0]
        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-6)


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
