import sys
import subprocess
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
import torch.nn.functional as F

from inference_checks import cache_fusion_inputs, permuted_risks
from loss import cox_loss
from path_query_ct import PathQueryCTModel


class CTAdaptationTests(unittest.TestCase):
    def test_standard_attention_cli_accepts_zscore_without_qknorm(self):
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run([
                sys.executable, str(Path(__file__).with_name('run.py')),
                '--ct_zscore', '--alignment_weight', '0',
                '--data_dir', str(Path(directory) / 'missing_cohort'),
            ], capture_output=True, text=True)
        # Reach real dataset validation without creating a run or starting training.
        self.assertIn('Dataset CSV not found', result.stderr)
        self.assertNotIn('require the QKNorm baseline', result.stderr)

    def test_standard_attention_zscore_matches_manual_input_transform_and_scaled_qk(self):
        baseline = PathQueryCTModel().eval()
        model = PathQueryCTModel(ct_zscore=True).eval()
        model.load_state_dict(baseline.state_dict(), strict=True)
        self.assertIs(type(model.fusion.coattn), torch.nn.MultiheadAttention)
        self.assertFalse(hasattr(model.fusion.coattn, 'scale'))
        ct, pa = torch.rand(1, 1, 16, 16, 16), torch.randn(1, 25, 1280)
        normalized = (ct-ct.mean())/ct.std(unbiased=False).clamp_min(1e-6)
        captured = []
        handle = model.fusion.coattn.register_forward_pre_hook(lambda m, inputs: captured.append(inputs))
        with torch.no_grad():
            actual, expected = model(ct, pa), baseline(normalized, pa)
        handle.remove()
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b)
        attention = model.fusion.coattn
        weights, biases = attention.in_proj_weight.chunk(3), attention.in_proj_bias.chunk(3)
        q = F.linear(captured[0][0], weights[0], biases[0])
        k = F.linear(captured[0][1], weights[1], biases[1])
        torch.testing.assert_close(actual[6], (q @ k.transpose(-2, -1)/16).softmax(-1))

    def test_zscore_is_per_case_population_standardization_in_train_and_eval(self):
        model = PathQueryCTModel(qk_norm=True, ct_zscore=True)
        ct = torch.rand(2, 1, 16, 16, 16)
        ct[1] = .5 + .05*ct[1]
        before = ct.clone()
        pa = torch.randn(2, 25, 1280)
        captured = []
        handle = model.ct_backbone.conv1.register_forward_pre_hook(lambda m, inputs: captured.append(inputs[0]))
        with torch.no_grad():
            model.train(); model(ct, pa)
            model.eval(); model(ct, pa)
        handle.remove()
        expected = (ct-ct.mean((1, 2, 3, 4), keepdim=True))/ct.std((1, 2, 3, 4), unbiased=False, keepdim=True).clamp_min(1e-6)
        for actual in captured:
            torch.testing.assert_close(actual, expected)
            torch.testing.assert_close(actual.mean((1, 2, 3, 4)), torch.zeros(2), atol=2e-5, rtol=0)
            torch.testing.assert_close(actual.std((1, 2, 3, 4), unbiased=False), torch.ones(2))
        torch.testing.assert_close(ct, before, rtol=0, atol=0)

    def test_zscore_handles_constant_input_and_does_not_depend_on_other_patients(self):
        model = PathQueryCTModel(qk_norm=True, ct_zscore=True).eval()
        ct = torch.rand(1, 1, 16, 16, 16)
        pa = torch.randn(1, 25, 1280)
        inputs = []
        handle = model.ct_backbone.conv1.register_forward_pre_hook(lambda m, i: inputs.append(i[0]))
        with torch.no_grad():
            model(ct, pa)
            result = model(torch.cat([ct, torch.full_like(ct, .5)]), pa.expand(2, -1, -1))
        handle.remove()
        torch.testing.assert_close(inputs[0], inputs[1][:1])
        torch.testing.assert_close(inputs[1][1], torch.zeros_like(ct[0]))
        self.assertTrue(all(torch.isfinite(x).all() for x in result))

    def test_zscore_preserves_fixed_bn_statistics_during_training(self):
        model = PathQueryCTModel(qk_norm=True, ct_zscore=True).train()
        buffers = {k: v.clone() for k, v in model.named_buffers()}
        self.assertTrue(model.ct_backbone.freeze_bn_stats)
        self.assertTrue(all(not m.training for m in model.ct_backbone.modules() if isinstance(m, torch.nn.BatchNorm3d)))
        with torch.no_grad():
            model(torch.rand(1, 1, 16, 16, 16), torch.randn(1, 25, 1280))
        for k, v in model.named_buffers():
            torch.testing.assert_close(v, buffers[k], rtol=0, atol=0)

    def test_bn_control_keeps_raw_ct_and_updates_statistics_only_in_train(self):
        model = PathQueryCTModel(qk_norm=True, ct_bn_adapt=True).train()
        ct, pa = torch.rand(1, 1, 16, 16, 16), torch.randn(1, 25, 1280)
        inputs = []
        handle = model.ct_backbone.conv1.register_forward_pre_hook(lambda m, i: inputs.append(i[0]))
        bn = model.ct_backbone.bn1
        initial = bn.running_mean.clone()
        with torch.no_grad():
            model(ct, pa)
        self.assertFalse(model.ct_backbone.freeze_bn_stats)
        self.assertEqual(bn.num_batches_tracked.item(), 1)
        self.assertFalse(torch.equal(initial, bn.running_mean))
        trained = {k: v.clone() for k, v in model.named_buffers()}
        model.eval()
        with torch.no_grad():
            model(ct*.5, pa)
        for k, v in model.named_buffers():
            torch.testing.assert_close(v, trained[k], rtol=0, atol=0)
        handle.remove()
        torch.testing.assert_close(inputs[0], ct, rtol=0, atol=0)
        torch.testing.assert_close(inputs[1], ct*.5, rtol=0, atol=0)
        self.assertTrue(all(p.requires_grad for p in model.parameters()))
        self.assertTrue(bn.weight.requires_grad and bn.bias.requires_grad)
        model.train()
        self.assertTrue(all(m.training for m in model.ct_backbone.modules() if isinstance(m, torch.nn.BatchNorm3d)))

    def test_both_controls_preserve_initialization_and_support_cox_gradients(self):
        torch.manual_seed(123)
        baseline = PathQueryCTModel(qk_norm=True)
        for flag in ['ct_zscore', 'ct_bn_adapt']:
            with self.subTest(flag=flag):
                torch.manual_seed(123)
                model = PathQueryCTModel(qk_norm=True, **{flag: True}).train()
                for k, value in baseline.state_dict().items():
                    torch.testing.assert_close(model.state_dict()[k], value, rtol=0, atol=0)
                risks = [model(torch.rand(1, 1, 16, 16, 16), torch.randn(1, 25, 1280))[0] for _ in range(2)]
                loss = cox_loss(torch.cat(risks), torch.tensor([3., 12.]), torch.tensor([1, 0]))
                loss.backward()
                for p in [model.ct_backbone.conv1.weight, model.ct_backbone.bn1.weight,
                          model.pa_branch.compression[0].weight, model.fusion.coattn.scale]:
                    self.assertTrue(torch.isfinite(p.grad).all())
                    self.assertGreater(p.grad.abs().sum().item(), 0)

    def test_both_controls_strict_reload_and_cached_inference_keep_bn_unchanged(self):
        loader = [(torch.rand(2, 1, 16, 16, 16), torch.randn(2, 25, 1280),
                   torch.tensor([1, 0]), torch.tensor([3., 12.]), ['a', 'b'])]
        for flag in ['ct_zscore', 'ct_bn_adapt']:
            with self.subTest(flag=flag):
                model = PathQueryCTModel(qk_norm=True, **{flag: True}).train()
                with torch.no_grad(): model(*loader[0][:2])
                restored = PathQueryCTModel(qk_norm=True, **{flag: True}).eval()
                restored.load_state_dict(model.state_dict(), strict=True)
                before = {k: v.clone() for k, v in restored.named_buffers()}
                cached = cache_fusion_inputs(restored, loader, 'cpu')
                actual = permuted_risks(restored.fusion, cached, np.arange(2), np.arange(2), 'cpu', 1)
                np.testing.assert_allclose(actual, cached['risk'], rtol=1e-4, atol=1e-6)
                model.eval()
                with torch.no_grad(): torch.testing.assert_close(model(*loader[0][:2])[0], cached['risk'])
                for k, v in restored.named_buffers():
                    torch.testing.assert_close(v, before[k], rtol=0, atol=0)

    def test_controls_cannot_be_enabled_together(self):
        with self.assertRaisesRegex(ValueError, 'one CT factor'):
            PathQueryCTModel(qk_norm=True, ct_zscore=True, ct_bn_adapt=True)


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
