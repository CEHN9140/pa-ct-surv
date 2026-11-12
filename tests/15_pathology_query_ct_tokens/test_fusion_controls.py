import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
import torch.nn.functional as F

from inference_checks import cache_fusion_inputs, permuted_risks
from loss import cox_loss
from path_query_ct import PathQueryCTFusion, PathQueryCTModel


class FusionControlTests(unittest.TestCase):
    def test_pathology_only_is_exactly_the_same_head_with_a_zero_ct_slot(self):
        fusion = PathQueryCTFusion(fusion_control='pathology_only').eval()
        pa, ct, patient = torch.randn(2, 25, 512), torch.randn(2, 32, 512), torch.randn(2, 256)
        actual = fusion(pa, ct, patient)
        torch.testing.assert_close(actual[0], fusion.head(torch.zeros_like(patient), patient))
        torch.testing.assert_close(actual[1], torch.zeros_like(patient), rtol=0, atol=0)
        torch.testing.assert_close(fusion(pa * 20, ct.flip(0) * 100, patient)[0], actual[0])
        for module in ('pa_projection', 'ct_projection', 'coattn', 'query_pool'):
            self.assertIsNone(getattr(fusion, module))

    def test_pathology_only_skips_ct_and_is_invariant_in_train_and_eval(self):
        model = PathQueryCTModel(fusion_control='pathology_only')
        self.assertIsNone(model.ct_backbone)
        ct, pa = torch.rand(2, 1, 16, 16, 16), torch.randn(2, 25, 1280)
        for training in (True, False):
            model.train(training)
            torch.manual_seed(31)
            first = model(ct, pa)
            torch.manual_seed(31)
            second = model(torch.full_like(ct, float('nan')), pa)
            self.assertEqual(len(first), 7)
            torch.testing.assert_close(first[0], second[0], rtol=0, atol=0)
            self.assertEqual(first[6].shape, (2, 25, 0))
            torch.testing.assert_close(first[1], torch.zeros(2), rtol=0, atol=0)
        loss = cox_loss(first[0], torch.tensor([3., 12.]), torch.tensor([1, 0]))
        loss.backward()
        for parameter in (model.pa_branch.compression[0].weight, next(model.fusion.head.parameters())):
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(parameter.grad.abs().sum().item(), 0)
        self.assertTrue(all(p.requires_grad for p in model.parameters()))

    def test_uniform_ct_is_the_exact_internal_value_and_output_projection_average(self):
        fusion = PathQueryCTFusion(fusion_control='uniform_ct').eval()
        pa, ct, patient = torch.randn(2, 25, 512), torch.randn(2, 32, 512), torch.randn(2, 256)
        risk, guided, attention, pool = fusion(pa, ct, patient)
        projected = fusion.ct_projection(ct)
        value = F.linear(projected, fusion.coattn.in_proj_weight[512:], fusion.coattn.in_proj_bias[512:])
        expected = fusion.coattn.out_proj(value.mean(dim=1))
        torch.testing.assert_close(guided, expected)
        torch.testing.assert_close(risk, fusion.head(expected, patient))
        torch.testing.assert_close(attention, torch.full((2, 25, 32), 1 / 32), rtol=0, atol=0)
        torch.testing.assert_close(pool.sum(1), torch.ones(2, 1))
        torch.testing.assert_close(fusion(pa.flip(0) * 15, ct, patient)[0], risk)
        self.assertGreater((fusion(pa, ct.flip(0), patient)[0] - risk).abs().max().item(), 1e-6)

    def test_controls_keep_the_retained_initialization_identical_to_standard(self):
        torch.manual_seed(123)
        baseline = PathQueryCTModel().state_dict()
        for mode in ('pathology_only', 'uniform_ct'):
            torch.manual_seed(123)
            model = PathQueryCTModel(fusion_control=mode)
            for name, value in model.state_dict().items():
                torch.testing.assert_close(value, baseline[name], rtol=0, atol=0)
            if mode == 'uniform_ct':
                self.assertEqual(set(model.state_dict()), set(baseline))
            model.eval()
            restored = PathQueryCTModel(fusion_control=mode).eval()
            restored.load_state_dict(model.state_dict(), strict=True)
            ct, pa = torch.rand(1, 1, 16, 16, 16), torch.randn(1, 25, 1280)
            with torch.no_grad():
                torch.testing.assert_close(model(ct, pa)[0], restored(ct, pa)[0])

    def test_uniform_ct_cox_updates_ct_pa_and_value_but_not_query_key_scores(self):
        model = PathQueryCTModel(fusion_control='uniform_ct').train()
        before = {k: v.clone() for k, v in model.named_buffers()}
        ct, pa = torch.rand(2, 1, 16, 16, 16), torch.randn(2, 25, 1280)
        loss = cox_loss(model(ct, pa)[0], torch.tensor([3., 12.]), torch.tensor([1, 0]))
        loss.backward()
        for parameter in (model.ct_backbone.conv1.weight, model.pa_branch.compression[0].weight,
                          model.fusion.ct_projection[0].weight, model.fusion.coattn.out_proj.weight):
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(parameter.grad.abs().sum().item(), 0)
        grad = model.fusion.coattn.in_proj_weight.grad
        torch.testing.assert_close(grad[:512], torch.zeros_like(grad[:512]), rtol=0, atol=0)
        self.assertGreater(grad[512:].abs().sum().item(), 0)
        self.assertTrue(all(p.requires_grad for p in model.parameters()))
        for k, v in model.named_buffers():
            torch.testing.assert_close(v, before[k], rtol=0, atol=0)

    def test_controls_preserve_cached_fusion_prediction_parity(self):
        loader = [(torch.rand(2, 1, 16, 16, 16), torch.randn(2, 25, 1280),
                   torch.tensor([1, 0]), torch.tensor([3., 12.]), ['a', 'b'])]
        for mode in ('pathology_only', 'uniform_ct'):
            model = PathQueryCTModel(fusion_control=mode).eval()
            cached = cache_fusion_inputs(model, loader, 'cpu')
            risk = permuted_risks(model.fusion, cached, np.arange(2), np.arange(2), 'cpu', 1)
            np.testing.assert_allclose(risk, cached['risk'], rtol=1e-4, atol=1e-6)

    def test_controls_cli_reaches_dataset_validation_and_rejects_mixed_interventions(self):
        script = str(Path(__file__).with_name('run.py'))
        for mode in ('pathology_only', 'uniform_ct'):
            with tempfile.TemporaryDirectory() as directory:
                command = [sys.executable, script, '--fusion_control', mode, '--alignment_weight', '0',
                           '--data_dir', str(Path(directory) / 'missing')]
                result = subprocess.run(command, capture_output=True, text=True)
                self.assertIn('Dataset CSV not found', result.stderr)
                for flag in ('--qk_norm', '--ct_zscore', '--ct_bn_adapt', '--all_patch_queries',
                             '--no_pa_l2_normalize'):
                    result = subprocess.run(command + [flag], capture_output=True, text=True)
                    self.assertIn('Fusion controls require', result.stderr)
        result = subprocess.run([sys.executable, script, '--fusion_control', 'uniform_ct'],
                                capture_output=True, text=True)
        self.assertIn('Fusion controls require', result.stderr)

    def test_model_rejects_unsupported_control_combinations(self):
        for mode in ('pathology_only', 'uniform_ct'):
            for options in ({'qk_norm': True}, {'ct_zscore': True}, {'ct_bn_adapt': True},
                            {'all_patch_queries': True}, {'pa_l2_normalize': False}, {'alignment_topk': 2}):
                with self.assertRaisesRegex(ValueError, 'Fusion controls require'):
                    PathQueryCTModel(fusion_control=mode, **options)


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
