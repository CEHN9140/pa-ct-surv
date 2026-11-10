import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from loss import cox_loss
from pact_train import build_pact_optimizer
from path_query_ct import PathQueryCTFusion, PathQueryCTModel


class PathQueryCTTests(unittest.TestCase):
    def test_no_l2_preserves_raw_features_for_both_pathology_routes(self):
        torch.manual_seed(123)
        model = PathQueryCTModel(pa_l2_normalize=False).eval()
        ct, pa = torch.rand(1, 1, 16, 16, 16), torch.randn(1, 25, 1280) * 3
        inputs, fusion_inputs = [], []
        handles = [
            model.pa_branch.compression.register_forward_pre_hook(lambda m, i: inputs.append(i[0])),
            model.fusion.register_forward_pre_hook(lambda m, i: fusion_inputs.append(i)),
        ]
        with torch.no_grad():
            output = model(ct, pa)
        for h in handles:
            h.remove()
        torch.testing.assert_close(inputs[0], pa)
        self.assertEqual(len(output), 7)
        with torch.no_grad():
            risk, patient, _, patches = model.pa_branch(pa, return_patch_features=True)
        torch.testing.assert_close(output[2], risk)
        torch.testing.assert_close(fusion_inputs[0][0], patches)
        torch.testing.assert_close(fusion_inputs[0][2], patient)

    def test_l2_switch_does_not_change_initialization_or_checkpoint_keys(self):
        torch.manual_seed(123)
        original = PathQueryCTModel()
        torch.manual_seed(123)
        raw = PathQueryCTModel(pa_l2_normalize=False)
        self.assertTrue(original.pa_l2_normalize)
        for name, tensor in original.state_dict().items():
            torch.testing.assert_close(tensor, raw.state_dict()[name], rtol=0, atol=0)
        raw.load_state_dict(original.state_dict(), strict=True)

    def test_attention_direction_order_and_gradients(self):
        torch.manual_seed(123)
        fusion = PathQueryCTFusion(dropout=.25).eval()
        pa = torch.randn(2, 25, 512, requires_grad=True)
        ct = torch.randn(2, 32, 512, requires_grad=True)
        patient = torch.randn(2, 256, requires_grad=True)
        retrieved = []
        handle = fusion.coattn.register_forward_hook(lambda m, i, o: retrieved.append(o[0]))
        risk, guided, weights, pool_weights = fusion(pa, ct, patient)
        handle.remove()
        self.assertEqual(weights.shape, (2, 25, 32))
        self.assertEqual(guided.shape, (2, 256))
        self.assertEqual(risk.shape, (2,))
        torch.testing.assert_close(weights.sum(-1), torch.ones(2, 25))
        self.assertEqual(pool_weights.shape, (2, 25, 1))
        torch.testing.assert_close(pool_weights.sum(1), torch.ones(2, 1))
        torch.testing.assert_close(guided, (pool_weights * retrieved[0]).sum(1))
        ct_order = torch.randperm(32)
        reordered = fusion(pa, ct[:, ct_order], patient)
        torch.testing.assert_close(risk, reordered[0])
        torch.testing.assert_close(weights[:, :, ct_order], reordered[2])
        pa_order = torch.randperm(25)
        reordered_pa = fusion(pa[:, pa_order], ct, patient)
        torch.testing.assert_close(risk, reordered_pa[0])
        torch.testing.assert_close(weights[:, pa_order], reordered_pa[2])
        other_query = fusion(pa.flip(0), ct, patient)[2]
        self.assertGreater((weights - other_query).abs().max().item(), 1e-5)
        risk.sum().backward()
        for tensor in [pa, ct, patient]:
            self.assertTrue(torch.isfinite(tensor.grad).all())
            self.assertGreater(tensor.grad.abs().sum().item(), 0)
        self.assertGreater(fusion.query_pool[-1].weight.grad.abs().sum().item(), 0)

    def test_model_spatial_tokens_and_auxiliary_parity(self):
        torch.manual_seed(123)
        model = PathQueryCTModel().eval()
        spatial, tokens, path_inputs = [], [], []
        handles = [
            model.ct_backbone.layer4.register_forward_hook(lambda m, i, o: spatial.append(o.detach())),
            model.fusion.ct_projection.register_forward_pre_hook(lambda m, i: tokens.append(i[0].detach())),
            model.pa_branch.compression.register_forward_pre_hook(lambda m, i: path_inputs.append(i[0].detach())),
        ]
        ct, pa = torch.rand(2, 1, 16, 16, 16), torch.randn(2, 25, 1280)
        with torch.no_grad():
            output = model(ct, pa)
        for h in handles:
            h.remove()
        self.assertEqual(len(spatial), 1)
        self.assertEqual(len(path_inputs), 1)
        torch.testing.assert_close(tokens[0], spatial[0].flatten(2).transpose(1, 2))
        torch.testing.assert_close(path_inputs[0].norm(dim=-1), torch.ones(2, 25))
        self.assertEqual(output[6].shape, (2, 25, 8))
        self.assertEqual(output[3].shape, (2, 25, 1))
        torch.testing.assert_close(output[3].sum(1), torch.ones(2, 1))
        with torch.no_grad():
            torch.testing.assert_close(output[1], model.ct_backbone(ct))
            torch.testing.assert_close(output[2], model.pa_branch(torch.nn.functional.normalize(pa, dim=-1))[0])
        restored = PathQueryCTModel().eval()
        restored.load_state_dict(model.state_dict(), strict=True)
        with torch.no_grad():
            torch.testing.assert_close(restored(ct, pa)[0], output[0])

    def test_fused_cox_trains_both_branches_and_query_key(self):
        model = PathQueryCTModel().train()
        ct, pa = torch.rand(2, 1, 16, 16, 16), torch.randn(2, 25, 1280)
        output = model(ct, pa)
        loss = cox_loss(output[0], torch.tensor([4., 12.]), torch.tensor([1, 0]))
        loss.backward()
        for parameter in [model.ct_backbone.conv1.weight, model.pa_branch.compression[0].weight,
                          model.fusion.pa_projection[0].weight, model.fusion.ct_projection[0].weight]:
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(parameter.grad.abs().sum().item(), 0)
        grad = model.fusion.coattn.in_proj_weight.grad
        self.assertGreater(grad[:256].abs().sum().item(), 0)
        self.assertGreater(grad[256:512].abs().sum().item(), 0)
        self.assertTrue(all(p.requires_grad for p in model.parameters()))
        optimizer = build_pact_optimizer(model, 1e-4, 5e-4, 1e-5)
        groups = {id(p): g['lr'] for g in optimizer.param_groups for p in g['params']}
        self.assertEqual(groups[id(model.ct_backbone.conv1.weight)], 1e-5)
        self.assertEqual(groups[id(model.fusion.coattn.in_proj_weight)], 1e-4)


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
