import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import torch.nn.functional as F
from pamt_alignment import pamt_alignment_loss
from path_query_ct import PathQueryCTModel


class PAMTAlignmentTests(unittest.TestCase):
    def test_original_sorted_cross_entropy_value_and_gradient(self):
        torch.manual_seed(123)
        pa = torch.randn(2, 25, 256, dtype=torch.float64, requires_grad=True)
        ct = torch.randn(2, 512, 256, dtype=torch.float64, requires_grad=True)
        scores = pa @ ct.transpose(-2, -1)
        # PAMT utils_cox.py: transpose, sort keys, target=1 for topK, then CE.
        original_scores = scores.transpose(-2, -1).sort(dim=1, descending=True).values
        targets = torch.zeros_like(original_scores)
        targets[:, :2, :] = 1
        expected = F.cross_entropy(original_scores, targets)
        actual = pamt_alignment_loss(pa, ct, topk=2)
        torch.testing.assert_close(actual, expected)
        original_grads = torch.autograd.grad(expected, (pa, ct), retain_graph=True)
        actual_grads = torch.autograd.grad(actual, (pa, ct))
        for a, b in zip(actual_grads, original_grads):
            torch.testing.assert_close(a, b)

    def test_target_sum_and_patient_mean_match_pamt(self):
        pa, ct = torch.zeros(3, 25, 256), torch.zeros(3, 512, 256)
        loss = pamt_alignment_loss(pa, ct, topk=2)
        torch.testing.assert_close(loss, torch.tensor(2 * math.log(512)))
        torch.manual_seed(1)
        pa, ct = torch.randn(3, 25, 256), torch.randn(3, 512, 256)
        whole = pamt_alignment_loss(pa, ct)
        per_patient = torch.stack([pamt_alignment_loss(pa[i:i+1], ct[i:i+1]) for i in range(3)])
        torch.testing.assert_close(whole, per_patient.mean())
        torch.testing.assert_close(whole, pamt_alignment_loss(pa, ct[:, torch.randperm(512)]))
        with self.assertRaises(ValueError):
            pamt_alignment_loss(pa, ct, topk=513)

    def test_alignment_does_not_change_prediction_or_checkpoint_keys(self):
        torch.manual_seed(123)
        baseline = PathQueryCTModel().eval()
        aligned = PathQueryCTModel(alignment_topk=2).eval()
        aligned.load_state_dict(baseline.state_dict(), strict=True)
        ct, pa = torch.rand(1, 1, 16, 16, 16), torch.randn(1, 25, 1280)
        projected = []
        handles = [
            aligned.fusion.pa_projection.register_forward_hook(lambda m, i, o: projected.append(o)),
            aligned.fusion.ct_projection.register_forward_hook(lambda m, i, o: projected.append(o)),
        ]
        with torch.no_grad():
            a, b = baseline(ct, pa), aligned(ct, pa)
        for h in handles:
            h.remove()
        self.assertEqual(len(a), 7)
        self.assertEqual(len(b), 8)
        for old, new in zip(a, b[:7]):
            torch.testing.assert_close(old, new)
        torch.testing.assert_close(b[7], pamt_alignment_loss(*projected, topk=2))


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
