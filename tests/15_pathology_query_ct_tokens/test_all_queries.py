import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import h5py
import numpy as np
import pandas as pd
import torch

from all_query_dataset import AllPatchQueryDataset
from dataset import build_dataloader
from inference_checks import cache_fusion_inputs, permuted_risks
from loss import cox_loss
from path_query_ct import PathQueryCTModel


class AllQueryDataTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        self.data_dir, self.full_dir = root / 'labels', root / 'full'
        self.data_dir.mkdir(); self.full_dir.mkdir()
        self.inputs, rows = [], []
        rng = np.random.default_rng(123)
        for i, n in enumerate([31, 43]):
            pa_id = f'pa{i}'
            full = rng.normal(size=(n, 1280)).astype('float32')
            indices = rng.permutation(n)[:25].astype('int64')
            coords = np.stack([np.arange(n)*300, np.arange(n)*400], axis=1).astype('int64')
            top_path, ct_path = root / f'{pa_id}.h5', root / f'ct{i}.npy'
            np.save(ct_path, np.zeros((64, 64, 64), dtype='float32'))
            for path, features, xy in [(self.full_dir / f'{pa_id}.h5', full, coords),
                                       (top_path, full[indices], coords[indices])]:
                with h5py.File(path, 'w') as h:
                    h['features'], h['coords'] = features, xy
                    h.attrs.update(downstream_encoder='Virchow2-CLS', source_stamp=f'/stamp/{pa_id}.h5',
                                   input_signature=f'input-{pa_id}')
                    if path == top_path:
                        h['selected_indices'] = indices
            rows.append(dict(pa_id=pa_id, pa_path=str(top_path), coord_path=str(top_path),
                             ct_id=f'ct{i}', ct_path=str(ct_path), event=1-i, time=3.+i*9, split=i))
            self.inputs.append((full, indices))
        self.rows = pd.DataFrame(rows)
        self.rows.to_csv(self.data_dir / 'all_label_roi64.csv', index=False)

    def tearDown(self):
        self.temporary.cleanup()

    def test_exact_top25_first_and_all_other_rows_once_with_unchanged_labels(self):
        dataset = AllPatchQueryDataset(self.data_dir, self.full_dir)
        pd.testing.assert_frame_equal(dataset.samples, self.rows)
        for i, (full, indices) in enumerate(self.inputs):
            _, patches, event, time, case = dataset[i]
            mask = np.ones(len(full), dtype=bool); mask[indices] = False
            order = np.concatenate([indices, np.flatnonzero(mask)])
            np.testing.assert_array_equal(patches.numpy(), full[order])
            np.testing.assert_array_equal(np.sort(dataset.query_orders[i]), np.arange(len(full)))
            self.assertEqual((event.item(), time.item(), case), (1-i, 3.+i*9, f'pa{i}|ct{i}'))
        batches = list(build_dataloader(dataset, batch_size=1, num_workers=0))
        self.assertEqual([tuple(b[1].shape) for b in batches], [(1, 31, 1280), (1, 43, 1280)])

    def test_feature_or_coordinate_mismatch_is_rejected(self):
        full, indices = self.inputs[0]
        path = self.full_dir / 'pa0.h5'
        with h5py.File(path, 'r+') as h:
            h['features'][indices[0], 0] += 1
        with self.assertRaisesRegex(ValueError, 'Top25 features'):
            AllPatchQueryDataset(self.data_dir, self.full_dir)
        with h5py.File(path, 'r+') as h:
            h['features'][:] = full
            h['coords'][indices[0], 0] += 1
        with self.assertRaisesRegex(ValueError, 'Top25 coordinates'):
            AllPatchQueryDataset(self.data_dir, self.full_dir)

    def test_duplicate_top25_indices_are_rejected(self):
        with h5py.File(self.rows.iloc[0].pa_path, 'r+') as h:
            h['selected_indices'][1] = h['selected_indices'][0]
        with self.assertRaisesRegex(ValueError, 'selected_indices'):
            AllPatchQueryDataset(self.data_dir, self.full_dir)


class AllQueryModelTests(unittest.TestCase):
    def test_25_queries_match_original_model_with_same_parameters(self):
        torch.manual_seed(123)
        original = PathQueryCTModel().eval()
        all_queries = PathQueryCTModel(all_patch_queries=True).eval()
        all_queries.load_state_dict(original.state_dict(), strict=True)
        ct, pa = torch.rand(1, 1, 16, 16, 16), torch.randn(1, 25, 1280)
        with torch.no_grad():
            a, b = original(ct, pa), all_queries(ct, pa)
        for x, y in zip(a, b):
            torch.testing.assert_close(x, y)

    def test_direct_pathology_uses_only_top25_but_queries_use_every_patch(self):
        model = PathQueryCTModel(all_patch_queries=True).eval()
        ct, pa = torch.rand(1, 1, 16, 16, 16), torch.randn(1, 43, 1280)
        other = pa.clone(); other[:, 25:] = torch.randn_like(other[:, 25:]) * 7
        attention_inputs = []
        handle = model.pa_branch.attention.register_forward_pre_hook(lambda m, i: attention_inputs.append(i[0]))
        with torch.no_grad():
            a, b = model(ct, pa), model(ct, other)
            expected = model.pa_branch(torch.nn.functional.normalize(pa[:, :25], dim=-1))
        handle.remove()
        self.assertTrue(all(x.shape[1] == 25 for x in attention_inputs))
        torch.testing.assert_close(a[5], b[5])
        torch.testing.assert_close(a[2], expected[0])
        torch.testing.assert_close(a[5], expected[1])
        self.assertEqual(a[6].shape, (1, 43, 8))
        self.assertEqual(a[3].shape, (1, 43, 1))

    def test_remaining_queries_and_both_branches_receive_cox_gradients(self):
        model = PathQueryCTModel(all_patch_queries=True).train()
        bags = [torch.randn(1, n, 1280, requires_grad=True) for n in [31, 43]]
        risks = [model(torch.rand(1, 1, 16, 16, 16), bag)[0] for bag in bags]
        cox_loss(torch.cat(risks), torch.tensor([3., 12.]), torch.tensor([1, 0])).backward()
        for bag in bags:
            self.assertTrue(torch.isfinite(bag.grad).all())
            self.assertGreater(bag.grad[:, 25:].abs().sum().item(), 0)
        for parameter in [model.ct_backbone.conv1.weight, model.pa_branch.compression[0].weight,
                          model.fusion.coattn.in_proj_weight]:
            self.assertGreater(parameter.grad.abs().sum().item(), 0)
        self.assertTrue(all(p.requires_grad for p in model.parameters()))

    def test_variable_query_cache_and_query_shuffle_keep_patient_feature(self):
        model = PathQueryCTModel(all_patch_queries=True).eval()
        loader = [(torch.rand(1, 1, 16, 16, 16), torch.randn(1, n, 1280),
                   torch.tensor([1-i]), torch.tensor([3.+i*9]), [f'case{i}'])
                  for i, n in enumerate([31, 43])]
        cached = cache_fusion_inputs(model, loader, torch.device('cpu'))
        self.assertIsInstance(cached['pa_patches'], list)
        self.assertEqual([x.size(0) for x in cached['pa_patches']], [31, 43])
        identity, donor = np.arange(2), np.array([1, 0])
        baseline = permuted_risks(model.fusion, cached, identity, identity, 'cpu', 16)
        np.testing.assert_allclose(baseline, cached['risk'], rtol=1e-4, atol=1e-6)
        actual = permuted_risks(model.fusion, cached, identity, donor, 'cpu', 16)
        with torch.no_grad():
            expected = [model.fusion(cached['pa_patches'][donor[i]][None], cached['ct_tokens'][i:i+1],
                                     cached['pa_patient'][i:i+1])[0].item() for i in range(2)]
        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-6)


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
