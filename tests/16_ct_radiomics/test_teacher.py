import sys
from pathlib import Path
from types import SimpleNamespace

import h5py
import joblib
import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from teacher_data import RadiomicsPreprocessor, PaRadiomicsDataset, teacher_partitions, join_teacher_tables
from teacher_model import RadiomicsTeacher
from teacher_train import fit_teacher, predict_teacher, load_teacher


def radiomics_table(n=60):
    rng = np.random.default_rng(123)
    groups = ['shape', 'firstorder', 'glcm', 'glrlm', 'glszm', 'gldm', 'ngtdm']
    return pd.DataFrame({f'original_{group}_feature{i}': rng.normal(size=n)
                         for group in groups for i in range(2)})


def test_preprocessing_uses_fit_only_and_preserves_group_layout(tmp_path):
    table = radiomics_table()
    table['original_shape_constant'] = 1.
    table['original_firstorder_duplicate'] = table['original_shape_feature0']
    prep = RadiomicsPreprocessor().fit(table.iloc[:40], list(range(40)))
    before = prep.scaler.mean_.copy()
    transformed = prep.transform(table.iloc[40:] + 1000)
    np.testing.assert_allclose(before, table.iloc[:40][prep.selected_].mean())
    np.testing.assert_array_equal(prep.scaler.mean_, before)
    assert transformed.shape == (20, len(prep.selected_))
    assert 'original_shape_constant' not in prep.selected_
    assert len(prep.group_indices) == 7
    assert sorted(i for group in prep.group_indices for i in group) == list(range(len(prep.selected_)))
    joblib.dump(prep, tmp_path/'prep.joblib')
    np.testing.assert_array_equal(joblib.load(tmp_path/'prep.joblib').transform(table), prep.transform(table))


def test_join_is_by_patient_identity_and_rejects_label_or_wsi_mismatch(tmp_path):
    rows = []
    for i in range(5):
        path = tmp_path / f'p{i}.h5'
        with h5py.File(path, 'w') as file:
            file['features'] = np.zeros((25, 1280), dtype=np.float32) + i
        rows.append(dict(pa_id=f'p{i}', ct_id=str(i), pa_path=str(path), event=i % 2,
                         time=float(i+1), split=i))
    samples = pd.DataFrame(rows)
    features = radiomics_table(5)
    features['ct_id'], features['pa_id'] = samples.ct_id, samples.pa_id
    features['dfs.month'], features['dfs.status'], features['split'] = samples.time, samples.event, samples.split
    aligned = join_teacher_tables(samples, features.iloc[::-1])
    assert aligned.ct_id.tolist() == samples.ct_id.tolist()
    features.loc[0, 'pa_id'] = 'wrong'
    with pytest.raises(ValueError, match='identity|labels'):
        join_teacher_tables(samples, features)


def test_teacher_attention_is_path_query_ct_key_value_and_masks_empty_group():
    torch.manual_seed(123)
    groups = [[0, 1], [2], [3], [], [4], [5], [6]]
    model = RadiomicsTeacher(groups, dropout=.25).eval()
    ct, pa = torch.randn(3, 7), torch.randn(3, 25, 1280)
    captured = []
    hook = model.pa_branch.compression.register_forward_pre_hook(lambda m, args: captured.append(args[0]))
    risk, info = model(ct, pa, return_attention=True)
    hook.remove()
    torch.testing.assert_close(captured[0], F.normalize(pa, p=2, dim=-1))
    assert risk.shape == (3,)
    assert info['attention'].shape == (3, 25, 7)
    assert info['ct_tokens'].shape == (3, 7, 256)
    torch.testing.assert_close(info['attention'].sum(-1), torch.ones(3, 25))
    assert torch.count_nonzero(info['attention'][:, :, 3]) == 0
    # Independently reconstruct MHA's actual internal Q/K projections.
    patches = model.pa_branch.compression(F.normalize(pa, p=2, dim=-1))
    query = model.pa_projection(patches)
    wq, wk, wv = model.coattn.in_proj_weight.chunk(3)
    bq, bk, bv = model.coattn.in_proj_bias.chunk(3)
    q, k, v = F.linear(query, wq, bq), F.linear(info['ct_tokens'], wk, bk), F.linear(info['ct_tokens'], wv, bv)
    logits = q @ k.transpose(-1, -2) / 16.
    logits[:, :, 3] = -torch.inf
    attention = logits.softmax(-1)
    torch.testing.assert_close(info['attention'], attention)
    retrieved = model.coattn.out_proj(attention @ v)
    weights = model.query_pool(retrieved).softmax(1)
    torch.testing.assert_close(info['guided_ct'], (retrieved * weights).sum(1))


def test_both_modalities_and_query_projection_receive_cox_gradients():
    from loss import cox_loss
    torch.manual_seed(7)
    model = RadiomicsTeacher([[i] for i in range(7)], dropout=.25)
    ct, pa = torch.randn(4, 7), torch.randn(4, 25, 1280)
    risk = model(ct, pa)
    loss = cox_loss(risk, torch.tensor([1., 2., 3., 4.]), torch.tensor([1, 0, 1, 1]))
    loss.backward()
    for module in [model.pa_branch.compression[0], model.pa_branch.attention[0],
                   model.pa_branch.post_compression[0], model.ct_encoders[0][0], model.pa_projection[0]]:
        assert torch.isfinite(module.weight.grad).all()
        assert module.weight.grad.abs().sum() > 0
    assert model.coattn.in_proj_weight.grad[:512].abs().sum() > 0
    assert not list(model.pa_branch.risk_head.parameters())
    assert all(parameter.requires_grad for parameter in model.parameters())


def test_query_changes_attention_and_ct_changes_risk_without_patient_mixing():
    torch.manual_seed(11)
    model = RadiomicsTeacher([[i] for i in range(7)], dropout=.25).eval()
    ct, pa = torch.randn(2, 7), torch.randn(2, 25, 1280)
    risk, original = model(ct, pa, return_attention=True)
    changed_pa = pa.clone(); changed_pa[0] = torch.randn_like(pa[0])
    _, changed = model(ct, changed_pa, return_attention=True)
    assert (changed['attention'][0] - original['attention'][0]).abs().max() > 0
    torch.testing.assert_close(changed['attention'][1], original['attention'][1])
    changed_ct = ct.clone(); changed_ct[0] += 3
    again = model(changed_ct, pa)
    assert not torch.isclose(again[0], risk[0], rtol=1e-5, atol=1e-7)
    torch.testing.assert_close(again[1], risk[1])
    with pytest.raises(ValueError):
        model(ct, pa[:, :24])


def test_nested_roles_exclude_outer_validation_and_match_source_protocol():
    samples = pd.DataFrame(dict(event=np.tile([0, 1], 60), time=np.arange(120.)+1))
    fit = np.arange(100)
    parts = teacher_partitions(samples, fit, seed=123)
    changed = samples.copy(); changed.loc[100:, 'event'] = 1
    repeated = teacher_partitions(changed, fit, seed=123)
    held_all = []
    for (a, b, c), (aa, bb, cc) in zip(parts, repeated):
        assert not set(a) & set(b) and not set(a) & set(c) and not set(b) & set(c)
        assert set(a) | set(b) | set(c) == set(fit)
        for x, y in [(a, aa), (b, bb), (c, cc)]:
            np.testing.assert_array_equal(x, y)
        held_all.extend(c)
    assert sorted(held_all) == fit.tolist()


def test_short_training_tail_eval_and_checkpoint_roundtrip(tmp_path):
    torch.set_num_threads(2)
    samples = pd.DataFrame(dict(pa_id=[f'p{i}' for i in range(14)], ct_id=[str(i) for i in range(14)],
                                case_id=[f'p{i}|{i}' for i in range(14)],
                                event=np.tile([1, 0], 7), time=np.arange(14.)+1))
    pathology = torch.randn(14, 25, 1280)
    raw = radiomics_table(14)
    dataset = PaRadiomicsDataset(samples, pathology, raw)
    args = SimpleNamespace(seed=123, dropout=.25, fusion_type='crossattention', lr=1e-4,
                           weight_decay=5e-4, cox_batch_size=4, num_epochs=2, patience=1,
                           num_workers=0)
    model, prep, metadata = fit_teacher(dataset, np.arange(10), None, args, torch.device('cpu'),
                                        tmp_path/'outer', fixed_epochs=2)
    assert metadata['training_risk_set_sizes'] == [4, 6]
    frame = predict_teacher(model, dataset, prep, np.arange(10, 14), args, torch.device('cpu'))
    loaded, same, stored = load_teacher(tmp_path/'outer', torch.device('cpu'))
    again = predict_teacher(loaded, dataset, same, np.arange(10, 14), args, torch.device('cpu'))
    np.testing.assert_allclose(frame.risk_score, again.risk_score, rtol=0, atol=0)
    assert len(frame) == 4 and np.isfinite(frame.risk_score).all()
    assert stored['selected_epoch'] == 2


def test_stop_patience_restores_selected_checkpoint_and_rejects_singleton_cox(tmp_path, monkeypatch):
    import teacher_train
    torch.set_num_threads(2)
    samples = pd.DataFrame(dict(case_id=[f'p{i}' for i in range(14)],
                                event=np.tile([1, 0], 7), time=np.arange(14.)+1))
    dataset = PaRadiomicsDataset(samples, torch.randn(14, 25, 1280), radiomics_table(14))
    args = SimpleNamespace(seed=123, dropout=.25, fusion_type='crossattention', lr=1e-4,
                           weight_decay=5e-4, cox_batch_size=4, num_epochs=5, patience=1, num_workers=0)
    original = teacher_train.cindex
    scores = iter([.8, .7])
    monkeypatch.setattr(teacher_train, 'cindex', lambda frame:
                        next(scores) if frame.case_id.iloc[0] == 'p10' else original(frame))
    model, prep, info = fit_teacher(dataset, np.arange(10), np.arange(10, 14), args,
                                    torch.device('cpu'), tmp_path/'selected')
    assert info['selected_epoch'] == 1 and info['epochs_completed'] == 2
    assert info['preprocessing']['fit_n'] == 10 and info['stop_n'] == 4
    saved = torch.load(tmp_path/'selected/best_model.pth', weights_only=True)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, saved[name], rtol=0, atol=0)
    args.cox_batch_size = 1
    with pytest.raises(ValueError, match='at least two'):
        fit_teacher(dataset, np.arange(10), None, args, torch.device('cpu'), tmp_path/'invalid', fixed_epochs=1)


def test_concat_control_keeps_both_modalities_and_common_initialization():
    torch.manual_seed(123)
    cross = RadiomicsTeacher([[i] for i in range(7)]).eval()
    torch.manual_seed(123)
    concat = RadiomicsTeacher([[i] for i in range(7)], fusion_type='concat').eval()
    torch.testing.assert_close(cross.head.net[0].weight, concat.head.net[0].weight)
    ct, pa = torch.randn(2, 7), torch.randn(2, 25, 1280)
    risk, info = concat(ct, pa, return_attention=True)
    assert risk.shape == (2,) and info['attention'] is None
    torch.testing.assert_close(info['guided_ct'], info['ct_tokens'].mean(1))
