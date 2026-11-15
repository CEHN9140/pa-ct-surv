import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from model.abmil_cox import ABMIL
from loss.survival_loss import cox_loss


def fixture():
    torch.manual_seed(123)
    model = ABMIL(in_dim=5, hidden_dim=7, attention_dim=3,
                  post_hidden_dim=4, dropout=0.25).double()
    bags = [torch.randn(n, 5, dtype=torch.double) for n in (4, 7, 5, 8)]
    time = torch.tensor([2., 2., 5., 9.], dtype=torch.double)
    event = torch.tensor([1., 1., 0., 1.], dtype=torch.double)
    return model, bags, time, event


def test_selection_is_training_only_stratified_and_reproducible():
    from diagnose import select_patients
    frame = pd.DataFrame(dict(pa_id=np.arange(100).astype(str),
                              split=np.arange(100) % 5,
                              event=np.arange(100) % 3 == 0,
                              time=np.arange(100) + 1))
    a = select_patients(frame, 2, 64, 123)
    pd.testing.assert_frame_equal(a, select_patients(frame, 2, 64, 123))
    assert len(a) == 64 and a.pa_id.is_unique and not a.split.eq(2).any()
    assert a.event.sum() == round(frame.loc[frame.split.ne(2), 'event'].mean() * 64)
    with pytest.raises(ValueError):
        select_patients(frame.assign(event=0), 2, 64, 123)


def test_compensated_model_and_dropout_are_functionally_equivalent():
    from diagnose import transform, risks, compensated_model
    model, bags, time, event = fixture()
    other = compensated_model(model, 40)
    for training in (False, True):
        r = risks(model, bags, 'raw', 123, training)
        rc = risks(other, bags, 'div40', 123, training)
        torch.testing.assert_close(r, rc, rtol=1e-11, atol=1e-12)
        assert torch.equal(r, risks(model, bags, 'raw', 123, training))
    assert torch.equal(model.compression[0].bias, other.compression[0].bias)
    torch.testing.assert_close(transform(bags[0], 'l2').norm(dim=-1),
                               torch.ones(4, dtype=torch.double))


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_recomputed_joint_cox_gradients_equal_full_graph_with_dropout(device):
    from diagnose import joint_gradients
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('GPU unavailable in this process')
    model, bags, time, event = fixture()
    model = model.to(device)
    bags = [x.to(device) for x in bags]
    time, event = time.to(device), event.to(device)
    direct = copy.deepcopy(model).train()
    torch.manual_seed(456)
    full_risk = torch.cat([direct(x)[0] for x in bags])
    loss = cox_loss(full_risk, time, event)
    loss.backward()
    r, computed_loss = joint_gradients(model, bags, time, event, 'raw', 456)
    assert torch.equal(r, full_risk.detach())
    torch.testing.assert_close(computed_loss, loss.detach())
    for a, b in zip(model.parameters(), direct.parameters()):
        torch.testing.assert_close(a.grad, b.grad, atol=1e-12, rtol=1e-10)
    for candidate in (model, direct):
        torch.optim.Adam(candidate.parameters(), lr=1e-4, weight_decay=5e-4).step()
    for a, b in zip(model.parameters(), direct.parameters()):
        torch.testing.assert_close(a, b, atol=1e-12, rtol=1e-10)


def test_adam_step_is_real_update_and_does_not_mutate_source():
    from diagnose import joint_gradients, adam_step
    model, bags, time, event = fixture()
    joint_gradients(model, bags, time, event, 'raw', 123)
    before = copy.deepcopy(model.state_dict())
    updated, rows = adam_step(model, lr=1e-4, weight_decay=5e-4, input_divisor=1.)
    reference = copy.deepcopy(model)
    for p, q in zip(reference.parameters(), model.parameters()):
        p.grad = q.grad.clone()
    torch.optim.Adam(reference.parameters(), lr=1e-4, weight_decay=5e-4).step()
    for key, value in before.items():
        assert torch.equal(value, model.state_dict()[key])
        torch.testing.assert_close(updated.state_dict()[key], reference.state_dict()[key])
    assert len(rows) == len(list(model.parameters()))
    assert all(row['update_norm'] > 0 for row in rows)
    assert all(np.isfinite(row['relative_update']) for row in rows)


def test_forward_audit_matches_native_output_and_measures_patient_variance():
    from diagnose import audit_forward, risks
    model, bags, time, event = fixture()
    row, outputs = audit_forward(model, bags, time, event, 'raw', 123, False)
    torch.testing.assert_close(outputs['risk'], risks(model, bags, 'raw', 123, False))
    assert row['patients'] == 4 and row['patches'] == 24
    assert 0 <= row['relu_active_fraction_patient_mean'] <= 1
    assert 0 <= row['attention_normalized_entropy_mean'] <= 1 + 1e-10
    assert row['pool512_patient_variance_mean'] > 0
    assert row['linear_std_patient_mean'] > 0


def test_compensated_gradient_follows_chain_rule_and_l2_has_no_raw_linear_mapping():
    from diagnose import compensated_model, joint_gradients, adam_step
    model, bags, time, event = fixture()
    other = compensated_model(model, 40.)
    joint_gradients(model, bags, time, event, 'raw', 123)
    joint_gradients(other, bags, time, event, 'div40', 123)
    for (name, p), (_, q) in zip(model.named_parameters(), other.named_parameters()):
        expected = p.grad / 40. if name == 'compression.0.weight' else p.grad
        torch.testing.assert_close(expected, q.grad, atol=1e-12, rtol=1e-10)
    _, rows = adam_step(model, 1e-4, 5e-4, None)
    first = next(r for r in rows if r['parameter'] == 'compression.0.weight')
    assert not first['effective_raw_parameter_update_defined']


def test_pipeline_outputs_only_four_result_files_and_refuses_overwrite(tmp_path):
    from diagnose import run_diagnostic
    model, bags, time, event = fixture()
    provenance = {'synthetic': True}
    run_diagnostic([(2, model, bags, time, event, provenance)], tmp_path, 123)
    assert {p.name for p in tmp_path.iterdir()} == {
        'input_activation.csv', 'gradient_update.csv',
        'function_equivalence.csv', 'report.md'}
    a = pd.read_csv(tmp_path / 'input_activation.csv')
    assert len(a) == 24  # four inputs: eval/train initial + eval/train after each WD
    g = pd.read_csv(tmp_path / 'gradient_update.csv')
    assert set(g.weight_decay) == {0., 5e-4}
    assert np.isfinite(g.select_dtypes('number')).all().all()
    f = pd.read_csv(tmp_path / 'function_equivalence.csv')
    initial = f[f.stage.eq('before') & f.comparison.eq('raw_vs_div40_compensated')]
    assert len(initial) == 8 and initial.within_tolerance.all()
    with pytest.raises(FileExistsError):
        run_diagnostic([], tmp_path, 123)
