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


def inputs():
    torch.manual_seed(123)
    base = ABMIL(in_dim=5, hidden_dim=7, attention_dim=3,
                 post_hidden_dim=4, dropout=.25).double()
    bags = [torch.randn(n, 5, dtype=torch.double) for n in (4, 6, 7, 5, 8, 3, 6, 5)]
    samples = pd.DataFrame(dict(pa_id=[f'p{i}' for i in range(8)],
                                split=[2, 4, 2, 1, 3, 0, 1, 3],
                                time=[2., 3., 5., 8., 2., 7., 6., 9.],
                                event=[1, 1, 0, 1, 1, 0, 1, 0]))
    return base, bags, samples


def test_groups_preserve_parameters_except_compensated_weight_and_d_weight_lr():
    from train_scale import create_groups
    base, bags, samples = inputs()
    models, optimizers = create_groups(base)
    for group, model in models.items():
        for name, value in model.state_dict().items():
            factor = 40 if group in ('C', 'D') and name == 'compression.0.weight' else 1
            torch.testing.assert_close(value, base.state_dict()[name] * factor, rtol=0, atol=0)
        rates = {id(p): g['lr'] for g in optimizers[group].param_groups for p in g['params']}
        for name, p in model.named_parameters():
            assert rates[id(p)] == (4e-3 if group == 'D' and name == 'compression.0.weight' else 1e-4)
        assert all(g['weight_decay'] == 0 and g['eps'] == 1e-8 for g in optimizers[group].param_groups)


def test_schedule_full_pool_tail_merge_reproducible_and_train_only():
    from train_scale import batch_schedule
    indices = np.arange(577)
    a = batch_schedule(indices, 64, 60, 123)
    b = batch_schedule(indices, 64, 60, 123)
    assert len(a) == 60
    assert [len(entry['indices']) for entry in a[:9]] == [64]*8 + [65]
    assert sorted(np.concatenate([entry['indices'] for entry in a[:9]])) == indices.tolist()
    assert [entry['dropout_seed'] for entry in a] == [entry['dropout_seed'] for entry in b]
    assert len(set(entry['dropout_seed'] for entry in a)) == 60
    for x, y in zip(a, b):
        np.testing.assert_array_equal(x['indices'], y['indices'])


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_three_persistent_adam_steps_equal_direct_joint_cox_graph(device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('GPU unavailable')
    from train_scale import joint_gradients, create_groups, train_step
    base, bags, samples = inputs()
    base = base.to(device)
    models, optimizers = create_groups(base)
    direct = copy.deepcopy(models['D'])
    reference_optimizer = torch.optim.Adam([
        {'params': [direct.compression[0].weight], 'lr': 4e-3},
        {'params': [p for p in direct.parameters() if p is not direct.compression[0].weight], 'lr': 1e-4},
    ], weight_decay=0., foreach=False)
    time = torch.tensor(samples.time.values[:4], dtype=torch.double, device=device)
    event = torch.tensor(samples.event.values[:4], dtype=torch.double, device=device)
    for seed in [123, 456, 789]:
        direct.train()
        reference_optimizer.zero_grad()
        torch.manual_seed(seed)
        risk = torch.cat([direct(bag.to(device)/40.)[0] for bag in bags[:4]])
        loss = cox_loss(risk, time, event)
        loss.backward()
        reference_optimizer.step()
        rows = train_step(models['D'], optimizers['D'], bags[:4], time, event, 'div40', seed, 'D')
        assert len(rows) == len(list(direct.parameters()))
        for p, q in zip(direct.parameters(), models['D'].parameters()):
            torch.testing.assert_close(p, q, atol=1e-11, rtol=1e-9)


def test_fixed_eval_preserves_rng_and_uses_full_patient_risk_set(tmp_path):
    from scale_eval import evaluate
    base, bags, samples = inputs()
    before = torch.get_rng_state().clone()
    row, values = evaluate(base, bags, samples, list(range(8)), 'raw', tmp_path/'pred.csv')
    assert torch.equal(before, torch.get_rng_state())
    assert not base.training
    frame = pd.read_csv(tmp_path/'pred.csv')
    assert len(frame) == 8
    torch.testing.assert_close(torch.tensor(frame.risk.values), values['risk'])
    expected = cox_loss(values['risk'], torch.tensor(samples.time.values), torch.tensor(samples.event.values))
    assert row['cox_loss'] == float(expected)
    assert row['post256_patient_variance_mean'] > 0


def test_short_four_group_run_has_fixed_checks_no_val_selection_and_equivalence(tmp_path):
    from train_scale import run_fold
    base, bags, samples = inputs()
    run_fold(base, bags, samples, 2, tmp_path, 123, max_steps=2, cox_batch_size=3)
    metrics = pd.read_csv(tmp_path/'metrics.csv')
    assert set(metrics.step) == {0, 1, 2}
    assert len(metrics) == 4 * 3 * 2
    assert set(metrics.group) == {'A', 'B', 'C', 'D'}
    diffs = pd.read_csv(tmp_path/'function_differences.csv')
    initial = diffs[diffs.step.eq(0) & diffs.comparison.isin(['A_C', 'A_D'])]
    assert initial.within_tolerance.all()
    assert not list(tmp_path.rglob('best_model.pth'))
    assert len(list((tmp_path/'private').glob('*/step_*/model.pth'))) == 12
    from scale_analysis import analyze
    analyze(tmp_path, make_plots=False)
    assert (tmp_path/'report.md').is_file()
    with pytest.raises(FileExistsError):
        run_fold(base, bags, samples, 2, tmp_path, 123, max_steps=2, cox_batch_size=3)


def test_analysis_can_be_repeated_when_root_contains_aggregate_metrics(tmp_path):
    from train_scale import run_fold
    from scale_analysis import analyze
    base, bags, samples = inputs()
    folder = tmp_path/'fold_2'
    run_fold(base, bags, samples, 2, folder, 123, max_steps=1, cox_batch_size=3)
    (tmp_path/'metrics.csv').write_bytes((folder/'metrics.csv').read_bytes())
    analyze(tmp_path, make_plots=False)
    analyze(tmp_path, make_plots=False)
    assert (tmp_path/'report.md').is_file()


def test_validation_labels_do_not_change_training_or_adam_states(tmp_path):
    from train_scale import run_fold
    base, bags, samples = inputs()
    changed = samples.copy()
    changed.loc[changed.split.eq(2), 'time'] = [1., 20.]
    for name, frame in [('original', samples), ('changed', changed)]:
        run_fold(base, bags, frame, 2, tmp_path/name, 123, max_steps=2, cox_batch_size=3)
    for group in ('A','B','C','D'):
        a, b = [torch.load(tmp_path/n/'private'/group/'step_002/model.pth', weights_only=True)
                for n in ('original', 'changed')]
        for key, value in a['model'].items():
            assert torch.equal(value, b['model'][key])
        for key, state in a['optimizer']['state'].items():
            for name, value in state.items():
                assert torch.equal(value, b['optimizer']['state'][key][name])
