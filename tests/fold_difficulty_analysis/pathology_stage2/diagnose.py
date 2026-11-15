"""Full-bag, fixed-patient UNI scale diagnosis. Never modifies shared training code."""

import argparse
import copy
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sksurv.metrics import concordance_index_censored

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from dataset import load_path_feature
from loss.survival_loss import cox_loss
from model.abmil_cox import ABMIL
from utils import seed_everything


def select_patients(frame, fold, count, seed):
    """Proportional event-stratified selection from saved outer TRAIN only."""
    if not frame['split'].isin(range(5)).all() or not frame['event'].isin([0, 1]).all():
        raise ValueError('Expected five saved folds and binary events')
    if frame['pa_id'].duplicated().any():
        raise ValueError('Duplicate pathology patients')
    pool = frame.loc[frame.split.ne(fold)]
    if fold not in range(5) or count < 2 or len(pool) < count:
        raise ValueError('Invalid fold or patient count')
    n_event = int(round(pool.event.mean() * count))
    if not 1 <= n_event < count:
        raise ValueError('Cox diagnosis requires both events and censoring')
    rng = np.random.default_rng(seed)
    indices = np.concatenate([
        rng.choice(pool.index[pool.event.eq(1)], n_event, replace=False),
        rng.choice(pool.index[pool.event.eq(0)], count - n_event, replace=False),
    ])
    rng.shuffle(indices)
    selected = frame.loc[indices].copy().reset_index(drop=True)
    if not np.isfinite(selected.time).all() or (selected.time < 0).any():
        raise ValueError('Invalid survival times')
    return selected


def transform(x, condition):
    if condition == 'raw':
        return x
    if condition in ('div40', 'div40_compensated'):
        return x / 40.
    if condition == 'l2':
        return F.normalize(x, p=2, dim=-1, eps=1e-12)
    raise ValueError(f'Unknown input: {condition}')


def compensated_model(model, divisor):
    other = copy.deepcopy(model)
    with torch.no_grad():
        other.compression[0].weight.mul_(divisor)
    return other


def risks(model, bags, condition, seed, training):
    device = next(model.parameters()).device
    model.train(training)
    with torch.random.fork_rng(devices=[device.index] if device.type == 'cuda' else []):
        torch.manual_seed(seed)
        with torch.no_grad():
            return torch.cat([model(transform(x.to(device), condition))[0] for x in bags])


def joint_gradients(model, bags, time, event, condition, seed):
    """Exact joint-Cox VJP with dropout replay; graph memory is one WSI.

    First compute dL/dr for the SAME 64-patient Cox objective, then replay every
    patient and backpropagate dr/dtheta weighted by that patient's dL/dr.
    This is not 64 singleton Cox losses or a sum of independent patient losses.
    """
    device = next(model.parameters()).device
    if len(bags) != time.numel() or time.numel() != event.numel() or not event.bool().any():
        raise ValueError('Cox inputs must align and contain an event')
    model.zero_grad(set_to_none=True)
    initial = risks(model, bags, condition, seed, True)
    leaf = initial.detach().requires_grad_(True)
    loss = cox_loss(leaf, time.to(device), event.to(device))
    derivative, = torch.autograd.grad(loss, leaf)
    with torch.random.fork_rng(devices=[device.index] if device.type == 'cuda' else []):
        torch.manual_seed(seed)
        for i, x in enumerate(bags):
            risk = model(transform(x.to(device), condition))[0]
            if not torch.equal(risk.detach(), initial[i:i+1]):
                raise RuntimeError('Dropout replay failed: joint-Cox gradient would be invalid')
            risk.backward(derivative[i:i+1])
    if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):
        raise RuntimeError('Missing or non-finite Cox gradient')
    return initial.detach(), loss.detach()


def summary(x):
    values = x.detach().cpu().numpy().reshape(-1)
    p5, p50, p95 = np.quantile(values, [.05, .5, .95])
    return dict(mean=float(values.mean(dtype=np.float64)),
                std=float(values.std(dtype=np.float64)),
                p5=float(p5), p50=float(p50), p95=float(p95))


def audit_forward(model, bags, time, event, condition, seed, training):
    """Native forward hooks; activation quantiles are exact per WSI then averaged."""
    device = next(model.parameters()).device
    model.train(training)
    captured = {}
    hooks = [
        model.compression[0].register_forward_hook(
            lambda module, inputs, output: captured.update(linear=output)),
        model.compression[1].register_forward_hook(
            lambda module, inputs, output: captured.update(relu=output)),
        model.post_compression.register_forward_pre_hook(
            lambda module, inputs: captured.update(pool=inputs[0])),
    ]
    scalars, all_risks, pools, embeddings, first_means, input_norms = [], [], [], [], [], []
    with torch.random.fork_rng(devices=[device.index] if device.type == 'cuda' else []):
        torch.manual_seed(seed)
        try:
            with torch.no_grad():
                for bag in bags:
                    x = transform(bag.to(device), condition)
                    risk, embedding, attention = model(x)
                    linear, relu = captured['linear'], captured['relu']
                    wx = F.linear(x, model.compression[0].weight, None)
                    bias_norm = model.compression[0].bias.norm().item()
                    weights = attention.reshape(-1)
                    entropy = -(weights * weights.clamp_min(1e-30).log()).sum()
                    normalized_entropy = entropy / math.log(len(weights)) if len(weights) > 1 else entropy * 0
                    stats = {f'linear_{key}_patient_mean': value for key, value in summary(linear).items()}
                    stats.update(
                        relu_active_fraction_patient_mean=float((relu > 0).double().mean()),
                        relu_std_patient_mean=float(relu.std(unbiased=False)),
                        wx_bias_ratio_patient_mean=float(wx.norm(dim=-1).mean() / (bias_norm + 1e-12)),
                        attention_normalized_entropy_mean=float(normalized_entropy),
                        attention_max_weight_mean=float(weights.max()),
                        attention_max_over_uniform_mean=float(weights.max() * len(weights)),
                        pool512_norm_mean=float(captured['pool'].norm()),
                        post256_norm_mean=float(embedding.norm()),
                    )
                    scalars.append(stats)
                    all_risks.append(risk.cpu())
                    pools.append(captured['pool'].cpu())
                    embeddings.append(embedding.cpu())
                    first_means.append(relu.mean(dim=1).cpu())
                    input_norms.append(x.norm(dim=-1).cpu())
        finally:
            for hook in hooks:
                hook.remove()
    r = torch.cat(all_risks)
    row = {key: float(np.mean([s[key] for s in scalars])) for key in scalars[0]}
    row.update({f'input_patch_norm_{key}': value for key, value in summary(torch.cat(input_norms)).items()})
    row.update({f'risk_{key}': value for key, value in summary(r).items()})
    for key, values in [('pool512', pools), ('post256', embeddings), ('relu_patient_mean', first_means)]:
        stacked = torch.cat(values)
        row[f'{key}_patient_variance_mean'] = float(stacked.var(dim=0, unbiased=False).mean())
    row['attention_normalized_entropy_patient_sd'] = float(np.std([
        s['attention_normalized_entropy_mean'] for s in scalars]))
    row.update(patients=len(bags), events=int(event.sum()), patches=sum(x.shape[0] for x in bags),
               cox_loss=float(cox_loss(r, time.cpu(), event.cpu())),
               cindex=float(concordance_index_censored(event.cpu().numpy().astype(bool),
                             time.cpu().numpy(), r.numpy())[0]))
    if not all(np.isfinite(v) for v in row.values()):
        raise RuntimeError('Non-finite forward diagnostic')
    return row, dict(risk=r, pool=torch.cat(pools), embedding=torch.cat(embeddings),
                     relu_patient_mean=torch.cat(first_means))


def adam_step(model, lr, weight_decay, input_divisor):
    """Independent Adam first step on identical saved data gradients."""
    other = copy.deepcopy(model)
    before = dict(model.named_parameters())
    for name, parameter in other.named_parameters():
        parameter.grad = before[name].grad.detach().clone()
    optimizer = torch.optim.Adam(other.parameters(), lr=lr, weight_decay=weight_decay,
                                 betas=(.9, .999), eps=1e-8, foreach=False)
    optimizer.step()
    rows = []
    for name, parameter in other.named_parameters():
        original = before[name].detach()
        gradient = before[name].grad.detach()
        decay = original * weight_decay
        delta = parameter.detach() - original
        gnorm, pnorm, dnorm = gradient.norm().item(), original.norm().item(), decay.norm().item()
        unorm = delta.norm().item()
        cosine = float(torch.dot(delta.flatten(), gradient.flatten()) / (unorm * gnorm)) if unorm and gnorm else 0.
        cosine = float(np.clip(cosine, -1., 1.))
        mapping_defined = input_divisor is not None or name != 'compression.0.weight'
        divisor = input_divisor if name == 'compression.0.weight' else 1.
        rows.append(dict(parameter=name, parameter_norm=pnorm, data_gradient_norm=gnorm,
                         decay_gradient_norm=dnorm, data_decay_ratio=gnorm / (dnorm + 1e-12) if weight_decay else 0.,
                         effective_gradient_norm=float((gradient + decay).norm()),
                         update_norm=unorm, relative_update=unorm / (pnorm + 1e-12),
                         effective_raw_parameter_update_norm=unorm / divisor if mapping_defined else 0.,
                         effective_raw_parameter_update_defined=mapping_defined,
                         update_data_gradient_cosine=cosine,
                         update_negative_gradient_angle_degrees=math.degrees(math.acos(-cosine)),
                         gradient_angle_defined=bool(unorm and gnorm),
                         weight_decay=weight_decay))
    return other, rows


def compare_outputs(left, right, fold, stage, mode, comparison, weight_decay):
    rows = []
    for name in left:
        difference = (left[name] - right[name]).abs()
        rows.append(dict(fold=fold, stage=stage, mode=mode, comparison=comparison,
                         weight_decay=weight_decay, output=name,
                         max_abs_difference=float(difference.max()),
                         rms_difference=float(difference.square().mean().sqrt()),
                         relative_l2_difference=float(difference.norm() / (left[name].norm() + 1e-12)),
                         within_tolerance=bool(torch.allclose(left[name], right[name], rtol=2e-4, atol=2e-6))))
    return rows


def write_report(output, activation, updates, equivalence, provenance, seed):
    lines = ['# UNI尺度：同初始化、同风险集的单步机制诊断', '',
             '本实验不评价泛化性能，也不使用外层验证集；全部指标来自固定的训练患者。',
             '没有加载已有训练checkpoint，正式模型、原始特征、旧结果均未修改。', '',
             f'配置：seed={seed}，ABMIL 1024→512，Attention128，池化512→256→1；Dropout=0.25。',
             '原项目Breslow Cox、按事件数平均；Adam lr=1e-4、betas=(0.9,0.999)、eps=1e-8。',
             '正式耦合衰减5e-4及零衰减对照均从同一初始参数、同一数据梯度独立执行一步。', '',
             '## 初始eval前向（Dropout关闭）', '',
             '|Fold|输入|输入范数均值|Linear SD¹|Wx/Bias²|Attention归一化熵|池化后患者方差³|Risk SD|C-index|',
             '|---|---|---|---|---|---|---|---|---|']
    for row in activation:
        if row['stage'] == 'before' and row['mode'] == 'eval':
            lines.append(f"|{row['fold']}|{row['condition']}|{row['input_patch_norm_mean']:.5g}|"
                         f"{row['linear_std_patient_mean']:.5g}|{row['wx_bias_ratio_patient_mean']:.5g}|"
                         f"{row['attention_normalized_entropy_mean']:.7f}|{row['post256_patient_variance_mean']:.5g}|"
                         f"{row['risk_std']:.5g}|{row['cindex']:.5f}|")
    lines += ['', '¹ 每位患者全部patch×512激活的总体标准差，再取患者均值；P5/P50/P95亦是每例精确分位数的患者均值。',
              '² 每位患者mean_patch‖Wx‖₂/(‖b‖₂+1e-12)，再取患者均值。',
              '³ post256每维在患者间的总体方差，再对维度平均；另存真正Attention池化后512维的同类指标。',
              '输入范数分位数为所选全部patch的合并分布。全部SD/方差使用总体定义ddof=0。', '',
              '## Adam首步：首层权重', '',
              '|Fold|输入|WD|数据梯度范数|数据/衰减梯度|相对参数更新|Raw输入下等效更新|更新相对负梯度夹角|',
              '|---|---|---|---|---|---|---|---|']
    for row in updates:
        if row['parameter'] == 'compression.0.weight':
            ratio = f"{row['data_decay_ratio']:.5g}" if row['weight_decay'] else '不适用（WD=0）'
            raw_update = f"{row['effective_raw_parameter_update_norm']:.5g}" if row['effective_raw_parameter_update_defined'] else '不适用（逐patch L2）'
            lines.append(f"|{row['fold']}|{row['condition']}|{row['weight_decay']:g}|{row['data_gradient_norm']:.5g}|"
                         f"{ratio}|{row['relative_update']:.5g}|{raw_update}|"
                         f"{row['update_negative_gradient_angle_degrees']:.3f}°|")
    lines += ['', '## 一步后的函数变化（仍为这64名训练患者，固定eval）', '',
              '|Fold|输入|WD|Risk SD前→后|Cox前→后|C-index前→后|',
              '|---|---|---|---|---|---|']
    for row in activation:
        if row['stage'] == 'after' and row['mode'] == 'eval':
            before = next(a for a in activation if a['fold'] == row['fold'] and
                          a['condition'] == row['condition'] and a['stage'] == 'before' and a['mode'] == 'eval')
            lines.append(f"|{row['fold']}|{row['condition']}|{row['weight_decay']:g}|"
                         f"{before['risk_std']:.5g}→{row['risk_std']:.5g}|{before['cox_loss']:.6f}→{row['cox_loss']:.6f}|"
                         f"{before['cindex']:.5f}→{row['cindex']:.5f}|")
    lines += ['', '## 测量所支持的机制与下一步', '']
    for fold in sorted({row['fold'] for row in activation}):
        initial = {row['condition']: row for row in activation if row['fold'] == fold
                   and row['stage'] == 'before' and row['mode'] == 'eval'}
        first = {row['condition']: row for row in updates if row['fold'] == fold
                 and row['weight_decay'] == 5e-4 and row['parameter'] == 'compression.0.weight'}
        raw, scaled, l2 = initial['raw'], initial['div40'], initial['l2']
        lines += [f"Fold {fold}：Raw→/40的首层激活SD为{raw['linear_std_patient_mean']:.6g}→{scaled['linear_std_patient_mean']:.6g}，"
                  f"Wx/Bias为{raw['wx_bias_ratio_patient_mean']:.5g}→{scaled['wx_bias_ratio_patient_mean']:.5g}。",
                  f"患者post256方差为{raw['post256_patient_variance_mean']:.6g}→{scaled['post256_patient_variance_mean']:.6g}；"
                  f"L2对应{l2['post256_patient_variance_mean']:.6g}。这些值直接度量了初始化表征尺度改变，并不等于患者不可区分。",
                  f"首层Raw和/40的数据梯度范数比为{first['raw']['data_gradient_norm'] / (first['div40']['data_gradient_norm'] + 1e-12):.5g}，"
                  f"实际Adam更新范数比为{first['raw']['update_norm'] / (first['div40']['update_norm'] + 1e-12):.5g}；"
                  '两者需分开解释，自适应更新不与梯度幅度成简单正比。', '']
    lines += ['若缩放组Wx/Bias仍大于1，就不能称Bias已占主导；若显著低于1，也只支持相对幅度变化，不能单凭它断言梯度消失。',
              '首先看/40与L2是否一致改变初始表征，再看实际一步函数变化；补偿组用于区分初始函数差异和参数化效应。',
              'WD=0与5e-4的函数差值也单列在function_equivalence.csv，避免把所有组间差别归因于衰减。',
              '当前不改学习率、优化器或模型；只有结合这些实测差异形成明确假设后，才值得进行预先固定更新预算的短程验证。']
    eq = [r for r in equivalence if r['stage'] == 'before' and
          r['comparison'] == 'raw_vs_div40_compensated']
    lines += ['', '## 功能等价与解释边界', '',
              f"Raw与补偿/40的初始eval及同Dropout前向等价检查：{sum(r['within_tolerance'] for r in eq)}/{len(eq)}项通过。",
              '功能等价检查比较Risk、真正pool512、post256和ReLU的患者均值；容差rtol=2e-4、atol=2e-6。',
              '各初始参数逐位克隆；只有补偿组首层weight乘40，bias保持不变。',
              '输入缩小与首层参数扩大使初始函数相同，但Adam参数化和L2惩罚不同；不能将它当作公平性能训练配置。',
              '有效Raw参数为W/input_divisor，因此补偿组首层实际更新需除40后才是同输入坐标下的更新。',
              '逐patch L2没有单一Raw线性参数映射：对应字段置0并以effective_raw_parameter_update_defined=False标识，不参与该比较。',
              'gradient_update.csv逐权重/偏置分开记录；WD=0时data_decay_ratio仅为有限数值占位，不能解释为衰减贡献。',
              '角度为实际更新与负数据梯度的夹角；undefined由gradient_angle_defined=False标记。',
              '功能等价表另保存初始/单步后三组输入差异及每组前后函数变化，不能把绝对Risk平移解释成排序改善。', '',
              '64人联合Cox使用两遍重算的链式法则，训练Dropout随机流每次恢复到相同seed；每个病例Risk逐位复现后才反传。',
              '小数据回归测试与直接保留所有患者计算图的Cox梯度及实际Adam更新对照；没有更改Cox公式或拆成病例级Cox。',
              '激活审计使用原ABMIL forward hooks，前后eval以及同Dropout train均记录。没有采样/截断patch。', '',
              '本实验只能定位初始化、局部梯度和首次Adam更新，不能证明后续梯度消失、某种归一化导致过拟合，',
              '也不能根据一次更新宣称改善了验证泛化。未启动60步短程对照或新的五折训练；是否继续需结合这些测量决定。', '',
              '## 可复核来源', '', '选择算法：保存的Outer Train内按事件比例抽样、固定顺序；与Outer Val无交集。',
              '只输出集合哈希，不公开患者ID、标签或风险；同CSV、代码和seed可重新生成所选患者。',
              '下列来源包含输入/输出完整性、代码哈希及运行版本；输入文件读取前后校验SHA256。', '', '```json',
              json.dumps(provenance, indent=2, ensure_ascii=False), '```', '']
    (output / 'report.md').write_text('\n'.join(lines))


def run_diagnostic(fold_inputs, output, seed):
    output = Path(output)
    filenames = ['input_activation.csv', 'gradient_update.csv', 'function_equivalence.csv', 'report.md']
    if any((output / name).exists() for name in filenames):
        raise FileExistsError('Refusing to overwrite existing stage2 results')
    output.mkdir(parents=True, exist_ok=True)
    activation, updates, equivalence, provenance = [], [], [], []
    for fold, base, bags, time, event, source in fold_inputs:
        outputs = {}
        for condition in ('raw', 'div40', 'l2', 'div40_compensated'):
            model = compensated_model(base, 40.) if condition == 'div40_compensated' else copy.deepcopy(base)
            divisor = None if condition == 'l2' else (40. if condition in ('div40', 'div40_compensated') else 1.)
            for training in (False, True):
                mode = 'train' if training else 'eval'
                row, values = audit_forward(model, bags, time, event, condition, seed, training)
                activation.append(dict(fold=fold, condition=condition, mode=mode,
                                       stage='before', weight_decay=0., **row))
                outputs[(condition, 'before', mode, 0.)] = values
            _, loss = joint_gradients(model, bags, time, event, condition, seed)
            print(f'Fold {fold} {condition}: joint Cox={float(loss):.7f}; gradient replay passed', flush=True)
            for wd in (0., 5e-4):
                updated, parameter_rows = adam_step(model, 1e-4, wd, divisor)
                updates.extend(dict(fold=fold, condition=condition, **row) for row in parameter_rows)
                for training in (False, True):
                    mode = 'train' if training else 'eval'
                    row, values = audit_forward(updated, bags, time, event, condition, seed, training)
                    activation.append(dict(fold=fold, condition=condition, mode=mode,
                                           stage='after', weight_decay=wd, **row))
                    outputs[(condition, 'after', mode, wd)] = values
                    equivalence.extend(compare_outputs(outputs[(condition, 'before', mode, 0.)], values,
                        fold, 'after', mode, f'{condition}_before_vs_after', wd))
        for mode in ('eval', 'train'):
            for stage, wd in [('before', 0.), ('after', 0.), ('after', 5e-4)]:
                for other in ('div40', 'l2', 'div40_compensated'):
                    equivalence.extend(compare_outputs(outputs[('raw', stage, mode, wd)],
                        outputs[(other, stage, mode, wd)], fold, stage, mode, f'raw_vs_{other}', wd))
            for condition in ('raw', 'div40', 'l2', 'div40_compensated'):
                equivalence.extend(compare_outputs(outputs[(condition, 'after', mode, 0.)],
                    outputs[(condition, 'after', mode, 5e-4)], fold, 'after', mode,
                    f'{condition}_wd0_vs_wd5e-4', 5e-4))
        checks = [r for r in equivalence if r['fold'] == fold and r['stage'] == 'before'
                  and r['comparison'] == 'raw_vs_div40_compensated']
        if not all(r['within_tolerance'] for r in checks):
            raise RuntimeError('Initial compensated-function equivalence failed')
        source['fold'] = fold
        provenance.append(source)
        print(f'Fold {fold}: all four conditions complete; initial function equivalence passed', flush=True)
    if not activation:
        raise ValueError('No diagnostic folds')
    for name, rows in zip(filenames, [activation, updates, equivalence]):
        pd.DataFrame(rows).to_csv(output / name, index=False)
    write_report(output, activation, updates, equivalence, provenance, seed)
    print(f'Completed: {len(provenance)} folds. Four result files: {output}', flush=True)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', type=Path, default=ROOT / 'data/trident/features_uni_v1/all_label_roi64.csv')
    parser.add_argument('--output', type=Path, default=Path(__file__).parent)
    parser.add_argument('--folds', type=int, nargs='+', default=[2, 4])
    parser.add_argument('--patients', type=int, default=64)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    torch.set_num_threads(4)
    seed_everything(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    csv_hash = sha256(args.csv)
    frame = pd.read_csv(args.csv, dtype={'pa_id': str, 'ct_id': str})
    if sha256(args.csv) != csv_hash:
        raise RuntimeError('CSV changed during read')
    if len(frame) != 721 or int(frame.event.sum()) != 251:
        raise ValueError('Expected the audited 721-patient, 251-event UNI cohort')
    if len(set(args.folds)) != len(args.folds):
        raise ValueError('Duplicate requested folds')
    device = torch.device(args.device)
    print(f'Starting full-bag one-step diagnosis: folds={args.folds}, patients={args.patients}, device={device}', flush=True)

    def fold_inputs():
        for fold in args.folds:
            selected = select_patients(frame, fold, args.patients, args.seed)
            files = [Path(path) for path in selected.pa_path]
            signatures = [sha256(path) for path in files]
            bags = [load_path_feature(path, feature_dim=1024) for path in files]
            seed_everything(args.seed)
            model = ABMIL(dropout=.25).to(device)
            init_hash = hashlib.sha256(b''.join(
                v.detach().cpu().numpy().tobytes() for v in model.state_dict().values())).hexdigest()
            norms = torch.cat([x.norm(dim=-1) for x in bags])
            nonzero = norms > 1e-12
            radial_error = ((norms[nonzero] / 40. - 1.).square().sum() / nonzero.sum()).sqrt()
            print(f'Fold {fold}: {len(bags)} TRAIN patients, {int(selected.event.sum())} events, '
                  f'{sum(len(x) for x in bags)} complete patches loaded', flush=True)
            source = dict(csv_sha256=csv_hash,
                          selected_order_labels_sha256=hashlib.sha256(
                              selected[['pa_id', 'ct_id', 'event', 'time', 'split']].to_csv(index=False).encode()).hexdigest(),
                          selected_features_sha256=hashlib.sha256(''.join(signatures).encode()).hexdigest(),
                          initial_parameters_sha256=init_hash, patients=len(bags), events=int(selected.event.sum()),
                          patches=sum(len(x) for x in bags),
                          div40_vs_l2_relative_frobenius_difference=float(radial_error),
                          zero_or_tiny_patch_count=int((~nonzero).sum()),
                          versions=dict(torch=torch.__version__, numpy=np.__version__, pandas=pd.__version__),
                          device=str(device), seed=args.seed, code_sha256={str(path.relative_to(ROOT)):sha256(path) for path in
                              [Path(__file__), ROOT/'model/abmil_cox.py', ROOT/'loss/survival_loss.py', ROOT/'dataset.py']})
            yield fold, model, bags, torch.tensor(selected.time.to_numpy(), dtype=torch.float32), \
                torch.tensor(selected.event.to_numpy(), dtype=torch.float32), source
            if signatures != [sha256(path) for path in files] or sha256(args.csv) != source['csv_sha256']:
                raise RuntimeError('Source features or CSV changed during the diagnosis')
            source['source_sha256_unchanged'] = True
    run_diagnostic(fold_inputs(), args.output, args.seed)


if __name__ == '__main__':
    main()
