"""Analyze every predeclared condition together; no checkpoint selection."""

import os
from pathlib import Path

import numpy as np
import pandas as pd


def analyze(root, make_plots=True):
    root = Path(root)
    folders = sorted(p for p in root.glob('fold_*') if p.is_dir())
    if not folders:
        folders = [root]
    frames = []
    for folder in folders:
        if not (folder/'completed.json').exists():
            raise RuntimeError(f'No completed fixed-budget fold: {folder}')
        frames.append(pd.read_csv(folder/'metrics.csv'))
    metrics = pd.concat(frames, ignore_index=True)
    differences = pd.concat([pd.read_csv(folder/'function_differences.csv') for folder in folders], ignore_index=True)
    if len(folders) > 1:
        metrics.to_csv(root/'metrics.csv', index=False)
        differences.to_csv(root/'function_differences.csv', index=False)
        pd.concat([pd.read_csv(f/'gradient_update.csv') for f in folders], ignore_index=True).to_csv(root/'gradient_update.csv', index=False)
    lines = ['# Stage3：输入尺度与优化轨迹验证', '',
             '仅Fold 2、4、完整Outer Train、UNI1024全WSI、ABMIL、Cox64且尾部合并、Dropout0.25。',
             '四组均Adam WD=0，seed123；固定60更新，不早停、不调度、不按验证结果选择权重。',
             'A=Raw/W；B=/40/W；C=/40/40W；D=/40/40W且仅首层weight学习率4e-3。其余参数lr1e-4。',
             '没有加载已训练单模态checkpoint，没有改动正式模型或损失。', '',
             '## 固定检查点指标', '',
             '|Fold|Group|Step|Train C-index|Val C-index|Train Cox|Val Cox|Train Risk SD|Val Risk SD|',
             '|---|---|---|---|---|---|---|---|---|']
    for (fold, group, step), part in metrics.groupby(['fold','group','step'], sort=True):
        a = part.loc[part.split.eq('train')].iloc[0]
        b = part.loc[part.split.eq('val')].iloc[0]
        lines.append(f"|{fold}|{group}|{step}|{a.cindex:.6f}|{b.cindex:.6f}|{a.cox_loss:.6f}|"
                     f"{b.cox_loss:.6f}|{a.risk_sd:.6g}|{b.risk_sd:.6g}|")
    lines += ['', '## 参数化补偿的函数差异', '',
              '所有预测均由固定参数的eval模型计算；RMSE包含风险整体平移，centered RMSE扣除平移。',
              'Spearman用于排序一致性；pool512/post256差值见function_differences.csv。', '',
              '|Fold|Split|Step|A/C Risk RMSE|A/D Risk RMSE|A/C Spearman|A/D Spearman|',
              '|---|---|---|---|---|---|---|']
    risks = differences.loc[differences.feature.eq('risk')]
    for (fold, split, step), part in risks.groupby(['fold','split','step'], sort=True):
        c = part.loc[part.comparison.eq('A_C')].iloc[0]
        d = part.loc[part.comparison.eq('A_D')].iloc[0]
        lines.append(f'|{fold}|{split}|{step}|{c.rmse:.7g}|{d.rmse:.7g}|{c.spearman:.7f}|{d.spearman:.7f}|')
    lines += ['', '## 按预先规则解释', '']
    for fold, part in risks.loc[risks.step.gt(0) & risks.split.eq('train')].groupby('fold'):
        paired = part.pivot(index='step', columns='comparison', values='centered_rmse')
        closer = int((paired['A_D'] < paired['A_C']).sum())
        lines += [f'Fold {fold}：在{len(paired)}个固定非零检查点中，D比C更接近A的训练Risk中心化RMSE有{closer}个。',
                  '这衡量轨迹接近程度，而非某组DFS性能更优。结合未中心化RMSE、Spearman和参数更新可判断补偿是否有效。', '']
    lines += ['预先判断规则：D持续接近A、C偏离，支持首层有效更新尺度的作用；C/D均偏离，说明仅补偿学习率不足。',
              'B早期慢于A、C/D较快，支持初始函数尺度影响早期学习；B若追上，只支持当前60步范围内的预算解释。',
              '60步不足以判断30轮预算是否不足，更不能解释180步的后期泛化下降。',
              '只含两折，不报告“五折均值”或据此重选正式配置，也不把临时最高验证点当成训练选择结果。', '',
              '## 随机控制及可复核性', '',
              '所有组使用同一预生成epoch患者随机排列、64人合尾批次和逐步Dropout种子。',
              '仅一个args.seed=123；患者与Dropout子随机流均由它派生。每步使用新的共同随机序列；',
              'Stage2的两遍VJP逐例检查训练Risk逐位重现，再以同一联合Cox的dL/dr反传。',
              '此随机协议用于条件间匹配，不声称复现历史训练的DataLoader随机轨迹。',
              '评估不使用Dropout、不消耗患者或Dropout随机流。Train/Val各自全部风险联合计算Cox及C-index。',
              '私有预测包含每个固定点的患者风险；checkpoint同时保存模型和持续Adam状态。没有best_model或外层选参。',
              'batch_schedule.csv仅发布每批患者顺序哈希，患者ID与权重位于被忽略的private目录。',
              'gradient_update.csv保存每一步所有权重/偏置的数据梯度范数、实际参数更新及Raw坐标下的首层更新。',
              'D恢复了学习率比例，但所有组eps仍为1e-8，因此不要求长期函数逐位等价。', '',
              '## 图表', '',
              'learning_curves_fold_2/4.png和PDF：固定eval下Train/Val C-index、Cox；',
              'function_differences.png和PDF：A/B、A/C、A/D的Risk RMSE及Spearman。',
              '仅连接实际固定检查点，无平滑、额外插值点或验证选轮。', '']
    (root/'report.md').write_text('\n'.join(lines))
    if make_plots:
        plot_results(metrics, risks, root)


def plot_results(metrics, risks, root):
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/pact_matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':16, 'font.weight':'bold', 'axes.labelweight':'bold',
                         'axes.titleweight':'bold', 'pdf.fonttype':42,
                         'axes.spines.top':False, 'axes.spines.right':False})
    colors = dict(A='#333333', B='#7c9bb5', C='#d69465', D='#9474ad')
    labels = dict(A='A: Raw', B='B: /40', C='C: /40, 40W', D='D: /40, 40W, 40LR')
    for fold, frame in metrics.groupby('fold'):
        fig, axes = plt.subplots(2,2, figsize=(17,11), constrained_layout=True)
        for ax, (split, metric, ylabel) in zip(axes.flat, [
                ('train','cindex','Train C-index'), ('val','cindex','Val C-index'),
                ('train','cox_loss','Train Cox loss'), ('val','cox_loss','Val Cox loss')]):
            for group in ('A','B','C','D'):
                data = frame.loc[frame.split.eq(split) & frame.group.eq(group)].sort_values('step')
                ax.plot(data.step, data[metric], color=colors[group], lw=3, marker='o', ms=5, label=labels[group])
            ax.set(xlabel='Optimizer step', ylabel=ylabel)
            ax.grid(alpha=.15)
        axes[0,0].legend(fontsize=12, frameon=False)
        fig.suptitle(f'Fold {fold}: fixed evaluation checkpoints', fontsize=21, fontweight='bold')
        for extension in ('png','pdf'):
            fig.savefig(root/f'learning_curves_fold_{fold}.{extension}', dpi=180, bbox_inches='tight')
        plt.close(fig)
    folds = sorted(risks.fold.unique())
    fig, axes = plt.subplots(len(folds), 2, figsize=(17,5*len(folds)), squeeze=False, constrained_layout=True)
    for row, fold in enumerate(folds):
        for col, metric in enumerate(('rmse','spearman')):
            ax = axes[row,col]
            for group in ('B','C','D'):
                for split, style in [('train','-'), ('val','--')]:
                    part = risks.loc[risks.fold.eq(fold)&risks.comparison.eq(f'A_{group}')&risks.split.eq(split)].sort_values('step')
                    ax.plot(part.step, part[metric], color=colors[group], lw=3, linestyle=style,
                            marker='o', ms=4, label=f'A vs {group} ({split})')
            ax.set(xlabel='Optimizer step', ylabel='Risk RMSE' if col == 0 else 'Risk Spearman', title=f'Fold {fold}')
            ax.grid(alpha=.15)
    axes[0,0].legend(fontsize=11, frameon=False, ncol=2)
    for extension in ('png','pdf'):
        fig.savefig(root/f'function_differences.{extension}', dpi=180, bbox_inches='tight')
    plt.close(fig)
