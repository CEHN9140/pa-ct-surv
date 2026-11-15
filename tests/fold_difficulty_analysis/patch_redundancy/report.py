"""Fold-wise metrics and Monte Carlo uncertainty for frozen patch diagnosis."""

from pathlib import Path

import numpy as np
import pandas as pd


def summarize(metrics,output,config):
    output=Path(output)
    expected={('full',0,-1)}
    expected|={(strategy,k,-1) for strategy in ('top','bottom') for k in config['ks']}
    expected|={('random',k,repeat) for k in config['ks'] for repeat in range(config['random_repeats'])}
    if set(metrics.fold)!=set(range(5)): raise ValueError('Five complete folds required')
    for fold,group in metrics.groupby('fold'):
        keys=list(zip(group.strategy,group.k,group.repeat))
        if len(keys)!=len(set(keys)) or set(keys)!=expected: raise ValueError(f'Incomplete strategy grid in fold {fold}')
    if not np.isfinite(metrics.cindex).all(): raise ValueError('Invalid C-index')
    baseline=metrics.loc[metrics.strategy.eq('full')].set_index('fold').cindex
    rows=[]
    for (strategy,k),group in metrics.groupby(['strategy','k'],sort=False):
        foldmeans=group.groupby('fold')[['cindex','spearman']].mean()
        repeatmeans=group.groupby('repeat').cindex.mean()
        spearman_repeat=group.groupby('repeat').spearman.mean()
        row=dict(strategy=strategy,k=int(k),cindex_mean=float(foldmeans.cindex.mean()),
                 cindex_fold_sd=float(foldmeans.cindex.std(ddof=1)),
                 cindex_repeat_sd=float(repeatmeans.std(ddof=1)) if strategy=='random' else np.nan,
                 spearman_mean=float(foldmeans.spearman.mean()) if group.spearman.notna().all() else np.nan,
                 spearman_fold_sd=float(foldmeans.spearman.std(ddof=1)) if group.spearman.notna().all() else np.nan,
                 spearman_repeat_sd=float(spearman_repeat.std(ddof=1)) if strategy=='random' and group.spearman.notna().all() else np.nan,
                 delta_full_mean=float((foldmeans.cindex-baseline).mean()),
                 positive_folds=int((foldmeans.cindex-baseline).gt(0).sum()),
                 mean_kept=float(group.mean_kept.mean()),full_retained_fraction=float(group.full_retained_fraction.mean()))
        rows.append(row)
    summary=pd.DataFrame(rows);summary.to_csv(output/'summary.csv',index=False)
    contrasts=[]
    for k in config['ks']:
        random=metrics.loc[metrics.strategy.eq('random')&metrics.k.eq(k)].pivot(index='fold',columns='repeat',values='cindex')
        for strategy in ('top','bottom'):
            chosen=metrics.loc[metrics.strategy.eq(strategy)&metrics.k.eq(k)].set_index('fold').cindex
            differences=random.rsub(chosen,axis=0)
            contrasts.append(dict(contrast=f'{strategy}_minus_random',k=k,delta_cindex=float(differences.to_numpy().mean()),
                                  delta_fold_sd=float(differences.mean(axis=1).std(ddof=1)),
                                  delta_repeat_sd=float(differences.mean(axis=0).std(ddof=1)),
                                  positive_folds=int(differences.mean(axis=1).gt(0).sum())))
    contrasts=pd.DataFrame(contrasts);contrasts.to_csv(output/'strategy_contrasts.csv',index=False)
    lines=['# Patch冗余与Attention有效性：冻结模型推理诊断','',
           '仅复用Stage6五个Raw UNI最终Outer模型；721例、251事件；无网络训练、无新模型选择。',
           '所有patch保留策略均在eval模式下对剩余patch重新Softmax；不足K的患者保留全部。',
           f"K={config['ks']}；Random每K重复{config['random_repeats']}次，seed={config['seed']}，无放回。",'',
           '|策略|K|Val C-index（五折mean±SD）|Random重复SD|与Full Spearman（五折mean±SD）|ΔC vs Full|',
           '|---|---:|---|---|---|---|']
    for row in summary.itertuples():
        jitter=f'{row.cindex_repeat_sd:.4f}' if row.strategy=='random' else '—'
        rho=f'{row.spearman_mean:.4f} ± {row.spearman_fold_sd:.4f}' if np.isfinite(row.spearman_mean) else '未定义（常数风险）'
        lines.append(f'|{row.strategy}|{row.k or "All"}|{row.cindex_mean:.4f} ± {row.cindex_fold_sd:.4f}|{jitter}|{rho}|{row.delta_full_mean:+.4f}|')
    lines+=['','Random的五折SD：先对每折30次结果取均值，再计算五折样本SD。',
            'Random重复SD：每次重复先取五折C-index均值，再计算30次样本SD；它不是患者Bootstrap置信区间。',
            '各重复的C-index分别计算后取平均，没有先平均30次风险来制造集成模型。',
            'Spearman逐折计算，不跨Fold混合风险；常数风险时不伪造相关系数。','',
            '|对照|K|平均ΔC|正增益Fold|','|---|---:|---:|---|']
    for row in contrasts.itertuples():
        lines.append(f'|{row.contrast}|{row.k}|{row.delta_cindex:+.4f}|{row.positive_folds}/5|')
    lines+=['','## Attention与实际删减幅度','',
            'attention_fold_summary.csv与attention_summary.csv记录Full Attention归一化熵、有效patch数/比例、',
            'Top-K Attention Mass、patch数及N≤K患者比例。指标按患者等权汇总，避免大WSI主导统计。',
            'N_eff=1/sum(a²)，不与exp(entropy)混用；单patch归一化熵约定0。','',
            '## 解释边界','',
            'Full预测必须重现Stage6存档。筛选来自当前ABMIL自身Attention，不是CHIEF评分。',
            '预测对随机删减稳定仅支持预测层面的冗余，不证明被删组织无用或质量差。',
            'Top-K优于Random-K只能支持本冻结模型的Attention筛选优势，不能证明重新训练Top-K会改善泛化。',
            'Bottom-K表现良好不等于Attention失效，因为重新归一化改变输入分布，低权重patch仍可能有信息。',
            '所有比较为同队列反复探索后的机制诊断；不按Outer Val挑K，不宣称独立验证或生物学对应关系。',
            '本次不含WSI组织质量的人工审查、CHIEF坐标映射或任何新的训练实验。','']
    (output/'report.md').write_text('\n'.join(lines))
    return summary,contrasts


def attention_summary(attention,output,ks):
    output=Path(output);attention=attention.copy()
    for k in ks: attention[f'fraction_N_le_{k}']=attention.patch_count.le(k).astype(float)
    columns=['patch_count','normalized_entropy','effective_patches','effective_ratio']
    columns += [f'top_mass_{k}' for k in ks]+[f'fraction_N_le_{k}' for k in ks]
    rows=[]
    for fold,group in attention.groupby('fold'):
        for name in columns:
            values=group[name]
            rows.append(dict(fold=fold,metric=name,n=len(values),mean=values.mean(),median=values.median(),
                             q25=values.quantile(.25),q75=values.quantile(.75),minimum=values.min(),maximum=values.max()))
    table=pd.DataFrame(rows);table.to_csv(output/'attention_fold_summary.csv',index=False)
    summary=table.groupby('metric')['mean'].agg(['mean','std']).reset_index().rename(columns={'mean':'fold_mean','std':'fold_sd'})
    summary.to_csv(output/'attention_summary.csv',index=False)
    with (output/'report.md').open('a') as stream:
        stream.write('\n## Full Attention统计（五折患者均值的mean±SD）\n\n|指标|结果|\n|---|---|\n')
        for row in summary.itertuples():
            stream.write(f'|{row.metric}|{row.fold_mean:.4f} ± {row.fold_sd:.4f}|\n')
