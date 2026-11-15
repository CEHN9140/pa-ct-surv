"""Aggregate-only report and scientific plots for Stage4; patient IDs stay private."""

import os

import numpy as np
import pandas as pd


def write_report(root, tables):
    summary=tables['fold_pair_summary']
    val=summary.loc[summary.split.eq('val')]
    groups=tables['pair_strata'].loc[lambda d:d.split.eq('val')]
    learning=tables['learning_summary']
    lines=['# Stage4：后期过拟合与患者对错误定位', '',
           '本次仅复用实验10已有结果，未启动任何神经网络或Ridge-Cox训练，也未启动Stage5。',
           '原721例、251个事件、保存五折；主分析full_repeat0，其他128/256/384人组用于学习曲线。',
           '所有Train指标来自固定checkpoint的eval；Best由Outer Val选择，Last固定180步。', '',
           '## 五折Best→Last：排序收益与损失', '',
           '|Fold|Best step|Train C-index Best→Last|Val C-index Best→Last|Val ΔC|完整纠正|完整破坏|可比较对|',
           '|---|---|---|---|---|---|---|---|']
    for row in val.itertuples():
        train=summary.loc[summary.fold.eq(row.fold)&summary.split.eq('train')].iloc[0]
        lines.append(f'|{row.fold}|{row.best_step}|{train.best_cindex:.4f}→{train.last_cindex:.4f}|'
                     f'{row.best_cindex:.4f}→{row.last_cindex:.4f}|{row.delta_cindex:+.5f}|'
                     f'{row.corrected_full}|{row.broken_full}|{row.pairs}|')
    lines += ['', f"五折验证均值：{val.best_cindex.mean():.4f} ± {val.best_cindex.std(ddof=1):.4f}"
              f" → {val.last_cindex.mean():.4f} ± {val.last_cindex.std(ddof=1):.4f}。",
              '表内完整纠正为0→1、完整破坏为1→0；风险tie计0.5分，其余四种半分转变另存汇总CSV。',
              'ΔC严格等于(所有获得分数−所有丢失分数)/该折可比较对数；每折分别复现sksurv。',
              'Best来自20次Outer Val检查，下降幅度含选模乐观偏差，不能作为独立验证效应量。', '',
              '## 错误发生的患者对类型与时间', '',
              '事件时间分组为[0,12]、(12,24]、(24,36]、>36月；同时间event/censor仍可比较，单列一类。',
              '下表先按事件时间合并患者对类型。ΔC贡献使用整折全部可比较对作分母，因此各时间组贡献可相加。', '',
              '|Fold|事件月数|可比较对|Best→Last C-index|完整破坏/完整纠正|ΔC贡献|原正确对被破坏比例|',
              '|---|---|---|---|---|---|---|']
    for (fold, bin_name), part in groups.groupby(['fold','event_time_bin']):
        n=int(part.pairs.sum())
        best=float((part.best_cindex*part.pairs).sum()/n)
        last=float((part.last_cindex*part.pairs).sum()/n)
        correct=int(part.best_correct_pairs.sum())
        broken=int(part.broken_full.sum())
        lines.append(f'|{fold}|{bin_name}|{n}|{best:.4f}→{last:.4f}|{broken}/{int(part.corrected_full.sum())}|'
                     f'{part.delta_cindex_contribution.sum():+.5f}|{broken/correct:.1%}|' if correct else
                     f'|{fold}|{bin_name}|{n}|{best:.4f}→{last:.4f}|{broken}/{int(part.corrected_full.sum())}|'
                     f'{part.delta_cindex_contribution.sum():+.5f}|NA|')
    lines += ['', '按Event–Event、Event–Later Censor及Event–Same-time Censor的完整交叉分解见pair_strata.csv。',
              '不能仅凭某一时间组破坏对数多就认定该组更难；需同时查看比较对数量、原正确对分母及相对破坏率。', '']
    for fold, part in groups.groupby('fold'):
        total=part.lost_credit.sum()
        late=part.loc[part.event_time_bin.eq('>36')]
        later=part.loc[part.pair_type.eq('event_later_censor')]
        lines.append(f"Fold {fold}：>36月事件患者对占全部可比较对{late.pairs.sum()/part.pairs.sum():.1%}，"
                     f"贡献丢失分数{late.lost_credit.sum()/total:.1%}；"
                     f"Event–Later Censor贡献丢失分数{later.lost_credit.sum()/total:.1%}。")
    part=groups.loc[groups.fold.eq(2)]
    early=-part.loc[part.event_time_bin.eq('0-12'),'net_credit'].sum()
    late=-part.loc[part.event_time_bin.eq('>36'),'net_credit'].sum()
    net=-part.net_credit.sum()
    lines += ['', f'重点结论：Fold2的净损失为{net:.0f}分，0–12月贡献{early:.0f}分（{early/net:.1%}），'
              f'>36月贡献{late:.0f}分（{late/net:.1%}）。',
              '本次结果不支持把训练后期退化主要归因于晚期事件。Fold4的晚期事件患者对原本就比较困难，',
              '但“基线持续较难”和“训练后期新增排序损害”是两个不同的问题。']
    lines += ['', '## 是否集中在少数患者、原本间隔是否很小', '',
              '患者按涉及的丢失分数排序，选取每折ceil(10%×患者数)；下面的覆盖率对患者对取并集，仅计一次。',
              '比较机会覆盖率表示同一患者集合覆盖全部可比较对的比例，作为描述性参照，并非随机检验。', '',
              '|Fold|净新增错误的患者数/验证人数|Top10%人数|覆盖丢失分数|覆盖全部比较机会|',
              '|---|---|---|---|---|']
    conc=tables['patient_concentration']
    for row in val.itertuples():
        c=conc.loc[conc.fold.eq(row.fold)&conc.split.eq('val')&conc.patient_fraction.eq(.1)].iloc[0]
        lines.append(f'|{row.fold}|{row.patients_with_net_added_errors}/{row.patients}|{int(c.patients_selected)}|'
                     f'{c.lost_credit_coverage:.1%}|{c.comparable_pair_coverage:.1%}|')
    margins=tables['risk_margin_breakdown'].loc[lambda d:d.split.eq('val')]
    for fold,part in margins.groupby('fold'):
        total=part.broken_full.sum()
        near=part.loc[part.margin_bin.isin(['0-.1','.1-.25','.25-.5']),'broken_full'].sum()
        far=part.loc[part.margin_bin.isin(['1-2','>=2']),'broken_full'].sum()
        lines.append(f'\nFold {fold}：完整破坏对中，Best间隔<0.5个Risk SD占{near/total:.1%}，≥1个Risk SD占{far/total:.1%}。')
    lines += ['', '这些患者仍可能同时参与被纠正的患者对，不能把丢失分数覆盖率等同于净性能下降贡献。',
              '损失贡献集中与受影响患者广泛可以同时成立；Top15患者覆盖较多错误，并不意味着其他患者没有新增错误。',
              '患者级E_i采用(1−pair credit)累加/可比较对数，tie算0.5错误；无可比较对时为缺失值。',
              '病例错误表及逐对翻转名单保留在private/，仅用于原始随访/切片复核，不纳入公开汇总。', '',
              '## 已有样本量学习曲线', '',
              '每个规模先在同一Fold平均3次抽样，再跨5折计算mean±样本SD；full仅一次/折。',
              '所有组固定180次更新、20次验证检查。Best表现仅描述验证选模结果，Last才是共同预算终点。', '',
              '|训练人数|事件数均值|Best Val C-index|Last Train C-index|Last Val C-index|Last Gap|患者曝光数均值|曝光/训练人数|',
              '|---|---|---|---|---|---|---|---|']
    for row in learning.itertuples():
        n=f'{row.n_train_mean:.1f}' if row.size_label=='full' else f'{row.n_train_mean:.0f}'
        lines.append(f'|{n}|{row.n_events_train_mean:.1f}|{row.best_val_cindex_mean:.4f} ± {row.best_val_cindex_sd:.4f}|'
                     f'{row.last_train_cindex_mean:.4f} ± {row.last_train_cindex_sd:.4f}|'
                     f'{row.last_val_cindex_mean:.4f} ± {row.last_val_cindex_sd:.4f}|{row.last_gap_mean:.4f}|'
                     f'{row.training_patient_exposures_mean:.0f}|{row.patient_exposure_epochs_mean:.1f}|')
    small=learning.loc[learning.size_label.eq('n128')].iloc[0]
    full=learning.loc[learning.size_label.eq('full')].iloc[0]
    lines += ['', f"128人→全量：Last Val均值变化{full.last_val_cindex_mean-small.last_val_cindex_mean:+.4f}，"
              f"Last Gap变化{full.last_gap_mean-small.last_gap_mean:+.4f}；Best Val变化{full.best_val_cindex_mean-small.best_val_cindex_mean:+.4f}。",
              '应观察全部规模，而不是把端点差值解读成严格单调的样本量效应。相同180更新近似固定总患者曝光，',
              '但小训练集每位患者被重复观察更多次，训练事件数也不同；这不是仅改变样本数而训练程度完全相同的因果实验。', '',
              '### 同一Fold内抽样重复的波动', '',
              '下面是先逐折计算3次抽样之间的样本SD，再对5折取均值；它与上表的折间SD不同。', '',
              '|训练规模|每折抽样数|折内Best Val SD的五折均值|折内Last Val SD的五折均值|',
              '|---|---|---|---|']
    for label,part in tables['learning_fold_summary'].groupby('size_label',sort=False):
        best_sd=part.within_repeat_best_val_sd.mean()
        last_sd=part.within_repeat_last_val_sd.mean()
        a=f'{best_sd:.4f}' if np.isfinite(best_sd) else '不适用'
        b=f'{last_sd:.4f}' if np.isfinite(last_sd) else '不适用'
        lines.append(f'|{label}|{int(part.n_repeats.iloc[0])}|{a}|{b}|')
    lines += ['',
              '## 当前可回答和不能回答的问题', '',
              '现有结果能定位风险排序恶化的患者对、事件时间组成和集中程度；它们不能识别ABMIL具体学到了哪种组织学伪相关。',
              '若后期训练纠正部分患者对却破坏更多，说明训练目标的继续优化未转化为这批验证患者的排序收益。',
              '容量、有效监督事件数、WSI未覆盖的临床预后因素均仍是候选解释，不能由这次相关性分解单独判定。',
              '患者对共享患者，未将数千对当独立样本做显著性检验；Top患者在同一数据事后排序，覆盖率不是独立预测能力。',
              '删失仅表示该时点之前未观察到事件；不等于以后永不发生事件。DFS事件定义、Time Zero、失访/死亡处理仍需临床来源确认。',
              '本次不更改标签、不筛除困难患者、不重新划分数据。Stage5尚未启动；若继续，应先核对可复用MeanPool结果，',
              '再以Outer Train内部选参的低容量MeanPool+Ridge-Cox对照，而不是直接与Outer Val选出的Best比较宣称优劣。', '',
              '## 文件与复核', '',
              'fold_pair_summary.csv、pair_strata.csv、risk_margin_breakdown.csv、patient_concentration.csv：完整五折分解。',
              'learning_summary.csv、learning_fold_summary.csv、learning_run_details.csv、learning_update_curves.csv：已有50次训练和1000个评估点。',
              'verified_prediction_metrics.csv：200份Best/Last Train/Val预测的独立复算。validation.json：源文件哈希及未启动训练记录。',
              'late_error_sources、sample_size_learning、sample_size_trajectories各有PNG/PDF；只连接实际更新点，无补造初始点、平滑或填补。', '']
    (root/'report.md').write_text('\n'.join(lines))


def plot_results(root,tables):
    os.environ.setdefault('MPLCONFIGDIR','/tmp/pact_matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':15,'font.weight':'bold','axes.labelweight':'bold',
                         'axes.titleweight':'bold','pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    def save(fig,name):
        for ext in ('png','pdf'):
            fig.savefig(root/f'{name}.{ext}',dpi=180,bbox_inches='tight')
        plt.close(fig)
    val=tables['fold_pair_summary'].loc[lambda d:d.split.eq('val')]
    strata=tables['pair_strata'].loc[lambda d:d.split.eq('val')]
    fig,axes=plt.subplots(2,2,figsize=(17,11),constrained_layout=True)
    ax=axes[0,0]
    ax.bar(val.fold-.18,val.gained_credit,.36,label='Gained credit',color='#719b86')
    ax.bar(val.fold+.18,-val.lost_credit,.36,label='Lost credit',color='#be7b73')
    ax.axhline(0,color='black',lw=1);ax.set(xlabel='Fold',ylabel='Comparable-pair credit',title='Best to last: validation')
    ax.legend(fontsize=11,frameon=False)
    names=['0-12','12-24','24-36','>36']
    time=strata.groupby('event_time_bin')[['gained_credit','lost_credit']].sum().reindex(names)
    ax=axes[0,1];x=np.arange(4)
    ax.bar(x-.18,time.gained_credit,.36,color='#719b86')
    ax.bar(x+.18,-time.lost_credit,.36,color='#be7b73')
    ax.axhline(0,color='black',lw=1);ax.set(xticks=x,xticklabels=names,xlabel='Earlier event time (months)',ylabel='Pair credit',title='Five-fold descriptive totals')
    colors=['#537b99','#bc8d56','#71977c','#9275a7','#a96d6d']
    conc=tables['patient_concentration'].loc[lambda d:d.split.eq('val')]
    ax=axes[1,0]
    for fold,part in conc.groupby('fold'):
        ax.plot(part.patient_fraction*100,part.lost_credit_coverage*100,color=colors[fold],lw=3,marker='o',label=f'Fold {fold}')
        ax.plot(part.patient_fraction*100,part.comparable_pair_coverage*100,color=colors[fold],lw=1.7,linestyle='--')
    ax.set(xlabel='Selected patients (%)',ylabel='Pair coverage (%)',title='Lost credit (solid); pair opportunity (dashed)')
    ax.legend(fontsize=10,frameon=False,ncol=2)
    margin=tables['risk_margin_breakdown'].loc[lambda d:d.split.eq('val')].groupby('margin_bin')[['broken_full','best_correct_pairs']].sum()
    bins=['0-.1','.1-.25','.25-.5','.5-1','1-2','>=2'];margin=margin.reindex(bins)
    ax=axes[1,1]
    ax.bar(np.arange(6),100*margin.broken_full/margin.best_correct_pairs.replace(0,np.nan),color='#8b9db4')
    ax.set(xticks=np.arange(6),xticklabels=bins,xlabel='Best risk gap / risk SD',ylabel='Correct pairs broken (%)',title='Risk-gap-specific damage rate')
    save(fig,'late_error_sources')

    curve=tables['learning_summary'];x=curve.n_train_mean.to_numpy()
    fig,axes=plt.subplots(2,2,figsize=(17,11),constrained_layout=True)
    for key,label,color in [('last_train_cindex','Train: step 180','#65869e'),('last_val_cindex','Val: step 180','#b07e70'),
                            ('best_val_cindex','Val: selected best','#929292')]:
        axes[0,0].errorbar(x,curve[key+'_mean'],yerr=curve[key+'_sd'],lw=3,marker='o',capsize=4,color=color,label=label)
    axes[0,0].set(ylabel='C-index',xlabel='Training patients');axes[0,0].legend(fontsize=11,frameon=False)
    for key,label,color in [('best_gap','Selected best','#929292'),('last_gap','Step 180','#b07e70')]:
        axes[0,1].errorbar(x,curve[key+'_mean'],yerr=curve[key+'_sd'],lw=3,marker='o',capsize=4,color=color,label=label)
    axes[0,1].set(ylabel='Train minus val C-index',xlabel='Training patients');axes[0,1].legend(fontsize=11,frameon=False)
    axes[1,0].plot(x,curve.patient_exposure_epochs_mean,lw=3,marker='o',color='#65869e')
    axes[1,0].set(xlabel='Training patients',ylabel='Patient exposures / training size',title='Same updates; different repetition')
    fold=tables['learning_fold_summary']
    paired=fold.pivot(index='fold',columns='size_label',values='last_val_cindex')
    axes[1,1].bar(paired.index,paired['full']-paired['n128'],color='#8b9db4')
    axes[1,1].axhline(0,color='black',lw=1)
    axes[1,1].set(xlabel='Fold',ylabel='Val C-index: full minus 128',title='Paired folds at step 180')
    save(fig,'sample_size_learning')

    points=tables['learning_curve_points']
    fig,axes=plt.subplots(2,2,figsize=(17,11),constrained_layout=True)
    for ax,metric,title in zip(axes.flat,['train_cindex','val_cindex','train_loss','val_loss'],
                              ['Train C-index','Val C-index','Train Cox loss','Val Cox loss']):
        for label,color in zip(['n128','n256','n384','full'],colors):
            part=points.loc[points.size_label.eq(label)].sort_values('step')
            ax.plot(part.step,part[metric+'_mean'],color=color,lw=3,label=label,marker='o',ms=3)
        ax.set(xlabel='Optimizer step',ylabel=title);ax.grid(alpha=.15)
    axes[0,0].legend(fontsize=11,frameon=False)
    fig.suptitle('Repeats averaged within folds; then mean of five folds',fontsize=18,fontweight='bold')
    save(fig,'sample_size_trajectories')
