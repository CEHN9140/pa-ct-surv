"""Stage5 diagnostic report: describe protocol differences, never select a new winner."""

import os

import numpy as np
import pandas as pd


def report_and_plots(root,metrics,comparisons,strata,reference,inner_cv):
    full=reference.loc[reference.size_label.eq('full')&reference['repeat'].eq(0)].sort_values('fold')
    rows=[]
    for name,table,mapping,protocol in [
        ('MeanPool + Ridge-Cox',metrics,{},'Outer Train内3折选择alpha'),
        ('ABMIL Best',full,{},'Outer Val选checkpoint：仅诊断参照'),
        ('ABMIL Last180',full,{'train_cindex':'last_train_cindex','val_cindex':'last_val_cindex',
                              'auc_36m':'last_auc_36m','auc_60m':'last_auc_60m'},'固定180步：已有后期过拟合')]:
        row=dict(model=name,protocol=protocol)
        for field in ('train_cindex','val_cindex','auc_36m','auc_60m'):
            values=table[mapping.get(field,field)]
            row[field+'_mean']=float(values.mean()) if values.notna().all() else np.nan
            row[field+'_sd']=float(values.std(ddof=1)) if values.notna().all() else np.nan
            row[field+'_n_folds']=int(values.notna().sum())
        gaps=table[mapping.get('train_cindex','train_cindex')]-table[mapping.get('val_cindex','val_cindex')]
        row.update(gap_mean=float(gaps.mean()),gap_sd=float(gaps.std(ddof=1)))
        rows.append(row)
    summary=pd.DataFrame(rows)
    summary.to_csv(root/'comparison_summary.csv',index=False)
    lines=['# Stage5：Raw UNI MeanPool + Ridge-Cox', '',
           '本次仅新增一个低容量模型类型，完成原始五折Outer CV，每折内3折选择Ridge惩罚。',
           '原721例、251事件、TRIDENT UNI1024全WSI；全部patch在float64中均值池化，未做patch L2或重新筛选。',
           '没有调用带MLP的MeanPool、没有额外Attention或深度风险头；模型只有1024个线性Cox系数。',
           '患者特征Z-score按维度使用对应FIT统计量，与patch向量L2归一化不同。', '',
           '## 模型、选参与目标', '',
           '每个Outer Train采用event分层三折，唯一seed=123。alpha固定[0.1,1,10,100,1000,10000]。',
           '每个Inner FIT分别拟合StandardScaler和Cox；均值在1e-12以内相同优先强惩罚。Outer Val仅评估。',
           f'共90次Inner拟合尝试，其中{int(inner_cv.converged.sum())}次成功、{int((~inner_cv.converged).sum())}次数值失败；另有5次完整Outer Train成功拟合，只保存5个最终模型。',
           'CoxPHSurvivalAnalysis显式ties=breslow、n_iter=1000、tol=1e-9；任何警告使该次拟合失败，不能静默打分。',
           '数值协议修订：第一次运行在Fold0的FIT子集、alpha=0.1时出现exp溢出，并在查看任何Outer Val结果前停止。',
           '随后保持同一候选网格、划分和求解器，记录所有失败；某alpha只要有一个Inner失败就整体不可选，不能平均剩余折或填补分数。',
           '每个候选仍尝试全部3折；若无完整候选或最终Outer拟合失败则整轮中止。此次修订没有依据Outer Val调整。',
           '本地软件目标为[-sum_event(log partial likelihood)+alpha/2*||beta||²]/n_fit；',
           '与主流程相同事件方向和Breslow规则，但后者按事件平均、mini-batch风险集与Adam，',
           '因此本实验同时改变聚合方式、非线性容量、正则化与优化，并非Attention的单独因果消融。',
           '时间使用历史深度模型的float32表示后转回float64计算，确保相同可比较患者对；风险输出不经Sigmoid。', '',
           '## 五折结果及历史参照', '',
           '|模型|Train C-index|Val C-index|Gap|36月AUC|60月AUC|',
           '|---|---|---|---|---|---|']
    for row in rows:
        values=[]
        for field in ('train_cindex','val_cindex','gap','auc_36m','auc_60m'):
            mean,sd=row[field+'_mean'],row[field+'_sd']
            values.append(f'{mean:.4f} ± {sd:.4f}' if np.isfinite(mean) else '不可完整估计（见状态列）')
        lines.append('|'+row['model']+'|'+'|'.join(values)+'|')
    lines+=['', '以上是五折mean±样本SD，不是独立测试集或置信区间。AUC用Outer Train估计删失分布的累计/动态IPCW。',
            '不能估计AUC时保留患者并记录原因；不通过删样或替换数值保证完整结果。',
            'ABMIL Best使用Outer Val选轮，可能乐观；Last180有固定训练预算下的过拟合。两者均不与新Ridge构成严格公平优劣检验。', '',
            '## Ridge逐折结果', '',
            '|Fold|alpha|Train C-index|Val C-index|Gap|36月AUC|60月AUC|',
            '|---|---|---|---|---|---|---|']
    for row in metrics.itertuples():
        lines.append(f'|{row.fold}|{row.alpha:g}|{row.train_cindex:.4f}|{row.val_cindex:.4f}|{row.gap:.4f}|{row.auc_36m:.4f}|{row.auc_60m:.4f}|')
    ridge_mean=float(metrics.val_cindex.mean())
    lines+=['',f"Ridge五折Val均值相对历史ABMIL Best差值为{ridge_mean-full.val_cindex.mean():+.4f}，"
            f"相对Last180为{ridge_mean-full.last_val_cindex.mean():+.4f}；"
            f"Ridge平均Gap为{metrics.gap.mean():.4f}。这些仅是不同协议下的描述性差值。",
            f"Fold0 Val={metrics.loc[metrics.fold.eq(0),'val_cindex'].iloc[0]:.4f}，"
            f"Fold4 Val={metrics.loc[metrics.fold.eq(4),'val_cindex'].iloc[0]:.4f}；"
            '需结合其他三折判断是否延续共同困难，不能据此调整划分。']
    if metrics.alpha.eq(10000).any():
        lines+=['',f'有{int(metrics.alpha.eq(10000).sum())}/5折选择预设网格上界10000。这不证明最优惩罚已被网格完全包住；本轮不事后扩展网格。']
    invalid=inner_cv.loc[~inner_cv.candidate_eligible,['fold','alpha']].drop_duplicates()
    if len(invalid):
        lines+=['', '不可选的数值失败候选（不是低分候选）：'+', '.join(
            f'Fold{r.fold}/alpha={r.alpha:g}' for r in invalid.itertuples())+'。完整异常类/信息和成功的其他折仍保留在inner_cv.csv。']
    lines+=['', '## 与ABMIL患者级风险和患者对错误的一致性', '',
            '下表仅Outer Val，各模型在完全相同患者及标签上比较。Jaccard=双方均完全错误的对数/至少一方完全错误的对数。',
            '风险ties仍以0.5分计算C-index，但不计入“完全错误”；共享tie另列。ΔC为Ridge−相应历史ABMIL。', '',
            '|Fold|参照|Risk Spearman|错误Jaccard|共享完全错误对|Ridge纠正参照/破坏参照|ΔC|',
            '|---|---|---|---|---|---|---|']
    for row in comparisons.loc[comparisons.split.eq('val')].itertuples():
        lines.append(f'|{row.fold}|{row.reference}|{row.spearman:.4f}|{row.wrong_jaccard:.4f}|{row.both_wrong}|'
                     f'{row.reference_wrong_ridge_correct}/{row.reference_correct_ridge_wrong}|{row.delta_cindex:+.4f}|')
    lines+=['', '共享错误同时按事件时间及Event–Event/Event–Later Censor/同时间Event–Censor分组，见error_overlap_strata.csv。',
            '患者级错误表在private/，包含共享/各自独有的完全错误对数和tie对数；患者对共享患者，不将对数当独立样本做显著性检验。', '',
            '## 本次诊断能说明什么', '',
            '先判断Ridge的排序能力和Gap，再结合其与ABMIL共享/独有的错误，不能只按一次均值选模型。',
            '若Ridge接近历史ABMIL且Gap更小，支持低容量聚合值得进一步研究；仍需匹配选参协议确认。',
            '若Ridge明显较弱，说明“平均特征+线性风险”不足以复现ABMIL的全部预测信号；不能据此单独证明Attention有效。',
            '若Ridge仍有明显Gap，说明高维患者特征在有限结局监督下也能过拟合，不能只归因于Epoch或Adam。',
            '1024维仍是高维；本轮正则化网格及内层选择并不保证不存在过拟合。',
            '两类模型共享UNI编码器和队列：相关风险/共享错误不能证明WSI缺少DFS信息，也不能判定DFS标签错误。',
            '尚未重训采用严格内层选轮的ABMIL；下一步是否补齐该对照应在查看本诊断后决定。', '',
            '## 保存与核验', '',
            'metrics.csv：五折指标及AUC状态；inner_cv.csv：90次候选验证；comparison_summary.csv：明确协议的mean±SD。',
            'prediction_comparison.csv、error_overlap_strata.csv：风险相关和患者对错误分解；validation.json：来源/代码哈希及拟合数。',
            'private/保存独立患者特征缓存、五个Scaler+Cox模型、病例级Train/Val及721例OOF风险、Inner角色和错误表，均不发布。',
            '不直接把不同Fold的OOF风险混合计算总体C-index，因为各Fold的风险尺度/基准并不相同。',
            '所有源特征/标签/历史结果只读；各最终模型重载后的预测逐位相同。无新ABMIL或教师/学生训练。', '']
    (root/'report.md').write_text('\n'.join(lines))
    plot_results(root,metrics,comparisons,full)


def plot_results(root,metrics,comparisons,full):
    os.environ.setdefault('MPLCONFIGDIR','/tmp/pact_matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':15,'font.weight':'bold','axes.labelweight':'bold',
                         'axes.titleweight':'bold','pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    metrics=metrics.sort_values('fold')
    fig,axes=plt.subplots(2,2,figsize=(17,11),constrained_layout=True)
    for ax,key,title in zip(axes.flat,['train_cindex','val_cindex','auc_36m','auc_60m'],
                            ['Train C-index','Val C-index','36-month AUC','60-month AUC']):
        ax.plot(metrics.fold,metrics[key],color='#527b93',lw=3,marker='o',label='MeanPool + Ridge-Cox')
        ax.plot(full.fold,full[key],color='#8d8d8d',lw=2.5,marker='s',ls='--',label='ABMIL: outer-val selected')
        ax.plot(full.fold,full['last_'+key],color='#b68c73',lw=2.5,marker='^',ls=':',label='ABMIL: fixed step 180')
        ax.set(xlabel='Fold',ylabel=title,xticks=range(5));ax.grid(alpha=.15)
    axes[0,0].legend(fontsize=11,frameon=False)
    fig.suptitle('Diagnostic references with different selection protocols',fontsize=19,fontweight='bold')
    for ext in ('png','pdf'): fig.savefig(root/f'fold_comparison.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(17,5.5),constrained_layout=True)
    for phase,color in [('best','#6d8e9b'),('last','#b68c73')]:
        data=comparisons.loc[comparisons.split.eq('val')&comparisons.reference.eq(phase)].sort_values('fold')
        axes[0].plot(data.fold,data.spearman,color=color,lw=3,marker='o',label=f'ABMIL {phase}')
        axes[1].plot(data.fold,data.wrong_jaccard,color=color,lw=3,marker='o',label=f'ABMIL {phase}')
    axes[0].set(xlabel='Fold',ylabel='Risk Spearman')
    axes[1].set(xlabel='Fold',ylabel='Wrong-pair Jaccard')
    for ax in axes: ax.set_xticks(range(5));ax.grid(alpha=.15)
    axes[0].legend(frameon=False,fontsize=12)
    for ext in ('png','pdf'): fig.savefig(root/f'risk_error_overlap.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)
