"""Capacity ablation on matched raw UNI nested procedures; never compare to EAGLE."""

import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sksurv.metrics import concordance_index_censored

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'tests/15_pathology_query_ct_tokens'))
from risk_calibration import paired_comparison


def summarize(root,reference,samples,seed,track):
    root,reference=Path(root),Path(reference)
    small=pd.read_csv(root/'metrics.csv').sort_values('fold')
    full=pd.read_csv(track(reference/'metrics.csv')).sort_values('fold')
    if small.fold.tolist()!=list(range(5)) or full.fold.tolist()!=list(range(5)):
        raise ValueError('Matched capacity comparison requires complete five folds')
    curves,selections,audits,frames,contrasts=[],[],[],[],[]
    for fold in range(5):
        patients=samples.loc[samples.split.eq(fold)]
        a=pd.read_csv(root/f'fold_{fold}/private/outer/val_predictions.csv',dtype={'case_id':str})
        b=pd.read_csv(track(reference/f'fold_{fold}/private/outer/val_predictions.csv'),dtype={'case_id':str})
        for frame in (a,b):
            if frame.case_id.duplicated().any() or set(frame.case_id)!=set(patients.pa_id): raise ValueError('Comparison patients differ')
        a=a.set_index('case_id').loc[patients.pa_id].reset_index()
        b=b.set_index('case_id').loc[patients.pa_id].reset_index()
        for frame,metrics in [(a,small),(b,full)]:
            if not np.array_equal(frame['dfs.month'].to_numpy(np.float32),patients.time.to_numpy(np.float32)) \
                    or not np.array_equal(frame['dfs.status'].to_numpy(),patients.event.to_numpy()): raise ValueError('Comparison labels differ')
            frame['risk_score']=frame.risk_score.astype(np.float32)
            ci=concordance_index_censored(frame['dfs.status'].astype(bool),frame['dfs.month'],frame.risk_score)[0]
            if not np.isclose(ci,metrics.loc[metrics.fold.eq(fold),'val_cindex'].iloc[0],atol=1e-12,rtol=0): raise ValueError('Prediction metrics mismatch')
        a['full_abmil_risk']=b.risk_score.to_numpy();a['fold']=fold;frames.append(a)
        x=small.loc[small.fold.eq(fold)].iloc[0];y=full.loc[full.fold.eq(fold)].iloc[0]
        contrasts.append(dict(fold=fold,light_selected_step=int(x.selected_step),full_selected_step=int(y.selected_step),
                              light_val_cindex=x.val_cindex,full_val_cindex=y.val_cindex,delta_cindex=x.val_cindex-y.val_cindex,
                              light_gap=x.gap,full_gap=y.gap,risk_spearman=float(spearmanr(a.risk_score,b.risk_score).statistic)))
        for name,parts in [('inner_curves',curves),('step_selection',selections),('fit_audit',audits)]:
            parts.append(pd.read_csv(root/f'fold_{fold}/{name}.csv').assign(fold=fold))
    contrasts=pd.DataFrame(contrasts)
    contrasts.to_csv(root/'fold_comparison.csv',index=False)
    interval=paired_comparison(frames,seed=seed,repeats=2000,reference_columns={'full_abmil':'full_abmil_risk'})['fusion_minus_full_abmil']
    if not np.isclose(interval['delta_cindex'],contrasts.delta_cindex.mean(),atol=1e-12,rtol=0): raise RuntimeError('Paired difference mismatch')
    interval.update(positive_folds=int(contrasts.delta_cindex.gt(0).sum()),
                    method='2000 paired patient resamples within each outer fold; mean fold delta Light minus Full. Conditional on fitted models; excludes retraining, tuning and earlier exploration uncertainty.')
    (root/'paired_comparison.json').write_text(json.dumps(interval,indent=2))
    audit=pd.concat(audits,ignore_index=True)
    if len(audit)!=20 or not audit.groupby('fold').initial_state_sha256.nunique().eq(1).all() \
            or not audit.loc[audit.role.eq('outer'),'evaluation_calls'].eq(0).all(): raise RuntimeError('Light training isolation audit failed')
    for name,table in [('fit_audit',audit),('inner_curves',pd.concat(curves,ignore_index=True)),('step_selection',pd.concat(selections,ignore_index=True))]:
        table.to_csv(root/f'{name}.csv',index=False)
    (root/'private').mkdir(exist_ok=True)
    pd.concat(frames,ignore_index=True).to_csv(root/'private/oof_comparison.csv',index=False)
    summary=[]
    for name,table in [('Light ABMIL',small),('Full ABMIL Stage6',full)]:
        row=dict(model=name)
        for field in ('train_cindex','val_cindex','gap','auc_36m','auc_60m'):
            values=table[field]
            row[field+'_mean']=float(values.mean()) if values.notna().all() else np.nan
            row[field+'_sd']=float(values.std(ddof=1)) if values.notna().all() else np.nan
        summary.append(row)
    pd.DataFrame(summary).to_csv(root/'comparison_summary.csv',index=False)
    cfg=json.loads((root/'run_config.json').read_text())
    lines=['# 实验16：Raw UNI轻量ABMIL容量对照', '',
           '独立子实验，不改变现有Radiomics/教师、主流程ABMIL或v7日志。仅新增轻量ABMIL训练。',
           f"原始UNI1024维完整WSI，无L2、无EAGLE Top25。轻量投影1024→128，Attention128→{cfg['attention_dim']}→1，"
           '加权池化128维，后投影128→64，风险头64→1。',
           f"参数量{cfg['full_parameters']:,}→{cfg['light_parameters']:,}，减少{100*cfg['reduction_fraction']:.3f}%。",
           '仍调用原model/abmil_cox.py的ABMIL，仅构造维度不同；输出未经Sigmoid的Cox风险。', '',
           '## 控制变量及协议', '',
           f"沿用Stage6的原721例/251事件/五折，以及完全相同的内层三折角色、患者批次顺序和seed{seed}。",
           'Dropout0.25、Cox64尾部合并、Adam lr1e-4/WD5e-4/betas(0.9,0.999)/eps1e-8不变。',
           '固定候选Step[9,18,36,54,72,90,117,144,180]，按三折平均选步，平分取早步；外层从头训练相应步数。',
           'Outer Val只在最终固定权重后评估一次，无外层选轮。轻量模型没有加载完整ABMIL的训练权重。',
           '原v7 EAGLE日志不作为容量严格对照；参照是Stage6完成的Full Raw UNI ABMIL，未重训它。',
           '不同宽度改变Dropout张量维度：相同seed与批次，但不声称逐神经元遮罩完全相同。', '',
           '## 五折mean±样本SD', '',
           '|模型|Train C-index|Val C-index|Gap|36月AUC|60月AUC|',
           '|---|---|---|---|---|---|']
    for row in summary:
        values=[f'{row[k+"_mean"]:.4f} ± {row[k+"_sd"]:.4f}' for k in ('train_cindex','val_cindex','gap','auc_36m','auc_60m')]
        lines.append('|'+row['model']+'|'+'|'.join(values)+'|')
    lines+=['', '|Fold|Light Step|Full Step|Light Val|Full Val|ΔC|', '|---|---|---|---|---|---|']
    for r in contrasts.itertuples():
        lines.append(f'|{r.fold}|{r.light_selected_step}|{r.full_selected_step}|{r.light_val_cindex:.4f}|{r.full_val_cindex:.4f}|{r.delta_cindex:+.4f}|')
    lines+=['',f"ΔC(Light−Full)={interval['delta_cindex']:+.4f}，配对患者Bootstrap名义95%区间"
            f"[{interval['lower']:+.4f},{interval['upper']:+.4f}]，{interval['positive_folds']}/5折为正。", '',
            '## 解释边界', '',
            '缩小Gap本身不能证明泛化改善，应同时查看Val C-index、36/60月AUC、逐折变化及配对区间。',
            '当前对照检验宽度/容量修改的整体预测效果，不证明抽象参数数量本身是唯一原因。',
            '区间条件于当前训练完成的模型，不包含重训练/内层选步或此前同队列探索的不确定性；仍是内部验证。',
            '不把跨Fold的原始OOF风险混合算单个C-index；只报告折内指标。',
            'private/和fold_*/private/保留模型与病例；公开仅聚合结果与代码。本次不自动commit/push。', '']
    (root/'report.md').write_text('\n'.join(lines))
    plot(root,small,full,contrasts)


def plot(root,small,full,contrasts):
    os.environ.setdefault('MPLCONFIGDIR','/tmp/pact_matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':15,'font.weight':'bold','axes.labelweight':'bold','axes.titleweight':'bold',
                         'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(17,11),constrained_layout=True)
    for ax,key,title in zip(axes.flat,['val_cindex','gap','auc_36m','auc_60m'],['Val C-index','Train minus Val gap','36-month AUC','60-month AUC']):
        ax.plot(small.fold,small[key],lw=3,marker='o',color='#527b93',label='Light ABMIL')
        ax.plot(full.fold,full[key],lw=3,marker='s',color='#b68c73',label='Full ABMIL')
        ax.set(xlabel='Outer fold',ylabel=title,xticks=range(5));ax.grid(alpha=.15)
    axes[0,0].legend(frameon=False,fontsize=12)
    for ext in ('png','pdf'): fig.savefig(root/f'capacity_comparison.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)
