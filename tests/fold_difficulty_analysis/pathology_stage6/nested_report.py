"""Compare two nested procedures on the same outer patients, without pooled-risk CI."""

import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sksurv.metrics import concordance_index_censored

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'15_pathology_query_ct_tokens'))
from risk_calibration import paired_comparison


def paired_interval(frames,seed,repeats=2000):
    return paired_comparison(frames,repeats=repeats,seed=seed,
                             reference_columns={'ridge':'ridge_risk'})['fusion_minus_ridge']


def summarize(root,ridge_root,samples,seed,track):
    root=Path(root);ridge_root=Path(ridge_root)
    metrics=pd.read_csv(root/'metrics.csv').sort_values('fold')
    ridge=pd.read_csv(track(ridge_root/'metrics.csv')).sort_values('fold')
    if metrics.fold.tolist()!=list(range(5)) or ridge.fold.tolist()!=list(range(5)):
        raise ValueError('Both nested procedures need all five folds')
    frames,paired,curves,selections,audits=[],[],[],[],[]
    for fold in range(5):
        a=pd.read_csv(root/f'fold_{fold}/private/outer/val_predictions.csv',dtype={'case_id':str})
        b=pd.read_csv(track(ridge_root/f'private/fold_{fold}/val_predictions.csv'),dtype={'case_id':str})
        patients=samples.loc[samples.split.eq(fold)]
        for part in (a,b):
            if part.case_id.duplicated().any() or set(part.case_id)!=set(patients.pa_id):
                raise ValueError('Nested comparison has different validation patients')
        a=a.set_index('case_id').loc[patients.pa_id].reset_index()
        b=b.set_index('case_id').loc[patients.pa_id].reset_index()
        # NN exports float32; restore it after CSV decimal parsing, unlike float64 Ridge.
        a['risk_score']=a.risk_score.astype(np.float32)
        for part in (a,b):
            if not np.array_equal(part['dfs.month'].to_numpy(np.float32),patients.time.to_numpy(np.float32)) \
                    or not np.array_equal(part['dfs.status'].to_numpy(),patients.event.to_numpy()):
                raise ValueError('Nested comparison labels differ')
        a['ridge_risk']=b.risk_score.to_numpy()
        a['fold']=fold
        frames.append(a)
        x=metrics.loc[metrics.fold.eq(fold)].iloc[0]
        y=ridge.loc[ridge.fold.eq(fold)].iloc[0]
        for pred,expected in [(a,x.val_cindex),(b,y.val_cindex)]:
            ci=concordance_index_censored(pred['dfs.status'].astype(bool),pred['dfs.month'],pred.risk_score)[0]
            if not np.isclose(ci,expected,atol=1e-12,rtol=0): raise ValueError('Saved predictions do not reproduce fold C-index')
        paired.append(dict(fold=fold,selected_step=int(x.selected_step),ridge_alpha=float(y.alpha),
                           abmil_train_cindex=x.train_cindex,ridge_train_cindex=y.train_cindex,
                           abmil_val_cindex=x.val_cindex,ridge_val_cindex=y.val_cindex,
                           delta_cindex=x.val_cindex-y.val_cindex,
                           risk_spearman=float(spearmanr(a.risk_score,b.risk_score).statistic)))
        for name,collection in [('inner_curves',curves),('step_selection',selections),('fit_audit',audits)]:
            collection.append(pd.read_csv(root/f'fold_{fold}/{name}.csv').assign(fold=fold))
    (root/'private').mkdir(exist_ok=True)
    pd.concat(frames,ignore_index=True).to_csv(root/'private/oof_comparison.csv',index=False)
    paired=pd.DataFrame(paired);paired.to_csv(root/'fold_comparison.csv',index=False)
    interval=paired_interval(frames,seed)
    if not np.isclose(interval['delta_cindex'],paired.delta_cindex.mean(),atol=1e-12,rtol=0):
        raise RuntimeError('Bootstrap observed difference does not reproduce mean fold C-index')
    interval['method']='2000 paired-patient resamples within each fixed outer fold; mean of five fold deltas. Conditional on fitted models; excludes retraining/selection/prior exploration uncertainty.'
    interval['positive_folds']=int(paired.delta_cindex.gt(0).sum())
    (root/'paired_comparison.json').write_text(json.dumps(interval,indent=2))
    curves=pd.concat(curves,ignore_index=True)
    selection=pd.concat(selections,ignore_index=True)
    audit=pd.concat(audits,ignore_index=True)
    if len(audit)!=20 or not audit.groupby('fold').initial_state_sha256.nunique().eq(1).all():
        raise RuntimeError('Twenty fresh fits with identical seed initialization were not verified')
    if not audit.loc[audit.role.eq('outer'),'evaluation_calls'].eq(0).all():
        raise RuntimeError('Outer model was evaluated during fixed refit')
    curves.to_csv(root/'inner_curves.csv',index=False)
    selection.to_csv(root/'step_selection.csv',index=False)
    audit.to_csv(root/'fit_audit.csv',index=False)
    summary=[]
    for name,table in [('ABMIL Nested',metrics),('MeanPool Ridge Nested',ridge)]:
        row=dict(model=name)
        for key in ('train_cindex','val_cindex','gap','auc_36m','auc_60m'):
            values=table[key]
            row[key+'_mean']=float(values.mean()) if values.notna().all() else np.nan
            row[key+'_sd']=float(values.std(ddof=1)) if values.notna().all() else np.nan
            row[key+'_valid_folds']=int(values.notna().sum())
        summary.append(row)
    pd.DataFrame(summary).to_csv(root/'comparison_summary.csv',index=False)
    lines=['# Stage6：严格Nested ABMIL与Nested Ridge-Cox', '',
           '原721例/251事件、同一五折、Raw UNI1024全WSI；三折Inner患者角色与Stage5逐位相同。',
           'ABMIL结构、Dropout0.25、Cox64合尾、Adam lr1e-4/WD5e-4/betas(0.9,0.999)/eps1e-8不变。',
           '每次网络从seed123独立初始化，未载入任何项目训练模型作初始化。',
           '内层候选Step预先固定[9,18,36,54,72,90,117,144,180]，所有15个Inner训练到180步。',
           '按每个相同Step的三折HELD平均C-index选一步，1e-12内平分取较早者。',
           'Outer Train从头训练恰好选定的Step，不换算Epoch、不早停、没有调度器；结束后仅评估Outer Val一次。',
           '完整原始patch保留，使用已验证的两遍联合Cox VJP；批次与Dropout随机流仅从唯一seed派生。', '',
           '## 同外层患者、内层选参的五折结果', '',
           '|模型|Train C-index|Val C-index|Gap|36月AUC|60月AUC|',
           '|---|---|---|---|---|---|']
    for row in summary:
        text=[]
        for key in ('train_cindex','val_cindex','gap','auc_36m','auc_60m'):
            text.append(f'{row[key+"_mean"]:.4f} ± {row[key+"_sd"]:.4f}')
        lines.append('|'+row['model']+'|'+'|'.join(text)+'|')
    lines+=['', '均值±样本SD跨5折。AUC采用对应Outer Train估计删失分布，失败原因保留在metrics.csv，不删除病例或填补。', '',
            '|Fold|选定Step|ABMIL Val|Ridge Val|ΔC|',
            '|---|---|---|---|---|']
    for row in paired.itertuples():
        lines.append(f'|{row.fold}|{row.selected_step}|{row.abmil_val_cindex:.4f}|{row.ridge_val_cindex:.4f}|{row.delta_cindex:+.4f}|')
    lines+=['', f"平均ΔC(ABMIL−Ridge)={interval['delta_cindex']:+.4f}，配对患者Bootstrap名义95%区间"
            f"[{interval['lower']:+.4f},{interval['upper']:+.4f}]，{interval['positive_folds']}/5折为正。", '',
            '## 如何判断', '',
            '本轮比较消除了用Outer Val选择ABMIL训练轮次的直接乐观偏差；历史0.7452不作为本次公平比较结果。',
            '若出现跨折一致且区间支持的正差值，支持当前可训练ABMIL整体流程有额外预测收益；',
            '但它同时包含Patch投影、Attention和非线性头，不能将收益单独归因于Attention。',
            '若差值区间跨0，不能据此证明两者等效；只能说当前样本未确认增益。',
            'Ridge五折均选原网格上界10000，比较仅针对Stage5已完成的方案，不代表所有线性模型的最优性能。',
            '两类模型均在内部Outer Train选参，但模型类别、惩罚、优化与风险集大小仍不同。', '',
            '## 不确定性和边界', '',
            'Bootstrap仅在每个Outer Val内配对重采样患者，然后取五折ΔC均值；不跨Fold混合未校准风险。',
            '区间条件于已训练模型，不包含重训练、内层选择、折间训练重叠及此前同队列多次探索的全部不确定性。',
            '严格Nested CV仍是内部验证，不是独立外部测试。没有根据Outer Val调整候选范围、学习率或模型。',
            '本轮结束后不自动新增结构或Stage7诊断。正式病理–CT教师和CT-only学生方法未被改变。', '',
            '## 文件', '',
            'inner_curves.csv/step_selection.csv记录全部135个Inner固定点评估和45个候选均值；',
            'fit_audit.csv记录20次新初始化、预算与评估调用；metrics.csv记录外层唯一评估。',
            'comparison_summary.csv、fold_comparison.csv、paired_comparison.json及图记录嵌套比较。',
            '模型、内层HELD预测、最终Train/Val及OOF病例预测均本地保留，不发布。', '']
    (root/'report.md').write_text('\n'.join(lines))
    plot(root,curves,paired)


def plot(root,curves,paired):
    os.environ.setdefault('MPLCONFIGDIR','/tmp/pact_matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':15,'font.weight':'bold','axes.labelweight':'bold','axes.titleweight':'bold',
                         'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(17,6),constrained_layout=True)
    axes[0].plot(paired.fold,paired.abmil_val_cindex,lw=3,marker='o',color='#527b93',label='ABMIL nested')
    axes[0].plot(paired.fold,paired.ridge_val_cindex,lw=3,marker='s',color='#b68c73',label='Ridge-Cox nested')
    axes[0].set(xlabel='Outer fold',ylabel='Validation C-index',xticks=range(5));axes[0].legend(frameon=False)
    axes[1].bar(paired.fold,paired.delta_cindex,color='#819bad');axes[1].axhline(0,color='black',lw=1)
    axes[1].set(xlabel='Outer fold',ylabel='C-index: ABMIL minus Ridge',xticks=range(5))
    for ext in ('png','pdf'): fig.savefig(root/f'nested_comparison.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(19,11),constrained_layout=True)
    for fold,ax in zip(range(5),axes.flat):
        data=curves.loc[curves.fold.eq(fold)]
        for inner in range(3):
            part=data.loc[data.inner_fold.eq(inner)].sort_values('step')
            ax.plot(part.step,part.cindex,lw=2.5,marker='o',ms=4,label=f'Inner {inner}')
        mean=data.groupby('step').cindex.mean()
        ax.plot(mean.index,mean.values,lw=3.5,color='black',label='Mean')
        ax.axvline(int(paired.loc[paired.fold.eq(fold),'selected_step'].iloc[0]),color='#9f7f65',ls='--',lw=2)
        ax.set(title=f'Outer fold {fold}',xlabel='Optimizer step',ylabel='Inner HELD C-index');ax.grid(alpha=.15)
    axes[0,0].legend(fontsize=10,frameon=False,ncol=2);axes.flat[-1].axis('off')
    for ext in ('png','pdf'): fig.savefig(root/f'inner_step_selection.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)
