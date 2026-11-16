"""Primary Full-WSI comparison plus prespecified random-evaluation sensitivity."""

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sksurv.metrics import concordance_index_censored

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'tests/15_pathology_query_ct_tokens'))
from risk_calibration import paired_comparison


def summarize(root,reference,samples,seed,track):
    root,reference=Path(root),Path(reference)
    config=json.loads((root/'run_config.json').read_text());repeats=config['eval_random_repeats']
    current=pd.read_csv(root/'metrics.csv').sort_values('fold')
    old=pd.read_csv(track(reference/'metrics.csv')).sort_values('fold')
    if current.fold.tolist()!=list(range(5)) or old.fold.tolist()!=list(range(5)):raise ValueError('Five complete outer folds required')
    frames,random_tables,audits,curves,selections,contrasts=[],[],[],[],[],[]
    for fold in range(5):
        patients=samples.loc[samples.split.eq(fold)]
        a=pd.read_csv(root/f'fold_{fold}/private/outer/val_predictions.csv',dtype={'case_id':str})
        b=pd.read_csv(track(reference/f'fold_{fold}/private/outer/val_predictions.csv'),dtype={'case_id':str})
        for frame,table in [(a,current),(b,old)]:
            if frame.case_id.duplicated().any() or set(frame.case_id)!=set(patients.pa_id):raise ValueError('Comparison cohort differs')
            frame.set_index('case_id',inplace=True);frame=frame.loc[patients.pa_id]
            if not np.array_equal(frame['dfs.month'].to_numpy(np.float32),patients.time.to_numpy(np.float32)) \
                    or not np.array_equal(frame['dfs.status'].to_numpy(),patients.event.to_numpy()):raise ValueError('Comparison labels differ')
            ci=concordance_index_censored(frame['dfs.status'].astype(bool),frame['dfs.month'],frame.risk_score.to_numpy(np.float32))[0]
            if not np.isclose(ci,table.loc[table.fold.eq(fold),'val_cindex'].item(),atol=1e-12,rtol=0):raise ValueError('Prediction metrics differ')
        a=a.loc[patients.pa_id].reset_index();b=b.loc[patients.pa_id]
        a['risk_score']=a.risk_score.astype(np.float32);a['reference_risk']=b.risk_score.to_numpy(np.float32)
        a['fold']=fold;frames.append(a)
        random=pd.read_csv(root/f'fold_{fold}/random_metrics.csv')
        if len(random)!=repeats or set(random.repeat)!=set(range(repeats)) or not random.fold.eq(fold).all():raise ValueError('Random evaluation grid incomplete')
        random_tables.append(random)
        fit=pd.read_csv(root/f'fold_{fold}/fit_audit.csv').assign(fold=fold);audits.append(fit)
        baseline_fit=pd.read_csv(track(reference/f'fold_{fold}/fit_audit.csv'))
        if len(fit)!=4 or fit.initial_state_sha256.nunique()!=1 or baseline_fit.initial_state_sha256.nunique()!=1 \
                or fit.initial_state_sha256.iloc[0]!=baseline_fit.initial_state_sha256.iloc[0] \
                or not fit.loc[fit.role.eq('outer'),'evaluation_calls'].eq(0).all():raise ValueError('Matched initialization or outer isolation audit failed')
        curves.append(pd.read_csv(root/f'fold_{fold}/inner_curves.csv').assign(fold=fold))
        selections.append(pd.read_csv(root/f'fold_{fold}/step_selection.csv').assign(fold=fold))
        x=current.loc[current.fold.eq(fold)].iloc[0];y=old.loc[old.fold.eq(fold)].iloc[0]
        contrasts.append(dict(fold=fold,random_train_selected_step=int(x.selected_step),full_train_selected_step=int(y.selected_step),
                              random_train_val_cindex=x.val_cindex,full_train_val_cindex=y.val_cindex,delta_cindex=x.val_cindex-y.val_cindex,
                              random_train_gap=x.gap,full_train_gap=y.gap))
    random=pd.concat(random_tables,ignore_index=True);random.to_csv(root/'random_metrics.csv',index=False)
    for name,parts in [('fit_audit',audits),('inner_curves',curves),('step_selection',selections)]:pd.concat(parts,ignore_index=True).to_csv(root/f'{name}.csv',index=False)
    differences=pd.DataFrame(contrasts);differences.to_csv(root/'fold_comparison.csv',index=False)
    interval=paired_comparison(frames,seed=seed,repeats=2000,reference_columns={'full_training':'reference_risk'})['fusion_minus_full_training']
    if not np.isclose(interval['delta_cindex'],differences.delta_cindex.mean(),atol=1e-12,rtol=0):raise RuntimeError('Primary paired difference mismatch')
    interval.update(positive_folds=int(differences.delta_cindex.gt(0).sum()),
                    estimand='Random128-training minus Full-training; both evaluated on Full WSI',
                    method='2000 paired patient resamples within outer fold; mean fold delta; conditional on fitted models, excludes retraining/tuning/earlier exploration')
    (root/'paired_comparison.json').write_text(json.dumps(interval,indent=2))
    (root/'private').mkdir(exist_ok=True);pd.concat(frames,ignore_index=True).to_csv(root/'private/primary_oof_comparison.csv',index=False)
    fields=['train_cindex','val_cindex','gap','auc_36m','auc_60m']
    random_foldmeans=random.groupby('fold')[fields].agg(lambda values:values.mean(skipna=False)).reset_index()
    summary=[]
    policies=[('Full training / Full eval',old),('Random128 training / Full eval',current),
              ('Random128 training / Random128 eval (fixed repeat0)',random.loc[random.repeat.eq(0)]),
              ('Random128 training / Random128 eval (repeat mean)',random_foldmeans)]
    for policy,table in policies:
        row=dict(policy=policy)
        for field in fields:
            # Preserve undefined folds; do not silently average fewer than five folds.
            available=table[field].notna().all()
            row[field+'_mean']=float(table[field].mean()) if available else np.nan
            row[field+'_sd']=float(table[field].std(ddof=1)) if available else np.nan
        summary.append(row)
    pd.DataFrame(summary).to_csv(root/'comparison_summary.csv',index=False)
    repeat_means=random.groupby('repeat')[fields].mean()
    mc=pd.DataFrame([dict(metric=field,random_repeat_sd=float(repeat_means[field].std(ddof=1)) if random[field].notna().all() else np.nan) for field in fields])
    mc.to_csv(root/'random_repeat_variability.csv',index=False)
    lines=['# Random128训练：Raw UNI + ABMIL严格Nested CV','',
           '原721例/251事件/外层五折；同Stage6三折内层角色、患者Cox组及优化预算。',
           f'seed={seed}；ABMIL参数722178、Dropout0.25、Adam lr1e-4/WD5e-4、Cox64合尾不变。',
           '仅训练时每次无放回随机128个patch，N≤128用全部；不同更新重新采样。',
           'Cox两遍VJP重放同一次采样和Dropout，不是两遍分别随机；eval关闭Dropout。',
           '从头初始化20次，不加载项目训练模型；参数初始化hash与Stage6完全相同。',
           '内层仍用Full WSI在固定候选Step上选轮，Outer Val没有参与选轮。',
           '网络宽度/参数不变，但采样消耗随机数且Dropout张量变小，不能声称逐神经元遮罩与Full相同。','',
           '## 预先固定的最终评估（mean±五折样本SD）','',
           '|训练 / 评估|Train C-index|Val C-index|Gap|36月AUC|60月AUC|','|---|---|---|---|---|---|']
    for row in summary:
        values=[f'{row[field+"_mean"]:.4f} ± {row[field+"_sd"]:.4f}' for field in fields]
        lines.append('|'+row['policy']+'|'+'|'.join(values)+'|')
    lines+=['','主比较预先固定为：Random128训练→Full评估 vs Stage6 Full训练→Full评估。',
            f"主比较ΔC={interval['delta_cindex']:+.4f}，条件配对Bootstrap名义95%区间"
            f"[{interval['lower']:+.4f},{interval['upper']:+.4f}]；{interval['positive_folds']}/5折为正。",'',
            '|Fold|Random训练Step|Full训练Step|Random训练Val|Full训练Val|ΔC|','|---|---|---|---|---|---|']
    for row in differences.itertuples():
        lines.append(f'|{row.fold}|{row.random_train_selected_step}|{row.full_train_selected_step}|{row.random_train_val_cindex:.4f}|{row.full_train_val_cindex:.4f}|{row.delta_cindex:+.4f}|')
    lines+=['','## 随机推理与解释边界','',
            f'最终Random128评估预设seed={seed}、repeat0..{repeats-1}，每例独立无放回；Train/Val使用同一规则。',
            'repeat0为一个预先固定的随机子集策略；repeat mean先逐次计算指标，再取平均，未平均Risk形成集成。',
            '随机平均策略的五折SD由每Fold重复均值计算；random_repeat_variability.csv单独记录五折均值的重复SD。',
            '不根据Outer Val选择更好的评估策略/K/种子。降低Gap不等于提高验证性能，需同时看Val/AUC和逐折差值。',
            '所有结果仍为已反复探索队列的内部验证；配对区间不包含重训练、调参及前期探索的不确定性。',
            '本实验测试训练采样的整体效果，不能直接证明生物学组织质量，也不能保证一定缓解过拟合。','']
    (root/'report.md').write_text('\n'.join(lines))
    return pd.DataFrame(summary),interval
