"""Completed paired frozen-aggregation comparison; aggregate outputs only."""
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sksurv.metrics import concordance_index_censored
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[2]))
sys.path.insert(0,str(HERE.parent/'dfs_cohort_audit'))
from audit import pairs_for
sys.path.insert(0,str(HERE.parents[1]/'15_pathology_query_ct_tokens'))
from risk_calibration import paired_comparison


def build_report():
    root=HERE/'results';config=json.loads((HERE/'config.json').read_text());tables=[];inner=[];subgroups=[];frames={}
    for arm in ['conch_mean','titan']:
        d=root/arm;m=pd.read_csv(d/'metrics.csv');cv=pd.read_csv(d/'inner_cv.csv');assert len(m)==5 and len(cv)==150
        tables.append(m);inner.append(cv);frames[arm]=[]
        for fold in range(5):
            for role in ['train','val']:
                p=pd.read_csv(d/f'private/fold_{fold}/{role}_predictions.csv',dtype={'case_id':str})
                t=p['dfs.month'].to_numpy();e=p['dfs.status'].to_numpy(bool);r=p.risk_score.to_numpy()
                assert p.case_id.is_unique and abs(concordance_index_censored(e,t,r)[0]-m.set_index('fold').loc[fold,role+'_cindex'])<1e-12
                pairs=pairs_for(t,e,r);pairs['gap']=t[pairs.other.to_numpy()]-t[pairs.anchor.to_numpy()]
                groups=[(name,g) for name,g in pairs.groupby('pair_type')]+[('EE_gap>24',pairs[(pairs.pair_type=='EE')&(pairs.gap>24)])]
                for group,g in groups:subgroups.append(dict(arm=arm,fold=fold,split=role,group=group,n_pairs=len(g),cindex=g.credit.mean()))
                if role=='val':frames[arm].append(p)
    m=pd.concat(tables,ignore_index=True);cv=pd.concat(inner,ignore_index=True);sg=pd.DataFrame(subgroups)
    m.to_csv(root/'fold_metrics.csv',index=False);cv.to_csv(root/'inner_cv.csv',index=False);sg.to_csv(root/'pair_metrics.csv',index=False)
    summary=m.groupby('arm').agg(train_cindex=('train_cindex','mean'),val_cindex=('val_cindex','mean'),val_sd=('val_cindex','std'),gap=('gap','mean'),
                                 auc36=('auc_36m','mean'),auc60=('auc_60m','mean'),auc36_folds=('auc_36m','count'),auc60_folds=('auc_60m','count')).reset_index()
    for horizon in ['36','60']:summary.loc[summary[f'auc{horizon}_folds']<5,f'auc{horizon}']=np.nan
    summary.to_csv(root/'summary.csv',index=False)
    pairs=[]
    for fold in range(5):
        p=frames['titan'][fold].copy();reference=frames['conch_mean'][fold].set_index('case_id').loc[p.case_id]
        assert np.array_equal(p['dfs.month'],reference['dfs.month']) and np.array_equal(p['dfs.status'],reference['dfs.status'])
        p['mean_risk']=reference.risk_score.to_numpy();pairs.append(p)
    result=paired_comparison(pairs,repeats=config['bootstrap_repeats'],seed=config['bootstrap_seed'],reference_columns={'conch_mean':'mean_risk'})['fusion_minus_conch_mean']
    delta=m.query("arm=='titan'").set_index('fold').val_cindex-m.query("arm=='conch_mean'").set_index('fold').val_cindex
    assert abs(result['delta_cindex']-delta.mean())<1e-12
    result['positive_folds']=int((delta>0).sum());
    fold_comparison=pd.DataFrame(dict(fold=delta.index,conch_mean=m.query("arm=='conch_mean'").set_index('fold').loc[delta.index].val_cindex.to_numpy(),titan=m.query("arm=='titan'").set_index('fold').loc[delta.index].val_cindex.to_numpy(),delta_cindex=delta.to_numpy()))
    fold_comparison.to_csv(root/'fold_comparison.csv',index=False)
    (root/'paired_comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    group=sg.query("split=='val'").groupby(['arm','group']).cindex.mean().unstack('group').reset_index();group.to_csv(root/'val_pair_summary.csv',index=False)
    def show(x):return '```text\n'+x.to_string(index=False)+'\n```'
    old=pd.read_csv(HERE.parent/'encoder_ridge_comparison/results/summary.csv').query("encoder=='uni_v1'").iloc[0]
    text=f'''# 同CONCHv1.5 Patch：MeanPool与冻结TITAN聚合对照

主对照B/C：同721例/251事件、原五折、同一20×512 Patch输入、同一CONCHv1.5特征和坐标。B对全部向量float64均值，C使用官方冻结TITAN encode_slide_from_patch_features(no_proj=True)；不加Patch/Slide L2，不训练CONCH/TITAN或ABMIL，不筛Patch或患者。

固定官方TITAN revision={config['revision']}，模型权重保存在用户指定目录，运行时仅离线读取。官方输入规范与接口来源：[MahmoodLab/TITAN](https://huggingface.co/MahmoodLab/TITAN)。TITAN推理为eval/inference_mode+CUDA float16 autocast；源向量只按数值转float32以匹配官方网格累加器，未归一化。Grid冲突/非法几何直接中止，不静默合并或删病例。重复真实推理核验见smoke_validation.json。

## 五折结果

{show(summary)}

## 配对差值 TITAN − CONCH MeanPool

```json
{json.dumps(result,indent=2)}
```

{show(fold_comparison)}

本次五折平均差值{result['delta_cindex']:+.4f}，{result['positive_folds']}/5折正向；该名义区间{'包含0，尚不能确认稳定改善' if result['lower']<=0<=result['upper'] else '不包含0，但仍需独立验证'}。不能仅凭均值改善认定预训练聚合已解决泛化问题。

2000次每折内配对患者bootstrap，取五折ΔC均值，不跨fold混合未校准风险。名义95%区间条件于已训练模型，不包含重新训练、内层选择、训练折重叠或此前同队列探索的不确定性，不是独立外部验证。

## 验证集EE/EC

{show(group)}

子组数值为五折均值；每折患者对计数见pair_metrics.csv，患者对不是独立样本。EE_gap>24表示两个事件时间相差超过24个月。

## 下游协议与边界

两组均为FIT-only StandardScaler + Ridge-Cox，Outer Train内event分层3折、seed123，alpha={config['alphas']}，Breslow/n_iter1000/tol1e-9；平分取强惩罚。每候选3折必须全成功；不完整候选不可选。两组各150次inner尝试+5次outer重新拟合，Outer Val仅最终评价。

上下文UNI参考来自上一轮同网格MeanPool+Ridge（Val C={old.val_cindex:.4f}），但其TRIDENT 20×256视野不同，只能作背景，不能把C−UNI解释成纯聚合收益。本轮主要解释B/C；二者同时改变聚合计算形式与外部预训练状态，不能将差值单独归因于空间建模或预训练某一因素。

没有自动新增学习曲线、D组ABMIL、TITAN微调或教师训练。当前项目教师/学生初始化约束不变，未将项目单模态权重用于教师/学生。代码、公开指标可发布；权重、特征、患者缓存/预测/模型、原始日志与第三方gated模型代码全部保留本地。
'''
    (root/'report.md').write_text('\n'.join(line.rstrip() for line in text.splitlines())+'\n')
    (root/'validation.json').write_text(json.dumps(dict(status='completed',patients=721,events=251,inner_fit_attempts=len(cv),inner_failed_fits=int((~cv.converged).sum()),outer_fits=10,
                         prediction_replays=20,model_roundtrips=10,bootstrap_repeats=config['bootstrap_repeats'],patient_data_private=True),indent=2)+'\n')
    print(summary.to_string(index=False),flush=True);print(result,flush=True)


if __name__=='__main__':build_report()
