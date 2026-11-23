"""Aggregate only; private predictions/models never published."""
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sksurv.metrics import concordance_index_censored

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'dfs_cohort_audit'))
from audit import pairs_for
sys.path.insert(0,str(HERE.parents[1]/'15_pathology_query_ct_tokens'))
from risk_calibration import paired_comparison


def build_report(config):
    root=HERE/'results';metrics=[];inner=[];pair_rows=[];frames={};validations=[]
    for name in config['encoders']:
        out=root/name;m=pd.read_csv(out/'metrics.csv');cv=pd.read_csv(out/'inner_cv.csv')
        assert len(m)==5 and set(m.fold)==set(range(5)) and len(cv)==150
        metrics.append(m);inner.append(cv);validations.append(json.loads((out/'validation.json').read_text()))
        frames[name]=[]
        for fold in range(5):
            for role in ['train','val']:
                p=pd.read_csv(out/f'private/fold_{fold}/{role}_predictions.csv',dtype={'case_id':str})
                t=p['dfs.month'].to_numpy();e=p['dfs.status'].to_numpy(bool);r=p.risk_score.to_numpy()
                assert p.case_id.is_unique and np.isfinite(r).all()
                assert abs(concordance_index_censored(e,t,r)[0]-m.set_index('fold').loc[fold,f'{role}_cindex'])<1e-12
                pairs=pairs_for(t,e,r)
                pairs['gap']=t[pairs.other.to_numpy()]-t[pairs.anchor.to_numpy()]
                for kind,part in pairs.groupby('pair_type'):
                    pair_rows.append(dict(encoder=name,fold=fold,split=role,group=kind,n_pairs=len(part),cindex=part.credit.mean()))
                part=pairs[(pairs.pair_type=='EE')&(pairs.gap>24)]
                pair_rows.append(dict(encoder=name,fold=fold,split=role,group='EE_gap>24',n_pairs=len(part),cindex=part.credit.mean()))
                if role=='val':frames[name].append(p)
    metrics=pd.concat(metrics,ignore_index=True);inner=pd.concat(inner,ignore_index=True);pairs=pd.DataFrame(pair_rows)
    metrics.to_csv(root/'fold_metrics.csv',index=False);inner.to_csv(root/'inner_cv.csv',index=False);pairs.to_csv(root/'pair_metrics.csv',index=False)
    summary=metrics.groupby('encoder').agg(train_cindex=('train_cindex','mean'),val_cindex=('val_cindex','mean'),val_sd=('val_cindex','std'),
                                          gap=('gap','mean'),auc36=('auc_36m','mean'),auc60=('auc_60m','mean'),
                                          auc36_folds=('auc_36m','count'),auc60_folds=('auc_60m','count')).reset_index()
    # Do not silently average incomplete AUC horizons.
    summary.loc[summary.auc36_folds<5,'auc36']=np.nan;summary.loc[summary.auc60_folds<5,'auc60']=np.nan
    summary.to_csv(root/'summary.csv',index=False)
    paired=[]
    for name in config['encoders']:
        if name=='uni_v1':continue
        comparisons=[]
        for fold in range(5):
            p=frames[name][fold].copy();u=frames['uni_v1'][fold].set_index('case_id').loc[p.case_id]
            assert np.array_equal(p['dfs.month'],u['dfs.month']) and np.array_equal(p['dfs.status'],u['dfs.status'])
            p['uni_risk']=u.risk_score.to_numpy();comparisons.append(p)
        result=paired_comparison(comparisons,repeats=config['bootstrap_repeats'],seed=config['bootstrap_seed'],reference_columns={'uni':'uni_risk'})['fusion_minus_uni']
        deltas=metrics.query('encoder==@name').set_index('fold').val_cindex-metrics.query("encoder=='uni_v1'").set_index('fold').val_cindex
        assert abs(result['delta_cindex']-deltas.mean())<1e-12
        paired.append(dict(encoder=name,positive_folds=int((deltas>0).sum()),**result))
    paired=pd.DataFrame(paired);paired.to_csv(root/'paired_vs_uni.csv',index=False)
    selection=metrics[['encoder','fold','alpha']].copy();selection['upper_boundary']=selection.alpha==max(config['alphas'])
    selection.to_csv(root/'alpha_selection.csv',index=False)
    group=pairs.query("split=='val'").groupby(['encoder','group']).cindex.mean().unstack('group').reset_index()
    group.to_csv(root/'val_pair_summary.csv',index=False)
    def table(x):return '```text\n'+x.to_string(index=False)+'\n```'
    text=f'''# 同Patch五编码器：MeanPool + Nested Ridge-Cox

721例、251事件、原五折。UNI/KEEP/GPFM/Phikon-v2/Kaiko同一TRIDENT Patch网格、逐例坐标顺序/几何一致；全部patch逐块检查finite后float64均值池化，无新增Patch L2/筛选。各编码器保留原有预处理和原生输出，维度不强制相同；比较的是固定特征提取流程，不能进一步把差异只归因于某个预训练因素。

同一Outer Train内按事件分层三折，seed123；每一患者的内层角色与Stage5一致。StandardScaler仅在FIT上拟合。统一alpha={config['alphas']}；CoxPHSurvivalAnalysis/Breslow/n_iter1000/tol1e-9，警告视为拟合失败；候选必须三折全部成功。平分取强惩罚。Outer Train重新拟合后仅最终评价一次，UNI也按扩展网格重新拟合。

## 五折结果

{table(summary)}

mean/SD为五折描述，非独立测试。AUC为36/60月累计动态IPCW，删失分布由对应Outer Train估计。任何失败保留原因，不删病例；不完整AUC不静默平均。

## 与同新协议UNI的配对差值

{table(paired)}

固定已拟合模型，每折内配对重采样患者2000次，取五折ΔC均值；不跨fold混合原始风险。这些名义95%区间不含重训练、内层选择、折间训练重叠、此前同队列探索或四组比较的多重性影响，不能仅凭最高均值选‘胜者’。

## EE/EC及大时间差EE

{table(group)}

这里是逐折子组C-index均值，不是把患者对当独立样本。对应每折计数见pair_metrics.csv。删失患者不是永久无复发者。EE_gap>24指事件时间差超过24月，不代表可精确预测时间。

## alpha选择

{table(selection)}

到达1e8上界时只报告，不事后扩大网格，也不将其视为理论最优。旧Stage5的0.7197与本轮网格不同，不能直接当作新编码器提升的唯一参照；旧ABMIL也不与新结果直接构成公平架构比较。

## 验证与范围

完成750次内层拟合尝试和25次外层拟合。全部模型重载预测一致；50项保存Train/Val C-index复算。UNI患者均值与旧Stage5缓存逐值一致（atol1e-12）；内层角色、来源stat与代码/config/cohort检查通过。数值失败见inner_cv.csv，不静默填分。

没有微调编码器或训练新神经网络。原始H5、患者缓存/预测、joblib、病例来源信息与完整日志保持private；本实验不改变正式病理–CT教师/CT学生，也不把项目训练的单模态权重用于教师/学生初始化。
'''
    (root/'report.md').write_text('\n'.join(line.rstrip() for line in text.splitlines())+'\n')
    validation=dict(status='completed',encoders=5,patients=721,events=251,inner_fit_attempts=len(inner),inner_failed_fits=int((~inner.converged).sum()),
                    outer_fits=25,cindex_prediction_replays=50,model_roundtrips=25,patient_data_private=True,
                    same_inner_roles_as_stage5=True,bootstrap_repeats=config['bootstrap_repeats'],encoder_validation=validations)
    (root/'validation.json').write_text(json.dumps(validation,indent=2)+'\n')
    print(summary.to_string(index=False),flush=True);print(paired.to_string(index=False),flush=True)
