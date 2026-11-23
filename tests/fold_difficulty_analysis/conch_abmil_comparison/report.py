"""Three matched CONCH pipelines and exploratory subgroup uncertainty."""
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sksurv.metrics import concordance_index_censored
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(HERE.parent/'dfs_cohort_audit'))
from audit import pairs_for
from compare import paired_group_interval


def build_report():
    root=HERE/'results';reference=HERE.parent/'titan_aggregation_comparison/results';config=json.loads((HERE/'config.json').read_text())
    m=pd.read_csv(root/'metrics.csv');assert len(m)==5 and set(m.fold)==set(range(5))
    curves=[];selections=[];audits=[]
    for fold in range(5):
        d=root/f'private/fold_{fold}'
        for name,collection in [('inner_curves',curves),('step_selection',selections),('fit_audit',audits)]:
            collection.append(pd.read_csv(d/f'{name}.csv').assign(fold=fold))
        complete=json.loads((d/'completed.json').read_text())
        assert complete['neural_fits']==4 and complete['outer_val_evaluations']==1 and complete['checkpoint_reload_exact']
    audit=pd.concat(audits,ignore_index=True);assert len(audit)==20 and audit.groupby('fold').initial_state_sha256.nunique().eq(1).all()
    assert audit.loc[audit.role=='outer','evaluation_calls'].eq(0).all()
    assert audit.loc[audit.role=='inner','evaluation_calls'].eq(len(config['candidate_steps'])).all()
    curves=pd.concat(curves,ignore_index=True);selections=pd.concat(selections,ignore_index=True)
    assert len(curves)==5*3*len(config['candidate_steps']) and len(selections)==5*len(config['candidate_steps'])
    assert audit.loc[audit.role=='inner','optimizer_steps'].eq(max(config['candidate_steps'])).all()
    assert np.array_equal(audit.query("role=='outer'").set_index('fold').sort_index().optimizer_steps.to_numpy(),m.set_index('fold').sort_index().selected_step.to_numpy())
    for data,name in [(curves,'inner_curves'),(selections,'step_selection'),(audit,'fit_audit')]:data.to_csv(root/f'{name}.csv',index=False)
    tables={'abmil':m,'conch_mean':pd.read_csv(reference/'conch_mean/metrics.csv'),'titan':pd.read_csv(reference/'titan/metrics.csv')}
    frames={name:[] for name in tables};pair_rows=[];summary=[]
    for name,table in tables.items():
        summary.append(dict(model=name,train_cindex=table.train_cindex.mean(),val_cindex=table.val_cindex.mean(),val_sd=table.val_cindex.std(ddof=1),
                            gap=table.gap.mean(),auc36=table.auc_36m.mean() if table.auc_36m.notna().all() else np.nan,
                            auc60=table.auc_60m.mean() if table.auc_60m.notna().all() else np.nan))
        for fold in range(5):
            base=root/f'private/fold_{fold}/private/outer' if name=='abmil' else reference/name/f'private/fold_{fold}'
            for role in ['train','val']:
                p=pd.read_csv(base/f'{role}_predictions.csv',dtype={'case_id':str})
                if name=='abmil':p['risk_score']=p.risk_score.astype(np.float32)
                assert p.case_id.is_unique
                t=p['dfs.month'].to_numpy();e=p['dfs.status'].to_numpy(bool);risk=p.risk_score.to_numpy()
                assert abs(concordance_index_censored(e,t,risk)[0]-table.set_index('fold').loc[fold,role+'_cindex'])<1e-12
                pairs=pairs_for(t,e,risk);pairs['gap']=t[pairs.other.to_numpy()]-t[pairs.anchor.to_numpy()]
                for group in config['groups']:
                    g=pairs if group=='all' else pairs[(pairs.pair_type=='EE')&(pairs.gap>24)] if group=='EE_gap>24' else pairs[pairs.pair_type==group]
                    pair_rows.append(dict(model=name,fold=fold,split=role,group=group,n_pairs=len(g),cindex=g.credit.mean()))
                if role=='val':frames[name].append(p)
    summary=pd.DataFrame(summary);summary.to_csv(root/'summary.csv',index=False)
    pairs=pd.DataFrame(pair_rows);pairs.to_csv(root/'pair_metrics.csv',index=False)
    group_summary=pairs.query("split=='val'").groupby(['model','group']).cindex.mean().unstack('group').reset_index();group_summary.to_csv(root/'val_pair_summary.csv',index=False)
    intervals=[];fold_deltas=[]
    for left,right in [('abmil','conch_mean'),('abmil','titan'),('titan','conch_mean')]:
        predictions=[]
        for fold in range(5):
            p=frames[left][fold].copy();q=frames[right][fold].set_index('case_id').loc[p.case_id]
            assert np.array_equal(p['dfs.month'].to_numpy(np.float32),q['dfs.month'].to_numpy(np.float32)) and np.array_equal(p['dfs.status'],q['dfs.status'])
            p['reference_risk']=q.risk_score.to_numpy();predictions.append(p)
            fold_deltas.append(dict(contrast=left+' minus '+right,fold=fold,delta_cindex=tables[left].set_index('fold').loc[fold,'val_cindex']-tables[right].set_index('fold').loc[fold,'val_cindex']))
        for group in config['groups']:
            result=paired_group_interval(predictions,group,repeats=config['bootstrap_repeats'],seed=config['bootstrap_seed'])
            if group=='all':
                actual=tables[left].val_cindex.mean()-tables[right].val_cindex.mean();assert abs(result['delta_cindex']-actual)<1e-12
                if left=='titan':
                    old=json.loads((reference/'paired_comparison.json').read_text())
                    for key in ['delta_cindex','lower','upper']:assert abs(result[key]-old[key])<1e-12
            intervals.append(dict(contrast=left+' minus '+right,**result))
    intervals=pd.DataFrame(intervals);intervals.to_csv(root/'paired_intervals.csv',index=False);pd.DataFrame(fold_deltas).to_csv(root/'fold_deltas.csv',index=False)
    def show(x):return '```text\n'+x.to_string(index=False)+'\n```'
    text=f'''# 同CONCH特征：MeanPool、从头ABMIL与冻结TITAN

新增D组为原ABMIL，768→512，Attention512→128→1，预测头512→256→1；从seed123新初始化，全部CONCHv1.5 20×512 Patch，无额外L2或Patch采样。未载入项目下游单模态checkpoint作为初始化，TITAN/MeanPool/Ridge既有结果未重训或修改。原721例、251事件及5outer/3inner患者角色不变。

ABMIL沿用Stage6：Dropout0.25、Adam lr1e-4/WD5e-4、Cox64合尾风险集，联合Cox两遍VJP/Dropout重放。15个inner完整训练180次更新，在预设{config['candidate_steps']}上按3HELD均值选步；相同时选较早者。5个outer重新初始化并训练恰好选定更新数，OuterVal仅最终评价。这里控制的是更新次数，不是统一Epoch；范围不随OuterVal调整。

## 三组同特征结果

{show(summary)}

ABMIL与Ridge两方案同时改变聚合、风险头、正则和优化，不是Attention或外部预训练的单一因果消融。比较仅表明这些完整学习流程的内部表现。MeanPool/TITAN两组Cox仍采用FIT-only StandardScaler及既定alpha网格。

## 验证EE/EC

{show(group_summary)}

子组是逐折C-index均值，不能把可比较患者对当独立样本；EE_gap>24仅事件时间差超过24月。删失病例不是确认的永久无事件病例。

## 配对患者Bootstrap

{show(intervals)}

每折从完整OuterVal患者集合配对抽样2000次，重复患者以各患者出现次数乘积赋予患者对权重；EE/EC/大时间差EE先固定mask，分母也重新计算。不是独立抽患者对，也不是仅抽成功或失败病例。所有contrast使用同一组随机抽样。all为平均五折差值；子组同样取五折差值均值。TITAN−MeanPool总体区间已复现之前结果。

这些名义95%区间条件于已拟合模型，不含重训练、内层选择、折间训练重叠、多次探索或本轮12项比较的多重性。EE区间是针对既有发现的后续探索，不能当作新独立验证。仅一个神经训练seed。

## 审计

20个新初始化训练；每折4个模型初态相同，outer中途HELD/Val评估调用为0；15inner各9个预设点评估，outer只最终评估。全部原CONCH向量/坐标hash逐例匹配上轮TITAN输入，未改标签/病例/划分。30项保存预测C-index复算通过，权重重载检查沿用Stage6 helper。代码/输入来源核验记录见source_validation.json。

所有checkpoint、患者预测、内层病例角色和源路径仅private，不发布。不会依据OuterVal筛Patch/患者或自动追加正则化、TITAN微调、学习曲线、教师/蒸馏实验。本诊断不改变正式病理–CT教师/CT学生的初始化约束。
'''
    (root/'report.md').write_text('\n'.join(line.rstrip() for line in text.splitlines())+'\n')
    (root/'validation.json').write_text(json.dumps(dict(status='completed',patients=721,events=251,neural_fits=20,inner_fits=15,outer_fits=5,
       outer_val_evaluations=5,prediction_replays=30,paired_intervals=len(intervals),bootstrap_repeats=config['bootstrap_repeats'],patient_data_private=True),indent=2)+'\n')
    print(summary.to_string(index=False),flush=True);print(intervals.to_string(index=False),flush=True)


if __name__=='__main__':build_report()
