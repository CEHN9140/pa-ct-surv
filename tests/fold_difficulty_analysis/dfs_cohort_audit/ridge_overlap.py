"""Strict nested Stage5 Ridge vs Stage6 ABMIL matched pair diagnostics only."""
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sksurv.metrics import concordance_index_censored
from audit import HERE, ROOT, pairs_for

STATES = {'wrong':0., 'tie':.5, 'correct':1.}


def transition_counts(ridge, abmil):
    r=np.asarray(ridge);a=np.asarray(abmil);n=len(r)
    if len(a)!=n or not np.isin(r,[0.,.5,1.]).all() or not np.isin(a,[0.,.5,1.]).all():
        raise ValueError('Aligned Harrell credits required')
    result=dict(n_pairs=n)
    for name,x in STATES.items():
        for other,y in STATES.items():
            result[f'ridge_{name}__abmil_{other}']=int(((r==x)&(a==y)).sum())
    result.update(tie_involved=int(((r==.5)|(a==.5)).sum()),ridge_credit=float(r.sum()),abmil_credit=float(a.sum()),
                  ridge_cindex=float(r.mean()) if n else np.nan,abmil_cindex=float(a.mean()) if n else np.nan,
                  delta_cindex=float((a-r).mean()) if n else np.nan)
    both=result['ridge_wrong__abmil_wrong'];any_wrong=int(((r==0)|(a==0)).sum())
    result['wrong_jaccard']=both/any_wrong if any_wrong else np.nan
    result['shared_fraction_ridge_wrong']=both/int((r==0).sum()) if (r==0).any() else np.nan
    result['shared_fraction_abmil_wrong']=both/int((a==0).sum()) if (a==0).any() else np.nan
    return result


def matched_pairs(time,event,ridge,abmil,ridge_sd,abmil_sd):
    if not np.isfinite([ridge_sd,abmil_sd]).all() or min(ridge_sd,abmil_sd)<=0:
        raise ValueError('Positive finite Outer Train risk SD required')
    p=pairs_for(time,event,ridge).rename(columns={'credit':'ridge_credit'})
    q=pairs_for(time,event,abmil)
    assert np.array_equal(p[['anchor','other']],q[['anchor','other']])
    p['abmil_credit']=q.credit.to_numpy()
    i=p.anchor.to_numpy();j=p.other.to_numpy()
    p['time_gap']=np.asarray(time)[j]-np.asarray(time)[i]
    p['gap_bin']=pd.cut(p.time_gap,[0,6,12,24,np.inf],labels=['0-6','6-12','12-24','>24']).astype(object)
    p.loc[p.pair_type!='EE','gap_bin']=np.nan
    p['ridge_margin_z']=(np.asarray(ridge)[i]-np.asarray(ridge)[j])/ridge_sd
    p['abmil_margin_z']=(np.asarray(abmil)[i]-np.asarray(abmil)[j])/abmil_sd
    p['both_wrong']=(p.ridge_credit==0)&(p.abmil_credit==0)
    p['both_strong_wrong']=p.both_wrong&(p.ridge_margin_z<=-1)&(p.abmil_margin_z<=-1)
    return p


def table(frame,cols):
    return '```text\n'+frame[cols].to_string(index=False)+'\n```'


def run():
    out=HERE/'ridge_overlap_results';out.mkdir(exist_ok=True)
    private=HERE/'private/ridge_overlap';private.mkdir(parents=True,exist_ok=True)
    c=pd.read_csv(ROOT/'data/trident/features_uni_v1/all_label_roi64.csv',dtype={'pa_id':str})
    assert len(c)==721 and c.pa_id.is_unique and c.event.sum()==251
    dirs={'ridge':HERE.parent/'pathology_stage5/results','abmil':HERE.parent/'pathology_stage6/results'}
    metrics={name:pd.read_csv(d/'metrics.csv').set_index('fold') for name,d in dirs.items()}
    reference=pd.read_csv(dirs['abmil']/'fold_comparison.csv').set_index('fold')
    fold_summary=[];strata=[];pair_types=[];correlations=[];strong=[];risk_sd_rows=[]
    for fold in range(5):
        groups={role:c[(c.split==fold) if role=='val' else (c.split!=fold)].reset_index(drop=True) for role in ['train','val']}
        risks={}
        for name,d in dirs.items():
            risks[name]={}
            for role,g in groups.items():
                path=d/(f'private/fold_{fold}' if name=='ridge' else f'fold_{fold}/private/outer')/f'{role}_predictions.csv'
                pred=pd.read_csv(path,dtype={'case_id':str}).set_index('case_id')
                assert pred.index.is_unique and set(pred.index)==set(g.pa_id)
                pred=pred.loc[g.pa_id];risk=pred.risk_score.to_numpy()
                assert np.array_equal(g.time.to_numpy(np.float32),pred['dfs.month'].to_numpy(np.float32))
                assert np.array_equal(g.event.to_numpy(bool),pred['dfs.status'].to_numpy(bool)) and np.isfinite(risk).all()
                score=concordance_index_censored(g.event.astype(bool),g.time.to_numpy(np.float32),risk)[0]
                assert abs(score-metrics[name].loc[fold,f'{role}_cindex'])<1e-12
                risks[name][role]=risk
        sds={name:r['train'].std() for name,r in risks.items()}
        risk_sd_rows.append(dict(fold=fold,**{name+'_train_risk_sd':float(sd) for name,sd in sds.items()}))
        for role,g in groups.items():
            p=matched_pairs(g.time.to_numpy(np.float32),g.event.to_numpy(bool),risks['ridge'][role],risks['abmil'][role],sds['ridge'],sds['abmil'])
            context=dict(fold=fold,split=role)
            counts=transition_counts(p.ridge_credit,p.abmil_credit)
            for name in ['ridge','abmil']:
                assert abs(counts[f'{name}_cindex']-metrics[name].loc[fold,f'{role}_cindex'])<1e-12
            fold_summary.append(dict(**context,**counts))
            corr=float(spearmanr(risks['ridge'][role],risks['abmil'][role]).statistic)
            correlations.append(dict(**context,risk_spearman=corr))
            if role=='val':
                assert abs(corr-reference.loc[fold,'risk_spearman'])<1e-12
                assert abs(counts['delta_cindex']-reference.loc[fold,'delta_cindex'])<1e-12
            for kind,part in p.groupby('pair_type'):
                pair_types.append(dict(**context,pair_type=kind,**transition_counts(part.ridge_credit,part.abmil_credit)))
            for gap,part in p[p.pair_type=='EE'].groupby('gap_bin'):
                strata.append(dict(**context,gap_bin=gap,**transition_counts(part.ridge_credit,part.abmil_credit)))
            large=p[(p.pair_type=='EE')&(p.time_gap>24)]
            common=large[large.both_wrong];confident=large[large.both_strong_wrong]
            strong.append(dict(**context,n_large_ee=len(large),both_wrong=len(common),
                               both_strong_wrong=len(confident),ridge_strong_wrong=int(((large.ridge_credit==0)&(large.ridge_margin_z<=-1)).sum()),
                               abmil_strong_wrong=int(((large.abmil_credit==0)&(large.abmil_margin_z<=-1)).sum()),
                               ridge_abs_margin_z_median_common=float(common.ridge_margin_z.abs().median()),
                               abmil_abs_margin_z_median_common=float(common.abmil_margin_z.abs().median())))
            if role=='val':
                p['anchor_id']=g.pa_id.to_numpy()[p.anchor];p['other_id']=g.pa_id.to_numpy()[p.other]
                p['anchor_time']=g.time.to_numpy()[p.anchor];p['other_time']=g.time.to_numpy()[p.other]
                p.to_csv(private/f'fold_{fold}_val_pairs.csv',index=False)
                p[(p.pair_type=='EE')&(p.time_gap>24)&p.both_strong_wrong].to_csv(private/f'fold_{fold}_large_gap_strong_shared_errors.csv',index=False)
    frames=dict(fold_overall=pd.DataFrame(fold_summary),fold_pair_type=pd.DataFrame(pair_types),fold_ee_gap=pd.DataFrame(strata),
                risk_correlations=pd.DataFrame(correlations),large_gap_errors=pd.DataFrame(strong),train_risk_sd=pd.DataFrame(risk_sd_rows))
    for name,f in frames.items():f.to_csv(out/f'{name}.csv',index=False)
    summaries={}
    for name,cols in [('overall',['split']),('pair_type',['split','pair_type']),('ee_gap',['split','gap_bin'])]:
        f=frames[{'overall':'fold_overall','pair_type':'fold_pair_type','ee_gap':'fold_ee_gap'}[name]]
        counts=[col for col in f if col.startswith('ridge_') and '__abmil_' in col]
        s=f.groupby(cols,observed=True)[['n_pairs','ridge_credit','abmil_credit','tie_involved']+counts].sum().reset_index()
        for model in ['ridge','abmil']:s[f'{model}_cindex']=s[f'{model}_credit']/s.n_pairs
        s['delta_cindex']=s.abmil_cindex-s.ridge_cindex
        s['delta_fold_mean']=f.groupby(cols,observed=True).delta_cindex.mean().to_numpy()
        wrong=s.ridge_wrong__abmil_wrong
        rwrong=sum(s[f'ridge_wrong__abmil_{state}'] for state in STATES)
        awrong=sum(s[f'ridge_{state}__abmil_wrong'] for state in STATES)
        s['shared_fraction_ridge_wrong']=wrong/rwrong.replace(0,np.nan)
        s['shared_fraction_abmil_wrong']=wrong/awrong.replace(0,np.nan)
        s['wrong_jaccard']=wrong/(rwrong+awrong-wrong).replace(0,np.nan)
        if name=='ee_gap':
            s.gap_bin=pd.Categorical(s.gap_bin,categories=['0-6','6-12','12-24','>24'],ordered=True)
            s=s.sort_values(['split','gap_bin'])
        s.to_csv(out/f'{name}_summary.csv',index=False);summaries[name]=s
    short=['n_pairs','ridge_cindex','abmil_cindex','delta_cindex','delta_fold_mean','ridge_correct__abmil_correct',
           'ridge_wrong__abmil_correct','ridge_correct__abmil_wrong','ridge_wrong__abmil_wrong','tie_involved','shared_fraction_ridge_wrong','shared_fraction_abmil_wrong']
    valtypes=summaries['pair_type'].query("split=='val'").set_index('pair_type')
    large=summaries['ee_gap'].query("split=='val' and gap_bin=='>24'").iloc[0]
    val_ee=frames['fold_pair_type'].query("split=='val' and pair_type=='EE'")
    report=f'''# Stage5 Ridge-Cox与Stage6严格Nested ABMIL：匹配排序错误

原721例、251事件、同一五折。仅复用已有严格Nested模型；不使用历史Outer Val选Best ABMIL。两者都是唯一seed123，不能混称为上一轮四Seed实验。无新训练、特征提取、模型选择、标签修改或病例剔除。

20份Train/Val预测成员、标签、float32时间及风险有限性均核对。20项C-index、5项Val Spearman与5项ΔC均复现既有Stage5/6记录。

## 全部患者对

{table(summaries['overall'],['split']+short)}

## EE与EC

{table(summaries['pair_type'],['split','pair_type']+short)}

## 仅EE：按事件时间间隔

{table(summaries['ee_gap'],['split','gap_bin']+short)}

时间区间为(0,6]、(6,12]、(12,24]、>24月；同时间EE不可比较。correct=1、tie=0.5、wrong=0，风险并列容差1e-8。完整九种转移均保存在CSV；纠正/破坏只指wrong→correct/correct→wrong，不把tie强行归为错误。credit差精确重构ΔC（包括tie产生的半分）。

n_pairs是折内可比较患者对计数。Val没有跨折比较原始风险；Train患者和患者对在不同Outer Train重复出现，不能解释为独立样本。ridge_cindex/abmil_cindex/ΔC为患者对加权描述值，delta_fold_mean为五折ΔC的均值，二者不能混淆。没有基于患者对数做独立性显著性检验。

## 风险相关性

{table(frames['risk_correlations'],['fold','split','risk_spearman'])}

高相关不代表模型等价；ABMIL含投影、Attention、非线性头，且优化、正则和风险集设置与Ridge不同。收益不能归因于Attention单独作用。Ridge五折都选择alpha网格上界10000，本轮不事后扩展网格，亦不将它视作最优线性模型上限。

## 大时间差且大错误margin的EE

{table(frames['large_gap_errors'],list(frames['large_gap_errors'].columns))}

预设大时间差>24月；每个模型的margin均除以其对应Outer Train风险SD，负值表示错误方向。both_strong_wrong要求两模型均完全错误且各自margin≤-1 SD。这个阈值不是置信概率，也没有依据Outer Val选取。所有Val患者对和大margin共同错误清单仅存在private/ridge_overlap/，不发布患者ID，不据此改变标签或剔除病例。

## 本轮判断

- Val EE共{int(valtypes.loc['EE','n_pairs'])}对：双方正确{int(valtypes.loc['EE','ridge_correct__abmil_correct'])}，ABMIL纠正{int(valtypes.loc['EE','ridge_wrong__abmil_correct'])}、破坏{int(valtypes.loc['EE','ridge_correct__abmil_wrong'])}，共同错误{int(valtypes.loc['EE','ridge_wrong__abmil_wrong'])}。共同错误占Ridge/ABMIL各自EE错误约{valtypes.loc['EE','shared_fraction_ridge_wrong']:.1%}/{valtypes.loc['EE','shared_fraction_abmil_wrong']:.1%}。
- >24月EE：纠正{int(large.ridge_wrong__abmil_correct)}、破坏{int(large.ridge_correct__abmil_wrong)}，净增加{int(large.abmil_credit-large.ridge_credit)}个正确比较；患者对加权ΔC={large.delta_cindex:+.4f}，五折平均ΔC={large.delta_fold_mean:+.4f}。共享错误{int(large.ridge_wrong__abmil_wrong)}对，其中两模型均错误margin≥1训练SD的共有{int(frames['large_gap_errors'].query("split=='val'").both_strong_wrong.sum())}对；并非独立患者数。
- EE的五折平均ΔC={valtypes.loc['EE','delta_fold_mean']:+.4f}，仅{int((val_ee.delta_cindex>0).sum())}/5折为正；EC的五折平均ΔC={valtypes.loc['EC','delta_fold_mean']:+.4f}。ABMIL并非没有任何增量，但本次增量主要在EC，大时间差EE没有呈现稳定超过线性基线的收益。
- Train >24月EE的患者对加权提升约0.0382，Val约0.0014。在当前内层选轮的checkpoint下，额外训练拟合未转为同等幅度的EE增益；这不是ABMIL的最大拟合能力测试。既有整体ΔC名义区间[-0.0010,+0.0282]仍跨0，本轮未重新估计EE子组区间或进行多重比较检验。
- ‘共同出错’仅定位当前两种模型都未解决的预测关系，不能区分表示缺失、共享读取偏差、有限监督或未测临床因素。本轮不支持自动换基础模型或改变Cox Loss，更不支持认定患者标签错误。

## 解释边界

共享错误支持‘当前两种读取方式未解决这些患者对’，不证明UNI缺少信息、WSI无法预测或标签错误。纠正和破坏是当前已训练模型的描述性比较，不能直接用Outer Val错误选取patch、患者或训练约束，再在同一Val声称独立提升。训练模型、内层选参、同队列长期探索及未测临床变量仍影响结论。不会据此自动启动编码器替换或新Loss实验。
'''
    (out/'report.md').write_text('\n'.join(line.rstrip() for line in report.splitlines())+'\n')
    (out/'validation.json').write_text(json.dumps(dict(training_started=False,canonical_patients=721,events=251,seed=123,
             prediction_files_checked=20,cindex_replays=20,val_correlation_replays=5,val_delta_replays=5,
             ties_explicit=True,large_gap_months=24,strong_wrong_margin_train_sd=1,patient_data_private=True),indent=2))
    print(summaries['pair_type'][['split','pair_type']+short].to_string(index=False))
    print(summaries['ee_gap'][['split','gap_bin']+short].to_string(index=False))
    print(frames['large_gap_errors'].query("split=='val'").to_string(index=False))


if __name__=='__main__':
    run()
