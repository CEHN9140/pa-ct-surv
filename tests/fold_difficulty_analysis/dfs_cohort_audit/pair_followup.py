"""No-training Train/Val EE/EC, event gap, batch and censoring diagnostics."""
import json
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from lifelines import KaplanMeierFitter
from sksurv.metrics import concordance_index_censored, concordance_index_ipcw
from sksurv.nonparametric import CensoringDistributionEstimator
from sksurv.util import Surv
from audit import HERE, ROOT, pairs_for

SEEDS = [123, 7, 42, 2026]
TAUS = [36., 60.]


def classified_pairs(time, event, risk, batch):
    p = pairs_for(time, event, risk)
    a = p.anchor.to_numpy(); b = p.other.to_numpy()
    p['gap'] = np.asarray(time)[b] - np.asarray(time)[a]
    p['gap_bin'] = pd.cut(p.gap, [0,6,12,24,np.inf], labels=['0-6','6-12','12-24','>24']).astype(object)
    p.loc[p.pair_type != 'EE', 'gap_bin'] = np.nan
    p['risk_margin'] = np.asarray(risk)[a] - np.asarray(risk)[b]
    p['batch_pair'] = [f'{x}->{y}' for x,y in zip(np.asarray(batch)[a], np.asarray(batch)[b])]
    p['batch_scope'] = np.where(np.asarray(batch)[a] == np.asarray(batch)[b], 'within', 'cross')
    return p


def uno_sensitivity(ytrain, time, event, risk, tau):
    time = np.asarray(time); event = np.asarray(event,bool); risk = np.asarray(risk)
    y = Surv.from_arrays(event,time)
    p = pairs_for(time,event,risk)
    p = p.loc[time[p.anchor.to_numpy()] < tau]
    if not len(p):
        raise ValueError('No comparable event anchors below tau')
    censor = CensoringDistributionEstimator().fit(ytrain)
    weights = np.zeros(len(time))
    mask = time < tau
    weights[mask] = censor.predict_ipcw(y[mask]) ** 2
    w = weights[p.anchor.to_numpy()]
    score = float(np.average(p.credit, weights=w))
    reference = concordance_index_ipcw(ytrain,y,risk,tau=tau)[0]
    if abs(score-reference) >= 1e-12:
        raise AssertionError('Uno pair replay differs from official implementation')
    return dict(tau=tau, n_pairs=len(p), n_event_anchors=int(np.unique(p.anchor).size),
                harrell_same_pairs=float(p.credit.mean()), uno_cindex=score,
                min_event_censor_survival=float(1/np.sqrt(w.max())), max_ipcw_squared=float(w.max()))


def records(p, columns, risk_sd=1., **context):
    result=[]
    for key,g in p.groupby(columns, observed=True, dropna=False):
        key = key if isinstance(key,tuple) else (key,)
        result.append(dict(**context, **dict(zip(columns,key)), n_pairs=len(g), cindex=float(g.credit.mean()),
                           correct_credit=float(g.credit.sum()), risk_margin_median=float(g.risk_margin.median()), risk_margin_z_median=float(g.risk_margin.median()/risk_sd),
                           abs_risk_margin_z_median=float(g.risk_margin.abs().median()/risk_sd),
                           risk_tie_fraction=float((g.credit==.5).mean())))
    return result


def markdown(frame):
    return '```text\n' + frame.to_string(index=False) + '\n```'


def run():
    out = HERE / 'followup_results'; out.mkdir(exist_ok=True)
    c = pd.read_csv(ROOT/'data/trident/features_uni_v1/all_label_roi64.csv', dtype={'pa_id':str,'ct_id':str})
    m = pd.read_csv(HERE.parent/'cohort.csv', dtype={'pa_id':str,'ct_id':str})
    c = c.merge(m[['pa_id','ct_id','source_batch','time','event','split']], on=['pa_id','ct_id'],
                validate='one_to_one', suffixes=('','_metadata'))
    assert len(c)==721 and c.event.sum()==251
    assert np.array_equal(c.time.to_numpy(np.float32),c.time_metadata.to_numpy(np.float32))
    assert (c.event==c.event_metadata).all() and (c.split==c.split_metadata).all() and c.source_batch.notna().all()
    composition=[]
    for batch,g in c.groupby('source_batch'):
        km=KaplanMeierFitter().fit(g.time,event_observed=g.event)
        follow=KaplanMeierFitter().fit(g.time,event_observed=1-g.event)
        composition.append(dict(batch=batch,n=len(g),events=int(g.event.sum()),event_fraction=g.event.mean(),
                                event_time_median=g.loc[g.event==1,'time'].median(),censor_time_median=g.loc[g.event==0,'time'].median(),
                                reverse_km_followup_median=follow.median_survival_time_,at_risk36=int((g.time>=36).sum()),at_risk60=int((g.time>=60).sum()),
                                survival36=float(km.predict(36)) if (g.time>=36).any() else np.nan,
                                survival60=float(km.predict(60)) if (g.time>=60).any() else np.nan))
    pd.DataFrame(composition).to_csv(out/'batch_followup.csv',index=False)
    types=[]; gaps=[]; batches=[]; scopes=[]; risks=[]; censoring=[]; totals=[]; batch_associations=[]
    for fold in range(5):
        groups={role:c.loc[(c.split==fold) if role=='val' else (c.split!=fold)].reset_index(drop=True) for role in ['train','val']}
        train=groups['train']; ytrain=Surv.from_arrays(train.event.astype(bool),train.time.to_numpy(np.float32))
        for seed in SEEDS:
            base=HERE.parent/('module_freeze' if seed==123 else f'module_freeze_repeats/seed_{seed}')
            predictions={}
            for role,g in groups.items():
                p=pd.read_csv(base/f'fold_{fold}/private/outer/baseline/results/{role}_predictions.csv',dtype={'case_id':str}).set_index('case_id')
                assert p.index.is_unique and set(p.index)==set(g.pa_id)
                p=p.loc[g.pa_id]
                assert np.array_equal(g.time.to_numpy(np.float32),p['dfs.month'].to_numpy(np.float32))
                assert np.array_equal(g.event.to_numpy(bool),p['dfs.status'].to_numpy(bool)) and np.isfinite(p.risk_score).all()
                predictions[role]=p.risk_score.to_numpy()
            mean=predictions['train'].mean();sd=predictions['train'].std()
            assert sd>0
            for role,g in groups.items():
                context=dict(fold=fold,seed=seed,split=role)
                time=g.time.to_numpy(np.float32);event=g.event.to_numpy(bool);risk=predictions[role]
                pairs=classified_pairs(time,event,risk,g.source_batch.to_numpy())
                score=concordance_index_censored(event,time,risk)[0]
                assert abs(score-pairs.credit.mean())<1e-12
                totals.append(dict(**context,n=len(g),events=int(event.sum()),cindex=score,n_pairs=len(pairs)))
                types+=records(pairs,['pair_type'],risk_sd=sd,**context)
                gaps+=records(pairs.loc[pairs.pair_type=='EE'],['gap_bin'],risk_sd=sd,**context)
                batches+=records(pairs,['pair_type','batch_pair'],risk_sd=sd,**context)
                scopes+=records(pairs,['pair_type','batch_scope'],risk_sd=sd,**context)
                z=(risk-mean)/sd
                for batch,group in g.groupby('source_batch'):
                    ix=group.index.to_numpy()
                    risks.append(dict(**context,batch=batch,n=len(group),risk_z_mean=float(z[ix].mean()),risk_z_sd=float(z[ix].std())))
                first=z[g.source_batch.to_numpy()=='第一批'];second=z[g.source_batch.to_numpy()=='第二批']
                batch_associations.append(dict(**context,second_vs_first_risk_auc=float(mannwhitneyu(second,first).statistic/(len(second)*len(first)))))
                for tau in TAUS:
                    try:
                        result=uno_sensitivity(ytrain,time,event,risk,tau)
                        censoring.append(dict(**context,**result,status='ok',error=''))
                    except ValueError as error:
                        censoring.append(dict(**context,tau=tau,status='unsupported',error=str(error)))
    frames=dict(overall=pd.DataFrame(totals),ee_ec=pd.DataFrame(types),ee_gap=pd.DataFrame(gaps),batch_pairs=pd.DataFrame(batches),
                batch_scope=pd.DataFrame(scopes),batch_risk=pd.DataFrame(risks),batch_risk_association=pd.DataFrame(batch_associations),censoring=pd.DataFrame(censoring))
    for name,frame in frames.items():frame.to_csv(out/f'{name}.csv',index=False)
    # Validation replay against preceding audit, including identical training seeds.
    prior=pd.read_csv(HERE/'results/seed_fold_metrics.csv')
    joined=frames['overall'].query("split == 'val'").merge(prior,on=['fold','seed'],validate='one_to_one')
    assert len(joined)==20 and np.allclose(joined.cindex_x,joined.cindex_y,rtol=0,atol=1e-12)
    summaries={}
    for name,cols in [('ee_ec',['split','pair_type']),('ee_gap',['split','gap_bin']),('batch_scope',['split','pair_type','batch_scope'])]:
        df=frames[name];s=df.groupby(cols,observed=True).agg(cindex_mean=('cindex','mean'),cindex_model_sd=('cindex','std'),
                                                          pair_entries=('n_pairs','sum'),credit=('correct_credit','sum')).reset_index()
        s['pair_weighted_cindex']=s.credit/s.pair_entries
        if name=='ee_gap':
            s['gap_bin']=pd.Categorical(s.gap_bin,categories=['0-6','6-12','12-24','>24'],ordered=True)
            s=s.sort_values(['split','gap_bin'])
        # Train patients appear in four folds and each fold appears in four seeds. Not unique patient-pair counts.
        s.to_csv(out/f'{name}_summary.csv',index=False);summaries[name]=s
    sens=frames['censoring'].query("status == 'ok'").groupby(['split','tau']).agg(harrell_same_pairs=('harrell_same_pairs','mean'),uno=('uno_cindex','mean'),max_weight=('max_ipcw_squared','max')).reset_index()
    sens['uno_minus_harrell']=sens.uno-sens.harrell_same_pairs;sens.to_csv(out/'censoring_summary.csv',index=False)
    foldtypes=frames['ee_ec'].groupby(['fold','split','pair_type']).cindex.mean().unstack('pair_type').reset_index()
    foldtypes.to_csv(out/'fold_ee_ec.csv',index=False)
    directional=frames['batch_pairs'].groupby(['split','pair_type','batch_pair']).agg(pair_entries=('n_pairs','sum'),credit=('correct_credit','sum')).reset_index()
    directional['pair_weighted_cindex']=directional.credit/directional.pair_entries
    directional.to_csv(out/'batch_direction_summary.csv',index=False)
    association=frames['batch_risk_association'].groupby('split').second_vs_first_risk_auc.mean().reset_index()
    association.to_csv(out/'batch_risk_association_summary.csv',index=False)
    report=f'''# Train/Val EE–EC、事件时间间隔、收集批次与删失敏感性

复用四个Seed（123/7/42/2026）×五折的匹配Baseline最终checkpoint；这些是内层STOP选轮后Outer Train重训的最终checkpoint，不是统一训练到最后一轮；Train预测由同一checkpoint的eval()计算。无新训练、选模、标签修改或病例删除。40份预测完成成员/标签/时间核对和C-index复算，Val与上一审计20项结果一致。统计单元仍是患者，不是独立患者对或独立20个模型队列。

## 1. Train/Val EE与EC

{markdown(summaries['ee_ec'])}

{markdown(foldtypes)}

cindex_mean是20个模型的均值；pair_weighted_cindex为描述性的患者对加权均值，均不跨模型比较原始风险值。pair_entries重复计数训练患者/Seed，绝非独立样本量。Train表现是拟合能力，不是独立预测性能。

## 2. 仅EE：事件时间间隔

{markdown(summaries['ee_gap'])}

时间差采用(0,6]、(6,12]、(12,24]、>24个月。同时间的EE不构成可比较对。ee_gap.csv保存每折每Seed的计数、风险margin中位数、以Outer Train风险SD标准化的有符号/绝对margin中位数与risk-tie比例。更长间隔易排序并不证明模型能精确预测时间；它也可能反映病例组成差异。

## 3. 数据收集批次

{markdown(pd.DataFrame(composition))}

{markdown(summaries['batch_scope'])}

{markdown(directional)}

{markdown(association)}

batch_pairs.csv中的A->B表示事件锚点来自A、比较对象来自B，保留方向；补充病例单独保留，不并入任一大批次；指定时间无患者仍在随访时不外推该批次KM值。batch_risk.csv使用各模型Outer Train风险均值/SD标准化，batch_risk_association.csv报告第二批风险高于第一批的概率（Mann–Whitney AUC，ties记半分），不是DFS AUC。

批次是既有文件收集批次，不是已验证的医院、扫描仪或染色批次。原始事件比例与删失和随访长度相关；KM估计仍依赖相应删失假设。同批/跨批排序差异未经病例构成匹配，不能直接判为技术混杂或因果机制。边际风险差异也不能证明模型编码了技术批次。

## 4. 删失敏感性：预设36/60月

{markdown(sens)}

Uno使用对应Outer Train估计删失KM。Harrell_same_pairs与Uno使用完全相同的事件锚点t<tau，保留晚于tau的比较对象；不将样本直接删到tau前，不把完整Harrell与截断Uno的差当成删失修正效果。所有成功结果均以官方concordance_index_ipcw复算。unsupported将原样记录，不改tau寻找有利结果。

Uno采用边际删失KM，不能证明删失与影像/批次独立，亦不能修复信息性删失。这个对照针对总体排序，不能自动修复EE预测或解释时间AUC。

方法依据：[scikit-survival IPCW官方文档](https://scikit-survival.readthedocs.io/en/stable/api/generated/sksurv.metrics.concordance_index_ipcw.html)。

## 本轮结果与下一步判断

1. EE并非只在Val较弱：Train均值0.6757、Val0.5794；二者均低于EC（0.9014/0.7817）。在当前选定checkpoint下，模型同时存在较弱的训练EE区分和Train→Val迁移损失；因训练预算由内层选定，Train EE不是模型最大拟合能力，不能简单归结为‘只学会是否复发’或‘完全无法学到时间信息’。
2. EE时间越近越难：≤6月Train/Val约0.533/0.515；>24月约0.757/0.599。近时间差是部分原因，但较大间隔仍有明显泛化差距。此处不是事件时间预测误差，也不是个体事件顺序的理论上限。
3. 第二批随访中位数（reverse-KM）79月，第一批61月；原始事件比例受观察机会影响。KM36/60仍有差异，但临床病例组成未知，不能据此判定技术批次混杂。验证集同批EC0.768、跨批EC0.798；第一批内部EC约0.813、第二批约0.757（患者对加权描述性均值），不支持高EC完全由跨批比较产生。第二批风险更高与真实病例差异、删失和技术因素均可能相关，尚未分离。
4. 固定tau36/60后，Val Uno与同一患者对Harrell仅差约-0.00078/-0.00038。当前指定时间窗内边际IPCW修正没有实质改变总体排序估计；不能推断不存在信息性删失，也不能据此解释全时段指标或单独EE。
5. 这些结果尚未提供选择新Loss/编码器的充分依据。优先保留匹配内层选模与低探索成本，不依据Outer Val挑选新配置。若下一步要验证表示，应采用预先固定的公平对照；并非本轮自动启动新的模型。

## 解释边界

所有结果仍是已反复探索的同队列内部诊断。Train EE高/Val EE低支持训练拟合未迁移；不等于患者数不足的因果证明，也不是UNI信息上限证据。不可据Outer Val删除批次、筛患者或选loss并声称独立泛化改进。没有启动任何新的模型或损失函数实验。
'''
    (out/'report.md').write_text('\n'.join(x.rstrip() for x in report.splitlines())+'\n')
    (out/'validation.json').write_text(json.dumps(dict(training_started=False,canonical_patients=721,events=251,seeds=SEEDS,
          checked_prediction_sets=len(totals),prior_validation_replays=len(joined),ipcw_reference_replays=int((frames['censoring'].status=='ok').sum()),
          unsupported_ipcw=int((frames['censoring'].status!='ok').sum()),taus_months=TAUS,patient_outputs_written=False),indent=2))
    print(summaries['ee_ec'].to_string(index=False));print(summaries['ee_gap'].to_string(index=False));print(sens.to_string(index=False))


if __name__=='__main__':
    run()
