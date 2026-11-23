"""Matched inner-selection freeze intervention, with private prediction/parameter audit."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
from sksurv.metrics import concordance_index_censored
from train import ARMS,partition
from freeze import choose_budget,weight_hash
from model.build import Pa_Model

BASE=Path(__file__).resolve().parent;ROOT=BASE.parents[2]


def summarize():
    source=pd.read_csv(ROOT/'data/trident/features_uni_v1/all_label_roi64.csv',dtype={'pa_id':str});records=[];choices=[];audit_rows=[];predictions={a:[] for a in ARMS}
    for fold in range(5):
        directory=BASE/f'fold_{fold}';complete=json.loads((directory/'completed.json').read_text());assert complete['passed'] and complete['outer_val_passes_per_arm']==1
        fit,stop,train,val=partition(source,fold);roles=pd.read_csv(directory/'private/inner_roles.csv',dtype={'pa_id':str})
        assert set(roles[roles.role.eq('fit')].pa_id)==set(source.iloc[fit].pa_id) and set(roles[roles.role.eq('stop')].pa_id)==set(source.iloc[stop].pa_id)
        selection=pd.read_csv(directory/'selection.csv');metric=pd.read_csv(directory/'metrics.csv');audits={}
        for scope in ['inner','outer']:
            warm=directory/f'private/{scope}/warmup';initial=json.loads((warm/'audit.json').read_text());assert initial['epochs']==3 and initial['optimizer_reset']
            common=torch.load(warm/'last_model.pth',weights_only=True,map_location='cpu')
            common_model=Pa_Model(feature_dim=1024,abmil_dropout=.25);common_model.load_state_dict(common);expected_hash=weight_hash(common_model)
            for arm in ARMS:
                path=directory/f'private/{scope}/{arm}';audit=json.loads((path/'audit.json').read_text());audit_rows.append(dict(fold=fold,scope=scope,**{k:v for k,v in audit.items() if k!='order_hashes'}));audits[(scope,arm)]=audit
                assert audit['initial_hash']==expected_hash
                assert audit['frozen_unchanged'] and audit['optimizer_initial_state_entries']==0 and not audit['outer_val_used']
                assert audit['fit_patients']==len(fit if scope=='inner' else train) and audit['stop_patients']==(len(stop) if scope=='inner' else 0)
                state=torch.load(path/'last_model.pth',weights_only=True,map_location='cpu')
                prefix=['mil.compression.','mil.attention.'] if arm=='freeze_ab' else ['mil.post_compression.','mil.risk_head.'] if arm=='freeze_c' else []
                for key in common:
                    if any(key.startswith(p) for p in prefix):assert torch.equal(common[key],state[key]),f'Frozen tensor differs:{key}'
            for a in ARMS:
                for b in ARMS:
                    first,second=audits[(scope,a)],audits[(scope,b)];n=min(first['epochs'],second['epochs'])
                    assert first['initial_hash']==second['initial_hash'] and first['start_rng_hash']==second['start_rng_hash'] and first['order_hashes'][:n]==second['order_hashes'][:n]
        for arm in ARMS:
            chosen=selection[selection.arm.eq(arm)].iloc[0];curve=pd.read_csv(directory/f'private/inner/{arm}/eval_curves.csv');rows=curve[curve.split.eq('stop')]
            candidates=[(float(chosen.score_at_fork),0)]+[(float(r.cindex),int(r.epoch)) for r in rows.itertuples()]
            score,budget=choose_budget(candidates)
            assert abs(score-chosen.inner_stop_cindex)<1e-12 and budget==chosen.continuation_epochs
            assert audits[('inner',arm)]['epochs']==20 and audits[('outer',arm)]['epochs']==budget
            m=metric[metric.arm.eq(arm)].iloc[0].to_dict();m['gap']=m['train_cindex']-m['val_cindex'];records.append(m);choices.append(dict(fold=fold,arm=arm,continuation_epochs=budget,stop_cindex=score))
            for split,indices in [('train',train),('val',val)]:
                prediction=pd.read_csv(directory/f'private/outer/{arm}/results/{split}_predictions.csv',dtype={'case_id':str}).set_index('case_id')
                expected=source.iloc[indices].set_index('pa_id');assert set(prediction.index)==set(expected.index);prediction=prediction.loc[expected.index]
                np.testing.assert_array_equal(prediction['dfs.status'],expected.event);np.testing.assert_allclose(prediction['dfs.month'],expected.time.to_numpy(np.float32),atol=1e-5,rtol=0)
                c=concordance_index_censored(prediction['dfs.status'].to_numpy(bool),prediction['dfs.month'].to_numpy(np.float32),prediction.risk_score.to_numpy(np.float32))[0]
                assert abs(c-m[split+'_cindex'])<1e-12
                if split=='val':predictions[arm].append(prediction)
    out=BASE/'results';out.mkdir(exist_ok=True);frame=pd.DataFrame(records);frame.to_csv(out/'fold_metrics.csv',index=False)
    pd.DataFrame(choices).to_csv(out/'selected_epochs.csv',index=False);pd.DataFrame(audit_rows).to_csv(out/'branch_audit.csv',index=False)
    summary=[]
    for arm,g in frame.groupby('arm'):
        row=dict(arm=arm)
        for k in ['train_cindex','val_cindex','train_loss','val_loss','gap','auc_36m','auc_60m']:row[k+'_mean']=g[k].mean(skipna=False);row[k+'_std']=g[k].std(skipna=False)
        summary.append(row)
    pd.DataFrame(summary).to_csv(out/'summary.csv',index=False)
    rng=np.random.default_rng(123);intervals=[];pivot=frame.pivot(index='fold',columns='arm',values='val_cindex')
    for arm in ARMS[1:]:
        draws=[]
        for repeat in range(1000):
            delta=[]
            for fold in range(5):
                p=predictions['baseline'][fold];q=predictions[arm][fold].loc[p.index];e=p['dfs.status'].to_numpy(bool);t=p['dfs.month'].to_numpy(np.float32)
                for attempt in range(1000):
                    idx=rng.integers(len(p),size=len(p))
                    if any(e[idx][i] and np.any((t[idx]>t[idx][i])|((t[idx]==t[idx][i])&~e[idx])) for i in range(len(idx))):break
                else:raise ValueError('No comparable bootstrap pairs')
                delta.append(concordance_index_censored(e[idx],t[idx],q.risk_score.to_numpy(np.float32)[idx])[0]-concordance_index_censored(e[idx],t[idx],p.risk_score.to_numpy(np.float32)[idx])[0])
            draws.append(np.mean(delta))
        diff=pivot[arm]-pivot.baseline;intervals.append(dict(arm=arm,mean_delta=diff.mean(),positive_folds=int(diff.gt(0).sum()),interval_low=np.quantile(draws,.025),interval_high=np.quantile(draws,.975)))
    pd.DataFrame(intervals).to_csv(out/'paired_comparisons.csv',index=False)
    lines=['# Epoch3 branch-freezing intervention','',
        'Raw UNI full bags, original721/251/fivefold. Each scope starts newly; common3Epoch warmup, exact last warmup state and RNG reused across three branches, Adam reset. Frozen modules remain train-mode including Dropout; Freeze-C gradient flows through frozen head to AB. No input/loss/architecture change.',
        '', 'Single event-stratified80/20 FIT/STOP inside each Outer Train, seed123. Each inner arm runs20continuation Epochs without early stopping; STOP selects continuation0..20, tie chooses earlier. Outer Train separately warms up3new Epochs, branches and runs the fixed selected budget. Outer Val is final-only; reported Train C/Loss is eval-mode at same final checkpoint.',
        '', '| Arm | Train C | Val C | Gap | AUC36 | AUC60 |','|---|---:|---:|---:|---:|---:|']
    for r in summary:lines.append('| '+r['arm']+' | '+' | '.join(f"{r[k+'_mean']:.4f} ± {r[k+'_std']:.4f}" for k in ['train_cindex','val_cindex','gap','auc_36m','auc_60m'])+' |')
    lines+=['','Historical0.7248/0.7458 are not matched comparators: this baseline resets Adam after3Epochs and uses a different fixed continuation window. Freeze effects are not a unique causal decomposition of modules; changing trainability changes optimization and interactions. Frozen Dropout remains stochastic in training. Reduced gap alone is not improvement. Conditional1000paired patient intervals omit refitting, selection, overlapping Train folds, two comparisons and prior exploration. Internal exploratory validation only.',
        '', 'All branch initial weight/RNG hashes and patient-order prefixes, frozen tensors, candidate selection and30final Train/Val C-indices were verified. No project-trained unimodal checkpoint initializes these models. All private predictions/weights remain local. No teacher/student change or automatic follow-up experiment follows.']
    (out/'report.md').write_text('\n'.join(lines)+'\n');(out/'validation.json').write_text(json.dumps(dict(passed=True,folds=5,arms=3,frozen_parameters_exact=True,branch_pairing_verified=True,selection_verified=True,final_cindices_recomputed=True),indent=2))


if __name__=='__main__':summarize()
