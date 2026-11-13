"""Read-only epoch-selection audit; no outer-guided changes to completed teachers."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def epoch_choices(history):
    if (history.empty or history.epoch.duplicated().any()
            or not np.isfinite(history[['epoch','val_cindex']]).all().all()):
        raise ValueError('Need finite, unique inner history epochs')
    history=history.sort_values('epoch')
    best=history.loc[history.val_cindex.idxmax()]
    rolling=history.val_cindex.rolling(3,min_periods=3).mean()
    index=rolling.idxmax() if rolling.notna().any() else None
    return dict(raw_epoch=int(best.epoch),raw_best_cindex=float(best.val_cindex),
        rolling3_epoch=None if index is None else int(history.loc[index,'epoch']),
        rolling3_mean=None if index is None else float(rolling.loc[index]))


def diagnose(source,destination):
    source,destination=Path(source),Path(destination)
    if not (source/'completed.json').is_file():raise ValueError('Require completed residual teacher')
    if destination.exists():raise FileExistsError(f'Refusing to overwrite {destination}')
    destination.mkdir(parents=True)
    config=yaml.safe_load((source/'run_config.yaml').read_text())
    checkpoints=Path(config['checkpoint_root'])
    nested=Path(config['source_run'])
    samples=pd.read_csv(Path(config['data_dir'])/'all_label_roi64.csv',dtype={'pa_id':str,'ct_id':str})
    samples['case_id']=samples.pa_id+'|'+samples.ct_id
    event_key='event' if 'event' in samples else 'dfs.status'
    time_key='time' if 'time' in samples else 'dfs.month'
    samples=samples.set_index('case_id')
    records=[];fold_epochs=[];within=[];trajectories=[]
    for fold in range(5):
        selection=json.loads((source/f'fold_{fold}/selection.json').read_text())
        roles=pd.read_csv(nested/f'fold_{fold}/inner_split_roles.csv',dtype={'case_id':str})
        for name in ['residual_r0','residual_r0p01']:
            model_rows=[]
            for inner in range(3):
                root=checkpoints/name/f'fold_{fold}/inner_{inner}'
                history=pd.read_csv(root/'training_history.csv')
                meta=yaml.safe_load((root/'checkpoint_metadata.yaml').read_text())
                choices=epoch_choices(history)
                assert choices['raw_epoch']==meta['selected_epoch']
                stop=roles.loc[roles.inner_fold.eq(inner)&roles.role.eq('stop'),'case_id']
                outcome=samples.loc[stop]
                t,e=outcome[time_key].to_numpy(),outcome[event_key].to_numpy().astype(bool)
                comparable=e[:,None]&((t[:,None]<t[None,:])|((t[:,None]==t[None,:])&~e[None,:]))
                first,last=history.iloc[0],history.iloc[-1]
                row=dict(fold=fold,name=name,inner=inner,**choices,stop_patients=len(stop),
                    stop_events=int(e.sum()),stop_comparable_pairs=int(comparable.sum()),
                    epochs_observed=len(history),updates_per_epoch=int(first.optimizer_updates),
                    first_cindex=float(first.val_cindex),last_cindex=float(last.val_cindex),
                    first_fused_cox=float(first.val_fused_cox),last_fused_cox=float(last.val_fused_cox),
                    first_pa_cox=float(first.val_pathology_cox),last_pa_cox=float(last.val_pathology_cox),
                    first_delta_mean_square=float(first.val_correction_l2),last_delta_mean_square=float(last.val_correction_l2))
                records.append(row);model_rows.append(row)
                trajectories.append(history.assign(fold=fold,name=name,inner=inner))
            raw=int(np.median([x['raw_epoch'] for x in model_rows]))
            assert raw==selection['per_candidate_epochs'][name]
            fold_epochs.append(dict(fold=fold,name=name,original_refit_epoch=raw,
                retrospective_rolling3_epoch=int(np.median([x['rolling3_epoch'] for x in model_rows])),
                primary_selected=name==selection['primary']['name']))
        frame=pd.read_csv(source/f'inner_selected/fold_{fold}/val_predictions.csv')
        details=json.loads((source/f'inner_selected/fold_{fold}/pair_changes.json').read_text())
        p,d=frame.pathology_risk.to_numpy(),frame.ct_correction.to_numpy()
        within.append(dict(fold=fold,epochs=selection['primary']['epochs'],**details['within_teacher'],
            delta_std=float(np.std(d)),pathology_std=float(np.std(p)),
            delta_std_over_pathology_std=float(np.std(d)/np.std(p))))
    pd.DataFrame(records).to_csv(destination/'inner_epoch_audit.csv',index=False)
    pd.DataFrame(fold_epochs).to_csv(destination/'epoch_rule_comparison.csv',index=False)
    pd.concat(trajectories,ignore_index=True).to_csv(destination/'inner_training_trajectories.csv',index=False)
    pd.DataFrame(within).to_csv(destination/'within_teacher_pairs.csv',index=False)
    summary=dict(raw_epoch_and_median_reproduced=True,
        comparable_pairs=sum(x['comparable_pairs'] for x in within),corrected_pairs=sum(x['corrected_pairs'] for x in within),
        broken_pairs=sum(x['broken_pairs'] for x in within),
        mean_within_teacher_delta_cindex=float(np.mean([x['delta_cindex'] for x in within])),
        epoch_alternative='Retrospective complete-window trailing3 mean of inner STOP C-index only; no outer values used.',
        limitations=['No later checkpoints exist for retrospective selections; cannot infer resulting validation performance.',
                    'Correction mean square is not its variance or standard deviation.',
                    'Absolute correction SD is scale-dependent; ratio to pathology SD is reported.',
                    'Comparable pairs share patients and are not independent observations.',
                    'Early stopped histories do not show what would happen beyond their observed epochs.'])
    (destination/'summary.json').write_text(json.dumps(summary,indent=2))
    print(pd.DataFrame(fold_epochs).to_string(index=False))
    print(pd.DataFrame(within)[['fold','epochs','reference_cindex','teacher_cindex','delta_cindex','delta_std','pathology_std','delta_std_over_pathology_std']].to_string(index=False))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    root=Path(__file__).resolve().parent
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',default=str(root/'results/residual_teacher'))
    parser.add_argument('--destination',default=str(root/'results/residual_followup/epoch_diagnosis'))
    args=parser.parse_args()
    diagnose(args.source,args.destination)
