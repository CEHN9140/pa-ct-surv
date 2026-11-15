"""Stage4: read-only late-overfit attribution and existing sample-size curves."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
import yaml
from sksurv.metrics import concordance_index_censored

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from loss.survival_loss import cox_loss
from pairs import pair_changes, summarize_changes, concentration


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_predictions(path, patients):
    saved = pd.read_csv(path, dtype={'case_id':str})
    if saved.case_id.duplicated().any() or set(saved.case_id) != set(patients.pa_id):
        raise ValueError(f'Prediction patients differ: {path}')
    saved = saved.set_index('case_id').loc[patients.pa_id].reset_index()
    if not np.isfinite(saved[['dfs.month','dfs.status','risk_score']].to_numpy()).all():
        raise ValueError(f'Nonfinite prediction: {path}')
    if not np.array_equal(saved['dfs.month'].to_numpy(np.float32), patients.time.to_numpy(np.float32)) \
            or not np.array_equal(saved['dfs.status'].to_numpy(),patients.event.to_numpy()):
        raise ValueError(f'Prediction labels differ: {path}')
    # Restore the actual float32 risk/time before CSV decimal rounding.
    saved['risk_score'] = saved.risk_score.astype(np.float32)
    saved['dfs.month'] = saved['dfs.month'].astype(np.float32)
    return saved


def learning_summary(frame):
    rows = frame.copy()
    rows['best_train_cindex'], rows['best_val_cindex'] = rows.train_cindex, rows.val_cindex
    rows['best_gap'] = rows.train_cindex-rows.val_cindex
    rows['last_gap'] = rows.last_train_cindex-rows.last_val_cindex
    rows['patient_exposure_epochs'] = rows.training_patient_exposures/rows.n_train
    fields = ['n_train','n_events_train','best_train_cindex','best_val_cindex','best_gap',
              'last_train_cindex','last_val_cindex','last_gap','training_patient_exposures',
              'patient_exposure_epochs','best_step']
    folds = rows.groupby(['size_label','fold'])[fields].mean().reset_index()
    repeats=rows.groupby(['size_label','fold'])[['best_val_cindex','last_val_cindex']].std(ddof=1)
    repeats.columns=['within_repeat_best_val_sd','within_repeat_last_val_sd']
    repeats['n_repeats']=rows.groupby(['size_label','fold']).size()
    folds=folds.merge(repeats.reset_index(),on=['size_label','fold'],validate='one_to_one')
    records=[]
    for label, part in folds.groupby('size_label'):
        row=dict(size_label=label,n_folds=len(part))
        for field in fields:
            row[field+'_mean'],row[field+'_sd']=part[field].mean(),part[field].std(ddof=1)
        records.append(row)
    return pd.DataFrame(records).sort_values('n_train_mean').reset_index(drop=True),folds


def run_analysis(source, output):
    source, output = Path(source), Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Use a fresh output directory; never overwrite earlier analysis')
    output.mkdir(parents=True, exist_ok=True)
    (output/'private').mkdir()
    audit={}

    def track(path):
        path=Path(path).resolve()
        value=digest(path)
        if str(path) in audit and audit[str(path)] != value:
            raise RuntimeError(f'Source changed: {path}')
        audit[str(path)]=value
        return path

    config=yaml.safe_load(track(source/'run_config.yaml').read_text())
    data=Path(config['output_dir'])/'data'
    cohort_path=track(config['label_csv'])
    cohort=pd.read_csv(cohort_path,dtype={'pa_id':str,'ct_id':str})
    if len(cohort)!=721 or cohort.event.sum()!=251 or cohort.pa_id.duplicated().any() or set(cohort.split)!=set(range(5)):
        raise ValueError('Expected fixed 721-patient/251-event five-fold cohort')
    if digest(cohort_path)!=config['cohort_sha256'] or digest(track(data/'manifest.json'))!=config['subset_manifest_sha256']:
        raise ValueError('Original cohort/subset hashes differ from the training config')
    manifest=json.loads((data/'manifest.json').read_text())
    for name, expected in manifest['file_hashes'].items():
        if digest(track(data/name))!=expected:
            raise ValueError(f'Frozen sampling manifest differs: {name}')
    subsets=pd.read_csv(data/'subsets.csv',dtype={'pa_id':str})
    validation=pd.read_csv(data/'validation.csv',dtype={'pa_id':str})
    frame=pd.read_csv(track(source/'metrics.csv'))
    expected={(label,repeat,fold) for label in ('n128','n256','n384') for repeat in range(3) for fold in range(5)}
    expected|={('full',0,fold) for fold in range(5)}
    actual=set(zip(frame.size_label,frame['repeat'],frame.fold))
    if actual!=expected or len(frame)!=50 or not frame.optimizer_steps.eq(180).all() or not frame.validation_checks.eq(20).all():
        raise ValueError('Incomplete or different experiment10 update protocol')
    if config['cox_batch_size']!=64 or config['max_updates']!=180 or config['eval_every']!=9:
        raise ValueError('Unexpected training budget')
    histories,checks,totals,strata,margins,concentrations,profiles=[],[],[],[],[],[],[]
    for record in frame.itertuples(index=False):
        folder=source/record.run_name/f'fold_{record.fold}'
        history=pd.read_csv(track(folder/'update_metrics.csv'))
        if history.step.tolist()!=list(range(9,181,9)):
            raise ValueError(f'Incomplete update history: {folder}')
        if not np.isfinite(history[['train_loss','train_cindex','val_loss','val_cindex']].to_numpy()).all():
            raise ValueError(f'Nonfinite history: {folder}')
        best=history.iloc[int(np.argmax(history.val_cindex.to_numpy()))]
        if int(best.step)!=record.best_step:
            raise ValueError('Best checkpoint is not the first validation maximum')
        held=cohort.loc[cohort.split.eq(record.fold)]
        saved_val=validation.loc[validation.fold.eq(record.fold)]
        if set(saved_val.pa_id)!=set(held.pa_id) or saved_val.pa_id.duplicated().any():
            raise ValueError('Frozen validation assignment differs')
        sub=subsets.loc[subsets.fold.eq(record.fold)&subsets.size_label.eq(record.size_label)&subsets['repeat'].eq(record.repeat)]
        if sub.row_index.duplicated().any() or not sub.pa_id.tolist()==cohort.loc[sub.row_index,'pa_id'].tolist():
            raise ValueError('Frozen training patients do not align with cohort')
        training=cohort.loc[sub.row_index]
        if len(training)!=record.n_train or set(training.pa_id)&set(held.pa_id) or training.split.eq(record.fold).any():
            raise ValueError('Training subset contains validation patients or has wrong size')
        if record.size_label=='full' and set(training.pa_id)!=set(cohort.loc[cohort.split.ne(record.fold),'pa_id']):
            raise ValueError('Incomplete full-training pool')
        checkpoint=Path(config['checkpoint_root'])/record.run_name/f'fold_{record.fold}'
        metadata=yaml.safe_load(track(checkpoint/'checkpoint_metadata.yaml').read_text())
        if metadata['optimizer_steps']!=180 or metadata['best_step']!=record.best_step \
                or metadata['training_patient_exposures']!=record.training_patient_exposures:
            raise ValueError('Checkpoint metadata does not match saved evaluation')
        predictions={}
        for phase, location, snapshot in [('best',folder,best),('last',folder/'last',history.iloc[-1])]:
            for split, patients in [('train',training),('val',held)]:
                pred=read_predictions(track(location/f'{split}_predictions.csv'),patients)
                predictions[phase,split]=pred
                risk=torch.tensor(pred.risk_score.to_numpy(),dtype=torch.float32)
                t=torch.tensor(pred['dfs.month'].to_numpy(),dtype=torch.float32)
                e=torch.tensor(pred['dfs.status'].to_numpy(),dtype=torch.float32)
                ci=concordance_index_censored(e.bool().numpy(),t.numpy(),risk.numpy())[0]
                loss=float(cox_loss(risk,t,e))
                if not np.isclose(ci,snapshot[f'{split}_cindex'],rtol=0,atol=1e-12) \
                        or not np.isclose(loss,snapshot[f'{split}_loss'],rtol=0,atol=2e-5):
                    raise ValueError(f'Exported predictions do not reproduce fixed-model evaluation: {location}/{split}')
                column=('last_' if phase=='last' else '')+split+'_cindex'
                if not np.isclose(ci,getattr(record,column),atol=1e-12,rtol=0):
                    raise ValueError('Metrics table differs from predictions')
                checks.append(dict(size_label=record.size_label,repeat=record.repeat,fold=record.fold,
                                   phase=phase,split=split,cindex=ci,cox_loss=loss))
        history=history.assign(fold=record.fold,size_label=record.size_label,repeat=record.repeat,n_train=record.n_train)
        histories.append(history)
        if record.size_label!='full':
            continue
        for name in ('best_model.pth','last_model.pth'):
            track(checkpoint/name)  # provenance only; weights are not loaded or changed
        for split, patients in [('train',training),('val',held)]:
            a,b=predictions['best',split],predictions['last',split]
            pair,patient,row=pair_changes(a['dfs.month'].to_numpy(float),a['dfs.status'].to_numpy(),
                                          a.risk_score.to_numpy(float),b.risk_score.to_numpy(float))
            context=dict(fold=record.fold,split=split,best_step=int(record.best_step),last_step=180)
            totals.append(dict(**context,**row,
                               patients=len(patient),patients_with_net_added_errors=int((patient.net_added_error_credit>0).sum()),
                               patients_with_net_fewer_errors=int((patient.net_added_error_credit<0).sum()),
                               best_risk_sd=float(a.risk_score.std(ddof=0)),last_risk_sd=float(b.risk_score.std(ddof=0))))
            conc=concentration(pair,patient).assign(**context)
            concentrations.append(conc)
            for (kind,time_bin),part in pair.groupby(['pair_type','event_time_bin']):
                entry=dict(**context,pair_type=kind,event_time_bin=time_bin,**summarize_changes(part))
                entry.update(pair_mass_share=len(part)/len(pair),
                             lost_credit_share=entry['lost_credit']/row['lost_credit'] if row['lost_credit'] else np.nan,
                             delta_cindex_contribution=entry['net_credit']/len(pair))
                strata.append(entry)
            pair['margin_bin']=pd.cut(pair.best_gap_sd,[0,.1,.25,.5,1,2,np.inf],
                                      include_lowest=True,right=False,labels=['0-.1','.1-.25','.25-.5','.5-1','1-2','>=2'])
            for margin,part in pair.groupby('margin_bin',observed=True):
                margins.append(dict(**context,margin_bin=str(margin),**summarize_changes(part)))
            patient=patient.assign(case_id=a.case_id.values,time=a['dfs.month'].values,event=a['dfs.status'].values,
                                   best_risk=a.risk_score.values,last_risk=b.risk_score.values,**context)
            profiles.append(patient)
            if split=='val':
                changed=pair.loc[pair.best_credit.ne(pair.last_credit)].copy()
                changed['event_case_id']=a.case_id.to_numpy()[changed.i]
                changed['partner_case_id']=a.case_id.to_numpy()[changed.j]
                changed.drop(columns=['i','j']).to_csv(output/'private'/f'fold_{record.fold}_val_changed_pairs.csv',index=False)
            print(f"Fold {record.fold} {split}: C {row['best_cindex']:.6f}->{row['last_cindex']:.6f}; "
                  f"corrected={row['corrected_full']}, broken={row['broken_full']}, net_credit={row['net_credit']:g}",flush=True)
    summary,by_fold=learning_summary(frame)
    old=pd.read_csv(track(source/'learning_curve.csv'))
    for row in summary.itertuples():
        previous=old.loc[old.size_label.eq(row.size_label)].iloc[0]
        for ours,theirs in [('best_val_cindex_mean','val_cindex_mean'),('best_val_cindex_sd','val_cindex_std'),
                            ('last_val_cindex_mean','last_val_cindex_mean'),('last_gap_mean','last_gap_mean')]:
            if not np.isclose(getattr(row,ours),previous[theirs],rtol=0,atol=1e-12):
                raise ValueError('Learning summary does not reproduce original fold-averaged table')
    history=pd.concat(histories,ignore_index=True)
    history['gap']=history.train_cindex-history.val_cindex
    curve_folds=history.groupby(['size_label','fold','step'])[['train_cindex','val_cindex','train_loss','val_loss','gap']].mean().reset_index()
    curve_summary=curve_folds.groupby(['size_label','step'])[['train_cindex','val_cindex','train_loss','val_loss','gap']].agg(['mean','std'])
    curve_summary.columns=['_'.join(c) for c in curve_summary.columns]
    outputs=dict(fold_pair_summary=pd.DataFrame(totals),pair_strata=pd.DataFrame(strata),
                 risk_margin_breakdown=pd.DataFrame(margins),patient_concentration=pd.concat(concentrations,ignore_index=True),
                 learning_summary=summary,learning_fold_summary=by_fold,learning_run_details=frame,
                 learning_update_curves=history,learning_curve_points=curve_summary.reset_index(),
                 verified_prediction_metrics=pd.DataFrame(checks))
    for name,values in outputs.items():
        values.to_csv(output/f'{name}.csv',index=False)
    pd.concat(profiles,ignore_index=True).to_csv(output/'private/patient_errors_best_last.csv',index=False)
    from stage4_report import write_report, plot_results
    write_report(output,outputs)
    plot_results(output,outputs)
    if any(digest(path)!=expected_hash for path,expected_hash in audit.items()):
        raise RuntimeError('A source file changed during analysis')
    verification=dict(status='completed',source_run=str(source.resolve()),n_training_runs=50,
                      n_history_points=len(history),n_prediction_tables_verified=len(checks),
                      full_train_val_pair_analyses=len(totals),source_hashes_unchanged=True,
                      source_sha256=audit,neural_training_started=False,stage5_started=False,
                      risk_tie_tolerance=1e-8,
                      code_sha256={str(p.relative_to(ROOT)):digest(p) for p in [Path(__file__),
                          Path(__file__).with_name('pairs.py'),Path(__file__).with_name('stage4_report.py')]})
    (output/'validation.json').write_text(json.dumps(verification,indent=2))
    print(f'Stage4 completed: {output}; no training started.',flush=True)
    return outputs


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=ROOT/'tests/10_pathology_learning_curve/results/uni_abmil_cox_dropout0.25_updates180_eval9_coxbs64_lr0.0001_wd0.0005_seed123')
    parser.add_argument('--output',type=Path,default=Path(__file__).parent/'results')
    args=parser.parse_args()
    torch.set_num_threads(2)
    run_analysis(args.source,args.output)


if __name__=='__main__':
    main()
