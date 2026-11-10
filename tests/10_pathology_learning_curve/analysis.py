"""Aggregate repeated draws within each fold before plotting the learning curve."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def curve_summary(frame):
    columns=[k for k in ['train_cindex','val_cindex','gap','auc_36m','auc_60m',
                        'last_train_cindex','last_val_cindex','last_gap','last_auc_36m','last_auc_60m']if k in frame]
    folds=frame.groupby(['size_label','fold'])[['n_train',*columns]].agg(lambda values:values.mean(skipna=False)).reset_index()
    records=[]
    for label,group in folds.groupby('size_label'):
        row=dict(size_label=label,n_train_mean=group.n_train.mean(),n_train_min=group.n_train.min(),
                 n_train_max=group.n_train.max(),n_folds=len(group))
        for key in columns:
            row[key+'_mean']=group[key].mean(skipna=False)
            row[key+'_std']=group[key].std(skipna=False)
        records.append(row)
    return pd.DataFrame(records).sort_values('n_train_mean').reset_index(drop=True)


def plot_learning_curve(frame,output_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root=Path(output_dir);config=yaml.safe_load((root/'run_config.yaml').read_text())
    expected=5*(len(config['train_sizes'])*config['repeats']+1)
    if (len(frame)!=expected or frame.duplicated(['fold','size_label','repeat']).any()
            or not frame.optimizer_steps.eq(config['max_updates']).all()
            or not frame.validation_checks.eq(config['max_updates']//config['eval_every']).all()):
        raise ValueError('Learning curve is incomplete or update/validation budgets differ')
    if not frame.groupby('size_label').fold.nunique().eq(5).all():raise ValueError('All five folds are required')
    summary=curve_summary(frame);summary.to_csv(root/'learning_curve.csv',index=False)
    x=summary.n_train_mean.to_numpy()
    labels=[f'All\n({int(r.n_train_min)}–{int(r.n_train_max)})'if r.size_label=='full'else str(int(r.n_train_mean))
            for r in summary.itertuples()]
    with plt.rc_context({'font.size':12,'font.weight':'bold','axes.labelweight':'bold','axes.titleweight':'bold'}):
        fig,axes=plt.subplots(2,2,figsize=(13,10))
        panels=[('C-index',[('last_train_cindex','Fixed-budget train','#2878b5'),
                            ('last_val_cindex','Fixed-budget validation','#c82423'),
                            ('val_cindex','Best-validation checkpoint','#888888')]),
                ('Train − validation gap',[('last_gap','Fixed budget','#c82423'),('gap','Best checkpoint','#888888')]),
                ('36-month AUC',[('last_auc_36m','Fixed budget','#c82423'),('auc_36m','Best checkpoint','#888888')]),
                ('60-month AUC',[('last_auc_60m','Fixed budget','#c82423'),('auc_60m','Best checkpoint','#888888')])]
        for ax,(title,series)in zip(axes.flat,panels):
            for key,label,color in series:
                ax.errorbar(x,summary[key+'_mean'],yerr=summary[key+'_std'],label=label,color=color,
                            marker='o',linewidth=2.5,capsize=4,linestyle='--'if color=='#888888'else'-')
            ax.set(title=title,xlabel='Number of training patients',xticks=x,xticklabels=labels)
            if title!='Train − validation gap':ax.set_ylim(0,1)
            else:ax.axhline(0,color='black',linewidth=.8)
            ax.grid(alpha=.2);ax.legend(fontsize=10)
        fig.suptitle('UNI + ABMIL + Cox: fixed validation patients and update budget',fontweight='bold')
        fig.text(.5,.01,'Draws averaged within each fold first; error bars = SD across five folds. Internal validation.',
                 ha='center',fontsize=10)
        fig.tight_layout(rect=(0,.03,1,.95))
        fig.savefig(root/'learning_curve.png',dpi=220);fig.savefig(root/'learning_curve.pdf');plt.close(fig)
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results_root',default=str(Path(__file__).resolve().parent/'results'))
    root=Path(parser.parse_args().results_root)
    if not(root/'metrics.csv').is_file():
        candidates=list(root.glob('*/metrics.csv'))
        if len(candidates)!=1:raise ValueError('Specify --results_root for one completed run')
        root=candidates[0].parent
    plot_learning_curve(pd.read_csv(root/'metrics.csv'),root)


if __name__=='__main__':main()
