"""All-epoch C/Neff, snapshot-only mass/fraction; no interpolation of missing attention."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matching import patch_stats

BASE=Path(__file__).resolve().parent;SOURCE=BASE.parents[1]


def run():
    curves=pd.read_csv(SOURCE/'results/learning_curves.csv');curves=curves[curves.batch.eq(32)].copy()
    snapshots=[];comparisons=[]
    for fold in range(5):
        directory=SOURCE/f'bs32/diagnostics/fold_{fold}'
        frame=pd.read_csv(directory/'private_cohort.csv',dtype={'pa_id':str});audit=json.loads((directory/'audit.json').read_text())
        best,last=audit['best_epoch'],audit['last_epoch']
        for epoch in audit['saved_feature_epochs']:
            with np.load(directory/f'private/epoch_{epoch:03d}.npz') as f:stats=patch_stats(f['attention'],f['offsets'])
            for split,mask in [('train',frame.split.ne(fold)),('val',frame.split.eq(fold))]:
                r=dict(fold=fold,epoch=epoch,split=split,patients=int(mask.sum()),mean_patches=float(stats['patches'][mask].mean()),
                    mean_neff=float(stats['effective'][mask].mean()),mean_effective_ratio=float(stats['effective_ratio'][mask].mean()),
                    mean_top128_mass=float(stats['top128_mass'][mask].mean()),mean_entropy=float(stats['entropy'][mask].mean()))
                old=curves[curves.fold.eq(fold)&curves.epoch.eq(epoch)&curves.split.eq(split)].iloc[0]
                assert abs(r['mean_neff']/old.effective_patches-1)<2e-5
                snapshots.append(r)
        subset=pd.DataFrame(snapshots);subset=subset[subset.fold.eq(fold)&subset.split.eq('val')].set_index('epoch')
        validation=curves[curves.fold.eq(fold)&curves.split.eq('val')].set_index('epoch')
        comparisons.append(dict(fold=fold,best_epoch=best,last_epoch=last,cindex_best=validation.loc[best,'cindex'],cindex_last=validation.loc[last,'cindex'],
            neff_best=subset.loc[best,'mean_neff'],neff_last=subset.loc[last,'mean_neff'],effective_ratio_best=subset.loc[best,'mean_effective_ratio'],
            effective_ratio_last=subset.loc[last,'mean_effective_ratio'],top128_mass_best=subset.loc[best,'mean_top128_mass'],top128_mass_last=subset.loc[last,'mean_top128_mass']))
    points=pd.DataFrame(snapshots);points.to_csv(BASE/'snapshot_attention.csv',index=False)
    pd.DataFrame(comparisons).to_csv(BASE/'best_last_comparison.csv',index=False)
    combined=curves.merge(points[['fold','epoch','split','mean_effective_ratio','mean_top128_mass']],on=['fold','epoch','split'],how='left',validate='one_to_one')
    combined.to_csv(BASE/'epoch_curve_data.csv',index=False)
    with plt.rc_context({'font.size':10,'axes.labelweight':'bold'}):
        fig,axes=plt.subplots(4,5,figsize=(20,12),sharex='col')
        for fold in range(5):
            best=comparisons[fold]['best_epoch']
            for split,color in [('train','#1f77b4'),('val','#d62728')]:
                g=combined[combined.fold.eq(fold)&combined.split.eq(split)]
                axes[0,fold].plot(g.epoch,g.cindex,color=color,label=split.title(),linewidth=2)
                axes[1,fold].plot(g.epoch,g.effective_patches,color=color,linewidth=2)
                known=g[g.mean_top128_mass.notna()]
                axes[2,fold].scatter(known.epoch,known.mean_top128_mass,color=color,s=28)
                axes[3,fold].scatter(known.epoch,known.mean_effective_ratio,color=color,s=28)
            axes[0,fold].set_title(f'Fold {fold} | best epoch {best}')
            for row,label in enumerate(['Eval C-index','Mean effective patches','Top128 mass (snapshots)','Mean Neff/N (snapshots)']):
                axes[row,fold].set_ylabel(label);axes[row,fold].axvline(best,color='black',linestyle='--',alpha=.5)
                axes[row,fold].grid(alpha=.2)
            axes[0,fold].set_ylim(0,1);axes[1,fold].set_ylim(bottom=0)
            axes[2,fold].set_ylim(0,1);axes[3,fold].set_ylim(0,1);axes[3,fold].set_xlabel('Epoch')
        axes[0,0].legend();fig.tight_layout();fig.savefig(BASE/'concentration_epoch_curves.png',dpi=180);plt.close(fig)
    (BASE/'curve_validation.json').write_text(json.dumps(dict(passed=True,all_epoch_cindex_neff=True,snapshot_top128_mass_only=True,
        no_missing_attention_interpolated=True,ratio_is_mean_of_patient_ratios=True,snapshots_reproduce_recorded_neff=True),indent=2))


if __name__=='__main__':run()
