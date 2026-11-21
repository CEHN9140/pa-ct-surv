"""Fixed late projection/head; intervene only on normalized patch weights."""
from pathlib import Path
import json,sys
import numpy as np
import pandas as pd
import torch
from sksurv.metrics import concordance_index_censored
from scipy.stats import spearmanr

BASE=Path(__file__).resolve().parent;ROOT=BASE.parents[3]
sys.path.insert(0,str(ROOT))
from dataset import load_path_feature
from model.build import Pa_Model
from loss.survival_loss import cox_loss


def run():
    torch.set_num_threads(4)
    source=pd.read_csv(ROOT/'data/trident/features_uni_v1/all_label_roi64.csv',dtype={'pa_id':str})
    metrics=[];predictions=[]
    for fold in range(5):
        directory=BASE.parent/f'bs32/diagnostics/fold_{fold}'
        frame=pd.read_csv(directory/'private_cohort.csv',dtype={'pa_id':str})
        assert frame.pa_id.tolist()==source.pa_id.tolist()
        last=json.loads((directory/'audit.json').read_text())['last_epoch']
        with np.load(directory/'private/epoch_005.npz',allow_pickle=False) as f:
            early_offsets=f['offsets'];early_attention=f['attention']
        with np.load(directory/f'private/epoch_{last:03d}.npz',allow_pickle=False) as f:
            late_risk=f['risk'];late_attention=f['attention'];late_offsets=f['offsets']
        assert np.array_equal(early_offsets,late_offsets)
        model=Pa_Model(feature_dim=1024,abmil_dropout=.25,l2_normalize=False).cuda().eval()
        model.load_state_dict(torch.load(BASE.parent/f'bs32/checkpoints/fold_{fold}/last_model.pth',map_location='cuda',weights_only=True))
        indices=np.flatnonzero(frame.split.eq(fold));risks={k:[] for k in ['late_native','early5_weights','half_early_half_late','uniform_weights']}
        maximum_error=0.
        with torch.no_grad():
            for index in indices:
                raw=load_path_feature(source.iloc[index].pa_path,1024).cuda()
                native,_,weights,h=model.mil(raw,return_patch_features=True)
                begin,end=late_offsets[index:index+2]
                if end-begin!=len(raw):raise ValueError('Patch order/count mismatch')
                np.testing.assert_allclose(weights.cpu().numpy().reshape(-1),late_attention[begin:end],atol=1e-7,rtol=1e-5)
                maximum_error=max(maximum_error,abs(float(native)-float(late_risk[index])))
                early=torch.from_numpy(early_attention[begin:end]).to(h).reshape(1,-1,1)
                late=weights
                alternatives=dict(early5_weights=early,half_early_half_late=(early+late)/2,uniform_weights=torch.ones_like(late)/len(raw))
                risks['late_native'].append(float(native))
                for name,w in alternatives.items():
                    w=w/w.sum(dim=1,keepdim=True)
                    pool=torch.bmm(w.transpose(1,2),h).reshape(1,-1)
                    risk=model.mil.risk_head(model.mil.post_compression(pool))
                    risks[name].append(float(risk))
        if maximum_error>2e-5:raise ValueError('Native late risk not reproduced')
        times=frame.iloc[indices].time.to_numpy(np.float32);events=frame.iloc[indices].event.to_numpy(bool)
        baseline=None
        for name,values in risks.items():
            values=np.asarray(values,np.float32)
            cindex=concordance_index_censored(events,times,values)[0]
            if name=='late_native':baseline=cindex
            row=dict(fold=fold,last_epoch=last,policy=name,cindex=float(cindex),delta_vs_late=float(cindex-baseline),
                cox_loss=float(cox_loss(torch.from_numpy(values),torch.from_numpy(times),torch.from_numpy(events).float())),
                spearman_vs_late=float(spearmanr(values,risks['late_native']).statistic),risk_reproduction_max_error=maximum_error)
            metrics.append(row)
            for index,risk in zip(indices,values):predictions.append(dict(fold=fold,pa_id=frame.iloc[index].pa_id,policy=name,risk=float(risk)))
        pd.DataFrame(metrics).to_csv(BASE/'attention_intervention_folds.csv',index=False)
        print('Frozen late attention intervention completed Fold',fold,flush=True)
    pd.DataFrame(predictions).to_csv(BASE/'private/attention_intervention_predictions.csv',index=False)
    table=pd.DataFrame(metrics).groupby('policy').agg(cindex_mean=('cindex','mean'),cindex_std=('cindex','std'),
        delta_mean=('delta_vs_late','mean'),positive_folds=('delta_vs_late',lambda x:int((x>0).sum())),mean_spearman_vs_late=('spearman_vs_late','mean'))
    table.to_csv(BASE/'attention_intervention_summary.csv')
    (BASE/'attention_intervention_validation.json').write_text(json.dumps(dict(passed=True,folds=5,network_training=False,
        only_attention_weights_changed=True,late_risks_reproduced=True,policies_predefined=True),indent=2))


if __name__=='__main__':run()
