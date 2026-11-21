"""Frozen late models: isolate concentration profile and patch-weight assignment."""
from pathlib import Path
import json,sys
import numpy as np
import pandas as pd
import torch
from sksurv.metrics import concordance_index_censored

BASE=Path(__file__).resolve().parent;SOURCE=BASE.parents[1];ROOT=BASE.parents[4]
sys.path.insert(0,str(ROOT))
from dataset import load_path_feature
from model.build import Pa_Model
from matching import effective,rank_preserving_profile,match_effective,patch_stats


def run():
    torch.set_num_threads(4)
    source=pd.read_csv(ROOT/'data/trident/features_uni_v1/all_label_roi64.csv',dtype={'pa_id':str})
    rows=[];predictions=[];audits=[]
    policies=['late_native','early_late_mix','same_profile_late_ranking','neff_matched_temperature','uniform']
    for fold in range(5):
        directory=SOURCE/f'bs32/diagnostics/fold_{fold}'
        frame=pd.read_csv(directory/'private_cohort.csv',dtype={'pa_id':str});assert frame.pa_id.tolist()==source.pa_id.tolist()
        last=json.loads((directory/'audit.json').read_text())['last_epoch']
        with np.load(directory/'private/epoch_005.npz') as f:early_weights=f['attention'];offsets=f['offsets']
        with np.load(directory/f'private/epoch_{last:03d}.npz') as f:
            late_risk=f['risk'];np.testing.assert_array_equal(offsets,f['offsets'])
        model=Pa_Model(feature_dim=1024,abmil_dropout=.25,l2_normalize=False).cuda().eval()
        model.load_state_dict(torch.load(SOURCE/f'bs32/checkpoints/fold_{fold}/last_model.pth',map_location='cuda',weights_only=True))
        indices=np.flatnonzero(frame.split.eq(fold));risks={k:[] for k in policies}
        with torch.no_grad():
            for index in indices:
                x=load_path_feature(source.iloc[index].pa_path,1024).cuda()
                native,_,late,h=model.mil(x,return_patch_features=True)
                start,end=offsets[index:index+2];assert end-start==len(x)
                early=torch.from_numpy(early_weights[start:end]).to(h).reshape(1,-1,1)
                mixture=(early+late)/2;mixture=mixture/mixture.sum(dim=1,keepdim=True)
                p_late=late.cpu().numpy().reshape(-1);p_mix=mixture.cpu().numpy().reshape(-1)
                p_profile=rank_preserving_profile(p_late,p_mix)
                p_temperature,temperature=match_effective(p_late,effective(p_mix))
                alternatives=dict(early_late_mix=mixture,
                    same_profile_late_ranking=torch.from_numpy(p_profile).to(h).reshape(1,-1,1),
                    neff_matched_temperature=torch.from_numpy(p_temperature).to(h).reshape(1,-1,1),
                    uniform=torch.ones_like(late)/len(x))
                risks['late_native'].append(float(native))
                native_error=abs(float(native)-float(late_risk[index]))
                if native_error>2e-5:raise ValueError('Native late risks not reproduced')
                temp_actual=alternatives['neff_matched_temperature'].cpu().numpy().reshape(-1)
                sorted_error=float(np.abs(np.sort(p_profile)-np.sort(p_mix)).max())
                matched_error=abs(effective(temp_actual)/effective(p_mix)-1)
                order=np.argsort(p_late,kind='stable')
                if sorted_error!=0 or matched_error>1e-5:raise ValueError('Concentration control mismatch')
                if (np.diff(p_profile[order])<0).any() or (np.diff(temp_actual[order])<0).any():raise ValueError('Late ranking inverted')
                audits.append(dict(fold=fold,pa_id=frame.iloc[index].pa_id,patches=len(x),temperature=temperature,
                    neff_late=effective(p_late),neff_mixture=effective(p_mix),neff_temperature=effective(temp_actual),
                    profile_sorted_error=sorted_error,temperature_neff_relative_error=matched_error,native_risk_error=native_error))
                for name,w in alternatives.items():
                    pooled=torch.bmm(w.transpose(1,2),h).reshape(1,-1)
                    risk=model.mil.risk_head(model.mil.post_compression(pooled))
                    risks[name].append(float(risk))
        times=frame.iloc[indices].time.to_numpy(np.float32);events=frame.iloc[indices].event.to_numpy(bool)
        baseline=concordance_index_censored(events,times,np.asarray(risks['late_native'],np.float32))[0]
        for name,values in risks.items():
            c=concordance_index_censored(events,times,np.asarray(values,np.float32))[0]
            rows.append(dict(fold=fold,last_epoch=last,policy=name,cindex=float(c),delta_vs_native=float(c-baseline)))
            predictions.extend(dict(fold=fold,pa_id=frame.iloc[index].pa_id,policy=name,risk=float(value)) for index,value in zip(indices,values))
        pd.DataFrame(rows).to_csv(BASE/'fold_metrics.csv',index=False)
        print('Concentration/assignment controls completed Fold',fold,flush=True)
    private=BASE/'private';private.mkdir(exist_ok=True)
    pd.DataFrame(predictions).to_csv(private/'predictions.csv',index=False)
    pd.DataFrame(audits).to_csv(private/'matching_audit.csv',index=False)
    table=pd.DataFrame(rows).groupby('policy').agg(cindex_mean=('cindex','mean'),cindex_std=('cindex','std'),
        delta_mean=('delta_vs_native','mean'),improved_folds=('delta_vs_native',lambda x:int((x>0).sum())))
    table.to_csv(BASE/'summary.csv')
    old=pd.read_csv(BASE.parent/'private/attention_intervention_predictions.csv',dtype={'pa_id':str})
    new=pd.DataFrame(predictions)
    for a,b in [('late_native','late_native'),('early_late_mix','half_early_half_late')]:
        left=new[new.policy.eq(a)].set_index('pa_id').risk;right=old[old.policy.eq(b)].set_index('pa_id').risk.loc[left.index]
        np.testing.assert_allclose(left,right,atol=2e-6,rtol=1e-6)
    audit=pd.DataFrame(audits)
    (BASE/'intervention_validation.json').write_text(json.dumps(dict(passed=True,patients=len(audit),folds=5,training_started=False,
        profile_distribution_exact=True,late_weak_ranking_preserved=True,maximum_native_risk_error=float(audit.native_risk_error.max()),
        maximum_temperature_neff_relative_error=float(audit.temperature_neff_relative_error.max()),prior_mixture_reproduced=True),indent=2))


if __name__=='__main__':run()
