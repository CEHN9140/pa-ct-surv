"""Harrell pair transitions with the project's tied-time and 1e-8 risk rules."""

import numpy as np
import pandas as pd
from sksurv.metrics import concordance_index_censored


def summarize_changes(pairs):
    n = len(pairs)
    a, b = pairs.best_credit.to_numpy(), pairs.last_credit.to_numpy()
    row = dict(pairs=n, best_cindex=float(a.mean()) if n else np.nan,
               last_cindex=float(b.mean()) if n else np.nan,
               net_credit=float((b-a).sum()), lost_credit=float(np.maximum(a-b,0).sum()),
               gained_credit=float(np.maximum(b-a,0).sum()))
    for name, first, last in [('corrected_full',0,1), ('broken_full',1,0),
                              ('wrong_to_tie',0,.5), ('tie_to_correct',.5,1),
                              ('correct_to_tie',1,.5), ('tie_to_wrong',.5,0)]:
        row[name] = int(((a == first) & (b == last)).sum())
    row['delta_cindex'] = row['net_credit']/n if n else np.nan
    row['best_correct_pairs'] = int((a == 1).sum())
    row['broken_fraction_of_best_correct'] = row['broken_full']/row['best_correct_pairs'] if row['best_correct_pairs'] else np.nan
    return row


def pair_changes(time, event, best, last):
    time, event, best, last = [np.asarray(x) for x in (time,event,best,last)]
    if any(x.ndim != 1 or len(x) != len(time) for x in (event,best,last)):
        raise ValueError('Unaligned patient arrays')
    if not all(np.isfinite(x).all() for x in (time,event,best,last)) or not np.isin(event,[0,1]).all():
        raise ValueError('Nonfinite values or nonbinary events')
    event = event.astype(bool)
    mask = event[:,None] & ((time[:,None] < time[None,:]) |
                            ((time[:,None] == time[None,:]) & ~event[None,:]))
    i, j = np.nonzero(mask)
    a, b = best[i]-best[j], last[i]-last[j]
    score_a = (a > 1e-8).astype(float) + .5*(np.abs(a) <= 1e-8)
    score_b = (b > 1e-8).astype(float) + .5*(np.abs(b) <= 1e-8)
    pairs = pd.DataFrame(dict(i=i,j=j, best_credit=score_a,last_credit=score_b,
                              pair_type=np.where(event[j], 'event_event', np.where(time[i] == time[j],
                                  'event_same_time_censor','event_later_censor')),
                              event_time_bin=np.array(['0-12','12-24','24-36','>36'])[
                                  np.digitize(time[i], [12,24,36], right=True)],
                              best_gap_sd=np.abs(a)/(np.std(best)+1e-12)))
    row = summarize_changes(pairs)
    if len(pairs):
        for risk, key in [(best,'best_cindex'), (last,'last_cindex')]:
            official = concordance_index_censored(event, time, risk)[0]
            if not np.isclose(official,row[key],atol=1e-14,rtol=0):
                raise RuntimeError('Pair implementation differs from official C-index')
    counts = np.bincount(np.r_[i,j], minlength=len(time))
    patients = pd.DataFrame(dict(comparable_pairs=counts))
    for name, credit in [('error_credit_best',1-score_a), ('error_credit_last',1-score_b),
                          ('broken_credit',np.maximum(score_a-score_b,0)),
                          ('corrected_credit',np.maximum(score_b-score_a,0))]:
        patients[name] = np.bincount(np.r_[i,j], weights=np.r_[credit,credit], minlength=len(time))
    for phase in ('best','last'):
        patients[f'error_rate_{phase}'] = np.divide(patients[f'error_credit_{phase}'], counts,
            out=np.full(len(time),np.nan), where=counts>0)
    patients['net_added_error_credit'] = patients.error_credit_last-patients.error_credit_best
    return pairs, patients, row


def concentration(pairs, patients, fractions=(.05,.1,.2)):
    order = np.argsort(-patients.broken_credit.to_numpy(), kind='stable')
    lost = np.maximum(pairs.best_credit.to_numpy()-pairs.last_credit.to_numpy(),0)
    is_broken = (pairs.best_credit == 1) & (pairs.last_credit == 0)
    rows=[]
    for fraction in fractions:
        k = int(np.ceil(len(patients)*fraction))
        top = order[:k]
        covered = np.isin(pairs.i,top) | np.isin(pairs.j,top)
        rows.append(dict(patient_fraction=fraction, patients_selected=k,
                         broken_pairs_covered=int((is_broken & covered).sum()),
                         broken_pair_coverage=float((is_broken & covered).sum()/is_broken.sum()) if is_broken.sum() else np.nan,
                         lost_credit_coverage=float(lost[covered].sum()/lost.sum()) if lost.sum() else np.nan,
                         comparable_pair_coverage=float(covered.mean()) if len(pairs) else np.nan))
    return pd.DataFrame(rows)
