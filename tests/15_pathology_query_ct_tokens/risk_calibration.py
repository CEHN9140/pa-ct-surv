"""Training-only splits and scalar-risk calibration for the nested diagnostic."""

import numpy as np
import torch
from scipy.optimize import minimize
from sklearn.model_selection import StratifiedKFold, train_test_split

from loss import cox_loss


def inner_splits(samples, outer_train, n_splits=3, stop_fraction=.15, seed=123):
    indices = np.asarray(outer_train, dtype=int)
    if len(np.unique(indices)) != len(indices) or n_splits < 2 or not 0 < stop_fraction < .5:
        raise ValueError('Need unique outer-training indices, >=2 inner folds and stop_fraction in (0,.5)')
    events = samples.iloc[indices].event.to_numpy()
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    result = []
    for remaining, held in splitter.split(indices, events):
        remaining, held = indices[remaining], indices[held]
        fit, stop = train_test_split(remaining, test_size=stop_fraction, random_state=seed,
                                    stratify=samples.iloc[remaining].event.to_numpy())
        result.append((fit, stop, held))
    return result


def risk_scaler(risk):
    risk = np.asarray(risk, dtype=float)
    if risk.ndim != 1 or not len(risk) or not np.isfinite(risk).all():
        raise ValueError('Risk scaler needs nonempty finite one-dimensional fit predictions')
    std = float(risk.std(ddof=0))
    return dict(mean=float(risk.mean()), std=std, scale=std if std > 1e-8 else 1., n_fit=len(risk))


def standardize_risk(risk, scaler):
    risk = np.asarray(risk, dtype=float)
    if not np.isfinite(risk).all() or not np.isfinite(scaler['scale']) or scaler['scale'] <= 0:
        raise ValueError('Nonfinite risks or invalid fitted scale')
    return (risk - scaler['mean']) / scaler['scale']


def fit_alpha(pathology, ct, time, event):
    arrays = [np.asarray(x, dtype=float) for x in (pathology, ct, time, event)]
    if (any(x.ndim != 1 or not np.isfinite(x).all() for x in arrays)
            or len({len(x) for x in arrays}) != 1 or len(arrays[0]) < 2
            or not np.isin(arrays[3], [0, 1]).all() or not arrays[3].sum()
            or (arrays[2] < 0).any()):
        raise ValueError('Need aligned finite OOF risks/times, binary events and at least one event')
    p, c, t, e = [torch.as_tensor(x, dtype=torch.float64) for x in arrays]

    def objective(value):
        alpha = torch.tensor(float(value[0]), dtype=torch.float64, requires_grad=True)
        loss = cox_loss(p + alpha * c, t, e)
        loss.backward()
        return float(loss.detach()), np.array([float(alpha.grad)])

    baseline, _ = objective([0.])
    result = minimize(objective, np.array([0.]), jac=True, method='BFGS',
                      options=dict(gtol=1e-7, maxiter=200))
    value, gradient = objective(result.x)
    if (not np.isfinite(result.x).all() or not np.isfinite(value)
            or abs(gradient[0]) > 1e-5 or value > baseline + 1e-8):
        raise RuntimeError(f'Alpha optimization did not converge: {result.message}; gradient={gradient}')
    return dict(alpha=float(result.x[0]), pathology_only_cox=baseline, fitted_cox=value,
                gradient=float(gradient[0]), iterations=int(result.nit),
                optimizer_success=bool(result.success), optimizer_message=str(result.message),
                source='standardized inner OOF predictions only', coefficient_constraint='unrestricted real')


def paired_comparison(predictions, repeats=2000, seed=123):
    """Paired patient bootstrap of the mean outer-fold C-index difference."""
    contrasts = {key: [] for key in ('pathology', 'ct')}
    observed = {key: [] for key in contrasts}
    for fold, frame in enumerate(predictions):
        t, e = frame['dfs.month'].to_numpy(), frame['dfs.status'].to_numpy().astype(bool)
        comparable = e[:, None] & ((t[:, None] < t[None, :]) |
                                  ((t[:, None] == t[None, :]) & ~e[None, :]))
        i, j = np.where(comparable)
        if not len(i):
            raise ValueError('No comparable outer-validation pairs')
        scores = {}
        for name in ('raw_pathology_risk', 'raw_ct_risk', 'risk_score'):
            risk = frame[name].to_numpy()
            delta = risk[i] - risk[j]
            scores[name] = np.where(abs(delta) <= 1e-8, .5, (delta > 0).astype(float))
        rng = np.random.default_rng(np.random.SeedSequence([seed, fold]))
        draws = rng.integers(len(t), size=(repeats, len(t)))
        counts = np.stack([np.bincount(draw, minlength=len(t)) for draw in draws]).astype(float)
        denominator = np.einsum('bi,bi->b', counts @ comparable.astype(float), counts)
        for key in contrasts:
            difference = scores['risk_score'] - scores['raw_' + key + '_risk']
            matrix = np.zeros_like(comparable, dtype=float)
            matrix[i, j] = difference
            numerator = np.einsum('bi,bi->b', counts @ matrix, counts)
            estimates = np.full(repeats, np.nan)
            np.divide(numerator, denominator, out=estimates, where=denominator > 0)
            contrasts[key].append(estimates)
            observed[key].append(float(difference.mean()))
    result = {}
    for key, values in contrasts.items():
        distribution = np.mean(values, axis=0)
        valid = distribution[np.isfinite(distribution)]
        if len(valid) < repeats / 2:
            raise ValueError('Too few valid patient bootstrap replicates')
        result['fusion_minus_' + key] = dict(delta_cindex=float(np.mean(observed[key])),
                lower=float(np.quantile(valid, .025)), upper=float(np.quantile(valid, .975)),
                valid_repeats=len(valid))
    result['method'] = ('Paired patients resampled within each outer fold; mean of fold differences. '
                        'Conditional on trained models; does not include retraining/configuration-selection uncertainty.')
    return result
