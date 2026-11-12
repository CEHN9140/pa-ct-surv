"""Small calibrated Cox models; every transform is fitted on training rows."""

import warnings

import numpy as np
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
from sksurv.util import Surv


def ct_pipeline(dim):
    steps = [('scale', StandardScaler())]
    if dim is not None:
        steps += [('pca', PCA(n_components=dim, svd_solver='full')), ('pc_scale', StandardScaler())]
    return Pipeline(steps)


class IncrementCox:
    def __init__(self, dim=None, ridge=0.):
        self.dim, self.ridge = dim, ridge

    def fit(self, pathology, ct, time, event):
        self.pa_scaler = StandardScaler().fit(np.asarray(pathology).reshape(-1, 1))
        self.ct_pipeline = None if ct is None else ct_pipeline(self.dim).fit(np.asarray(ct, dtype=float))
        design = self.design(pathology, ct)
        penalty = np.r_[0., np.full(design.shape[1] - 1, self.ridge)]
        self.model = CoxPHSurvivalAnalysis(alpha=penalty, ties='breslow', n_iter=200, tol=1e-9)
        with warnings.catch_warnings():
            warnings.simplefilter('error', ConvergenceWarning)
            self.model.fit(design, Surv.from_arrays(np.asarray(event).astype(bool), np.asarray(time, dtype=float)))
        if not np.isfinite(self.model.coef_).all():
            raise ValueError('Nonfinite Cox coefficients')
        return self

    def design(self, pathology, ct):
        p = self.pa_scaler.transform(np.asarray(pathology).reshape(-1, 1))
        return p if self.ct_pipeline is None else np.column_stack([p, self.ct_pipeline.transform(ct)])

    def predict(self, pathology, ct=None):
        risk = self.model.predict(self.design(pathology, ct))
        if not np.isfinite(risk).all():
            raise ValueError('Nonfinite conditional Cox prediction')
        return risk


def select_increment(contexts, dimensions=(16, 32), penalties=(.01, .1, 1., 10., 100., 1000.)):
    for context in contexts:
        if set(context['train_ids']) & set(context['val_ids']):
            raise ValueError('Calibration training/validation patient overlap')
    records = []
    for dim in dimensions:
        for ridge in penalties:
            scores, failures = [], []
            for fold, context in enumerate(contexts):
                try:
                    model = IncrementCox(dim, ridge).fit(context['train_pa'], context['train_ct'],
                                                         context['train_time'], context['train_event'])
                    risk = model.predict(context['val_pa'], context['val_ct'])
                    scores.append(float(concordance_index_censored(np.asarray(context['val_event']).astype(bool),
                                        context['val_time'], risk)[0]))
                except (ValueError, ArithmeticError, ConvergenceWarning, np.linalg.LinAlgError) as exc:
                    failures.append(f'fold{fold}: {type(exc).__name__}: {exc}')
            records.append(dict(dim=dim, ridge=float(ridge), scores=scores, failures=failures,
                                mean_cindex=float(np.mean(scores)) if not failures else None))
    valid = [r for r in records if r['mean_cindex'] is not None]
    if not valid:
        raise RuntimeError(f'No converged inner Cox candidates: {records}')
    best = max(valid, key=lambda r: (r['mean_cindex'], r['ridge'], -(r['dim'] or 0)))
    return dict(dim=best['dim'], ridge=best['ridge'], inner_mean_cindex=best['mean_cindex'], candidates=records,
                selection='mean inner C-index; exact ties favor stronger ridge, then fewer PCs')


def stable_increment(comparison, positive_folds, n_folds):
    return n_folds == 5 and all(comparison['fusion_minus_' + key]['delta_cindex'] > 0
        and comparison['fusion_minus_' + key]['lower'] > 0 and positive_folds[key] >= 4 for key in ('M0', 'M1'))
