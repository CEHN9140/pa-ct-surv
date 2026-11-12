"""Mask-derived Shape and linear volume probes for the conditional second stage."""

import numpy as np
import SimpleITK as sitk
from radiomics.shape import RadiomicsShape
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from increment_cox import ct_pipeline


def mask_shape(mask):
    array = sitk.GetArrayViewFromImage(mask)
    if mask.GetDimension() != 3 or not np.isin(np.unique(array), [0, 1]).all() or not (array == 1).any():
        raise ValueError('Shape requires a nonempty binary 3D tumor mask')
    binary = sitk.Cast(mask, sitk.sitkUInt8)
    statistics = sitk.LabelShapeStatisticsImageFilter()
    statistics.Execute(binary)
    box = statistics.GetBoundingBox(1)
    cropped = sitk.RegionOfInterest(binary, list(box[3:]), list(box[:3]))
    image = sitk.Image(cropped.GetSize(), sitk.sitkFloat32)
    image.CopyInformation(cropped)
    shape = RadiomicsShape(image, cropped, label=1)
    for name in ('VoxelVolume', 'Maximum3DDiameter', 'Sphericity'):
        shape.enableFeatureByName(name)
    values = shape.execute()
    result = dict(volume_mm3=float(values['VoxelVolume']), diameter_mm=float(values['Maximum3DDiameter']),
                  sphericity=float(values['Sphericity']))
    if not np.isfinite(list(result.values())).all() or min(result.values()) <= 0:
        raise ValueError('Invalid physical Shape measurements')
    return result


def volume_probe(contexts, train_x, train_log_volume, val_x, val_log_volume,
                 dimensions=(None,), penalties=(.01, .1, 1., 10., 100., 1000.)):
    candidates = []
    for dim in dimensions:
        for ridge in penalties:
            errors = []
            for context in contexts:
                if set(context['train_ids']) & set(context['val_ids']):
                    raise ValueError('Volume-probe train/validation overlap')
                transform = ct_pipeline(dim).fit(context['train_ct'])
                model = Ridge(alpha=ridge).fit(transform.transform(context['train_ct']), context['train_volume'])
                prediction = model.predict(transform.transform(context['val_ct']))
                errors.append(float(mean_squared_error(context['val_volume'], prediction)))
            candidates.append(dict(dim=dim, ridge=float(ridge), mean_inner_mse=float(np.mean(errors))))
    best = min(candidates, key=lambda r: (r['mean_inner_mse'], -r['ridge'], r['dim'] or 0))
    transform = ct_pipeline(best['dim']).fit(train_x)
    model = Ridge(alpha=best['ridge']).fit(transform.transform(train_x), train_log_volume)
    prediction = model.predict(transform.transform(val_x))
    raw_truth, raw_prediction = np.exp(val_log_volume), np.exp(prediction)
    if not np.isfinite(raw_prediction).all():
        raise ValueError('Nonfinite reconstructed volume')
    return dict(r2_log_volume=float(r2_score(val_log_volume, prediction)),
                mae_log_volume=float(mean_absolute_error(val_log_volume, prediction)),
                r2_volume=float(r2_score(raw_truth, raw_prediction)),
                mae_mm3=float(mean_absolute_error(raw_truth, raw_prediction)),
                prediction=prediction, selected=best, candidates=candidates, transform=transform, model=model)
