"""Read-only original NIfTI -> model ROI audit; never import process_ct.py."""

import argparse
import ast
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy import ndimage


def sample_grid(image,start_xyz,size_xyz,is_mask=False):
    """Sample the finite preprocessed grid; valid means current NIfTI support, not DICOM FOV."""
    start=np.asarray(start_xyz,dtype=int);size=np.asarray(size_xyz,dtype=int)
    spacing=np.asarray(image.GetSpacing());direction=np.asarray(image.GetDirection()).reshape(3,3)
    target_size=np.rint(np.asarray(image.GetSize())*spacing).astype(int)
    origin=np.asarray(image.GetOrigin())+direction@start.astype(float)
    resampler=sitk.ResampleImageFilter()
    resampler.SetSize(size.tolist());resampler.SetOutputSpacing((1.,1.,1.))
    resampler.SetOutputOrigin(origin.tolist());resampler.SetOutputDirection(image.GetDirection())
    resampler.SetInterpolator(sitk.sitkNearestNeighbor if is_mask else sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0 if is_mask else -1000)
    values=sitk.GetArrayFromImage(resampler.Execute(image))
    z,y,x=np.ogrid[start[2]:start[2]+size[2],start[1]:start[1]+size[1],start[0]:start[0]+size[0]]
    valid=(x>=0)&(x<target_size[0])&(y>=0)&(y<target_size[1])&(z>=0)&(z<target_size[2])
    # Source preprocessing crops a FINITE full resampled array then pads it.
    values[~valid]=0 if is_mask else -1000
    acquired=valid&(x/spacing[0]>=-.5)&(x/spacing[0]<image.GetSize()[0]-.5)
    acquired=acquired&(y/spacing[1]>=-.5)&(y/spacing[1]<image.GetSize()[1]-.5)
    acquired=acquired&(z/spacing[2]>=-.5)&(z/spacing[2]<image.GetSize()[2]-.5)
    return values,acquired


def resampled_tumor(mask):
    statistics=sitk.LabelShapeStatisticsImageFilter();statistics.Execute(sitk.Cast(mask,sitk.sitkUInt8))
    if not statistics.HasLabel(1):raise ValueError('Empty original/aligned tumor mask')
    box=np.asarray(statistics.GetBoundingBox(1))
    spacing=np.asarray(mask.GetSpacing())
    # Include the full physical support of nearest-neighbor tumor voxels.
    lower=np.floor((box[:3]-.5)*spacing).astype(int)-1
    upper=np.ceil((box[:3]+box[3:]-.5)*spacing).astype(int)+1
    lower=np.maximum(lower,0)
    upper=np.minimum(upper,np.rint(np.asarray(mask.GetSize())*spacing).astype(int)-1)
    cropped,_=sample_grid(mask,lower,upper-lower+1,True)
    points=np.argwhere(cropped>0)
    if not len(points):raise ValueError('Tumor disappears after1mm resampling')
    return dict(total=int(len(points)),min_zyx=points.min(0)+lower[::-1],max_zyx=points.max(0)+lower[::-1])


def roi_statistics(mask_with_halo,acquired,roi_size,halo):
    expected=(roi_size+2*halo,)*3
    if mask_with_halo.shape!=expected or acquired.shape!=expected:
        raise ValueError('Unexpected peritumoral-grid shape')
    roi=np.zeros(expected,dtype=bool);roi[halo:halo+roi_size,halo:halo+roi_size,halo:halo+roi_size]=True
    tumor=mask_with_halo&roi;points=np.argwhere(tumor)-halo
    if not len(points):raise ValueError('Empty model-ROI tumor')
    margins=np.r_[points.min(0),roi_size-1-points.max(0)]
    distance=ndimage.distance_transform_edt(~mask_with_halo)
    result=dict(tumor_voxels=int(tumor.sum()),tumor_fraction=float(tumor.sum()/roi_size**3),
        minimum_margin_mm=float(margins.min()),tumor_touches_roi_face=bool((margins==0).any()),
        acquired_roi_fraction=float(acquired[roi].mean()))
    for radius in (5,10):
        shell=(distance>0)&(distance<=radius)
        count=int(shell.sum())
        result[f'shell{radius}_coverage']=float((shell&roi&acquired).sum()/count)
        result[f'shell{radius}_inside_roi_fraction']=float((shell&roi).sum()/count)
    return result


def intensity_statistics(values,prefix):
    values=np.asarray(values).reshape(-1)
    if not len(values):return {prefix+'_count':0}
    q=np.quantile(values,[.01,.25,.5,.75,.99])
    result={prefix+'_count':len(values),prefix+'_min':float(values.min()),prefix+'_max':float(values.max()),
        prefix+'_mean':float(values.mean()),prefix+'_std':float(values.std()),
        prefix+'_below_window':float((values<-1000).mean()),prefix+'_above_window':float((values>400).mean())}
    result.update({prefix+'_'+key:float(v) for key,v in zip(['p01','p25','p50','p75','p99'],q)})
    return result


def audit_case(job):
    row,log,raw_root=job
    identity=str(row['ct_id']);result=dict(ct_id=identity,fold=int(row['split']))
    try:
        raw_path=Path(raw_root)/'image'/(identity+'.nii.gz')
        mask_path=Path(raw_root)/'mask'/(identity+'.nii.gz')
        if raw_path.resolve()!=Path(log['image_path']).resolve() or mask_path.resolve()!=Path(log['mask_path']).resolve():
            raise ValueError('Original image/mask identity differs from preprocessing log')
        if Path(row['ct_path']).resolve()!=Path(log['processed_image_path']).resolve():
            raise ValueError('Model CT path differs from preprocessing output for this CT ID')
        image=sitk.ReadImage(str(raw_path));mask=sitk.ReadImage(str(mask_path))
        raw_array=sitk.GetArrayViewFromImage(image)
        if not np.isfinite(raw_array).all():raise ValueError('Nonfinite original CT')
        stride=max(1,int(np.ceil(raw_array.size/200000)))
        result.update(intensity_statistics(raw_array.reshape(-1)[::stride],'raw_sample'))
        result.update(raw_global_min=float(raw_array.min()),raw_global_max=float(raw_array.max()),raw_sample_stride=stride)
        for axis,value in zip('xyz',image.GetSpacing()):result['input_spacing_'+axis]=float(value)
        result['input_shape_xyz']=json.dumps(image.GetSize())
        result['input_pixel_type']=image.GetPixelIDTypeAsString()
        fields={'scanner_manufacturer':'0008|0070','scanner_model':'0008|1090','kvp':'0018|0060',
                'slice_thickness':'0018|0050','reconstruction_kernel':'0018|1210','tube_current':'0018|1151'}
        for name,key in fields.items():result[name]=image.GetMetaData(key) if image.HasMetaDataKey(key) else None
        result['raw_geometry_equal']=bool(image.GetSize()==mask.GetSize() and all(np.allclose(a,b)
            for a,b in [(image.GetSpacing(),mask.GetSpacing()),(image.GetOrigin(),mask.GetOrigin()),(image.GetDirection(),mask.GetDirection())]))
        if not np.isin(np.unique(sitk.GetArrayViewFromImage(mask)),[0,1]).all():raise ValueError('Nonbinary original tumor mask')
        image=sitk.DICOMOrient(image,'LPS');mask=sitk.DICOMOrient(mask,'LPS')
        direction=np.asarray(image.GetDirection()).reshape(3,3)
        if not np.allclose(direction.T@direction,np.eye(3),atol=1e-5):raise ValueError('Nonorthonormal image direction')
        geometry_equal=image.GetSize()==mask.GetSize() and all(np.allclose(a,b)
            for a,b in [(image.GetSpacing(),mask.GetSpacing()),(image.GetOrigin(),mask.GetOrigin()),(image.GetDirection(),mask.GetDirection())])
        if not geometry_equal:mask=sitk.Resample(mask,image,sitk.Transform(),sitk.sitkNearestNeighbor,0,mask.GetPixelID())
        support=resampled_tumor(mask)
        span=support['max_zyx']-support['min_zyx']+1
        expected_center=support['min_zyx']-(64-span)//2+32
        center=np.asarray(ast.literal_eval(log['tumor_center_zyx']),dtype=int)
        issues=[]
        if not np.array_equal(center,expected_center):issues.append('crop_center_mismatch')
        if not np.array_equal(support['min_zyx'],ast.literal_eval(log['tumor_min_zyx'])):issues.append('tumor_min_mismatch')
        if not np.array_equal(support['max_zyx'],ast.literal_eval(log['tumor_max_zyx'])):issues.append('tumor_max_mismatch')
        start=(center-32)[::-1]
        roi_hu,acquired=sample_grid(image,start,[64]*3)
        saved=np.load(row['ct_path']);saved_mask=np.load(log['processed_mask_path'])
        if saved.shape!=(1,64,64,64) or saved_mask.shape!=(1,64,64,64):raise ValueError('Wrong saved ROI/mask shape')
        if not np.isfinite(saved).all() or saved.min()<0 or saved.max()>1:raise ValueError('Invalid model CT range')
        expected=np.clip(roi_hu.astype(np.float32),-1000,400)
        expected=(expected+1000)/1400
        error=np.abs(saved[0]-expected)
        result.update(roi_exact_match=bool(np.array_equal(saved[0],expected)),
            roi_max_absolute_error=float(error.max()),roi_mean_absolute_error=float(error.mean()),
            roi_largest_error_equivalent_hu=float(error.max()*1400))
        if error.max()>(1.001/1400):issues.append('ct_reconstruction_exceeds_1HU_rounding_tolerance')
        rebuilt_mask,_=sample_grid(mask,start,[64]*3,True)
        rebuilt_mask=rebuilt_mask>0
        result['mask_different_voxels']=int(np.count_nonzero(rebuilt_mask!=saved_mask[0]))
        if result['mask_different_voxels']:issues.append('mask_reconstruction_mismatch')
        retained=int(rebuilt_mask.sum())
        result.update(resampled_tumor_voxels=support['total'],retained_tumor_voxels=retained,
            tumor_retention_fraction=retained/support['total'],bbox_span_xyz_mm=json.dumps(span[::-1].tolist()))
        if retained!=support['total'] or np.any(span>64):issues.append('tumor_truncated')
        if support['total']!=int(log['total_tumor_voxels']) or retained!=int(log['retained_tumor_voxels']):issues.append('logged_tumor_count_mismatch')
        big_mask,big_acquired=sample_grid(mask,start-10,[84]*3,True)
        result.update(roi_statistics(big_mask>0,big_acquired,64,10))
        result.update(intensity_statistics(roi_hu[acquired],'roi_hu'))
        result.update(intensity_statistics(roi_hu[rebuilt_mask],'tumor_hu'))
        distance=ndimage.distance_transform_edt(~(big_mask>0))[10:74,10:74,10:74]
        result.update(intensity_statistics(roi_hu[(distance>0)&(distance<=10)&acquired],'shell10_visible_hu'))
        result.update(hard_issues=';'.join(issues),error=None)
    except Exception as exc:
        result.update(error=f'{type(exc).__name__}: {exc}',hard_issues='case_failed')
    return result


def run(args):
    root=Path(args.output_dir)
    if root.exists():raise FileExistsError(f'Refusing to overwrite {root}')
    source=Path(args.process_script).read_text()
    settings={}
    for node in ast.parse(source).body:
        if isinstance(node,ast.Assign):
            for target in node.targets:
                if isinstance(target,ast.Name) and target.id in ['target_spacing','orientation']:
                    settings[target.id]=ast.literal_eval(node.value)
                if isinstance(target,ast.Tuple) and [getattr(x,'id',None) for x in target.elts]==['hu_min','hu_max']:
                    settings.update(zip(['hu_min','hu_max'],ast.literal_eval(node.value)))
    if settings!=dict(target_spacing=(1.,1.,1.),orientation='LPS',hu_min=-1000,hu_max=400):
        raise ValueError(f'Audit reconstruction does not match current source settings: {settings}')
    samples=pd.read_csv(args.cohort,dtype={'ct_id':str,'pa_id':str})
    logs=pd.read_csv(args.preprocess_log,dtype={'ct_id':str}).set_index('ct_id')
    if len(samples)!=721 or int(samples.event.sum())!=251 or not samples.ct_id.is_unique or not logs.index.is_unique:
        raise ValueError('Require original721 unique paired patients/251 events and unique preprocessing logs')
    if not samples.split.isin(range(5)).all():raise ValueError('Require the existing five outer folds')
    root.mkdir(parents=True)
    config=vars(args).copy();config['source_settings']=settings
    config['sha256']={key:hashlib.sha256(Path(path).read_bytes()).hexdigest() for key,path in
        [('cohort',args.cohort),('preprocess_script',args.process_script),('preprocess_log',args.preprocess_log)]}
    (root/'run_config.json').write_text(json.dumps(config,indent=2))
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
    jobs=[(row,logs.loc[row['ct_id']].to_dict(),args.raw_root) for row in samples.to_dict('records')]
    records=[];started=time.monotonic()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for i,result in enumerate(pool.map(audit_case,jobs),1):
            records.append(result)
            if i%25==0 or i==len(jobs):print(f'Audited {i}/{len(jobs)} | elapsed {time.monotonic()-started:.1f}s',flush=True)
    frame=pd.DataFrame(records);frame.to_csv(root/'ct_input_cases.csv',index=False)
    failures=frame[frame.error.notna()|frame.hard_issues.ne('')]
    failures.to_json(root/'ct_input_failures.json',orient='records',indent=2)
    other_rois=[]
    base=samples.assign(case_id=samples.pa_id+'|'+samples.ct_id).set_index('case_id')
    for size in [64,96,128]:
        path=Path(args.cohort).with_name(f'all_label_roi{size}.csv')
        if not path.is_file():continue
        other=pd.read_csv(path,dtype={'ct_id':str,'pa_id':str})
        other=other.assign(case_id=other.pa_id+'|'+other.ct_id).set_index('case_id')
        common=base.index.intersection(other.index)
        other_rois.append(dict(roi_size=size,patients=len(other),common_with_roi64=len(common),
            common_fold_mismatches=int((base.loc[common,'split']!=other.loc[common,'split']).sum()),
            common_event_mismatches=int((base.loc[common,'event']!=other.loc[common,'event']).sum()),
            common_time_mismatches=int((~np.isclose(base.loc[common,'time'],other.loc[common,'time'])).sum())))
    pd.DataFrame(other_rois).to_csv(root/'roi_cohort_alignment.csv',index=False)
    numeric={}
    for key in ['tumor_fraction','minimum_margin_mm','shell5_coverage','shell10_coverage','acquired_roi_fraction',
                'tumor_hu_std','tumor_hu_below_window','tumor_hu_above_window','roi_max_absolute_error']:
        x=frame[key].dropna()
        numeric[key]=dict(min=float(x.min()),p25=float(x.quantile(.25)),median=float(x.median()),
                          p75=float(x.quantile(.75)),max=float(x.max()),mean=float(x.mean()))
    summary=dict(patients=len(frame),events=int(samples.event.sum()),failed_cases=len(failures),
        exact_ct_reconstructions=int(frame.roi_exact_match.fillna(False).sum()),
        masks_with_any_voxel_difference=int(frame.mask_different_voxels.fillna(-1).ne(0).sum()),
        all_tumors_retained=bool(frame.tumor_retention_fraction.eq(1).all()),
        tumor_touches_roi_face=int(frame.tumor_touches_roi_face.fillna(False).sum()),
        minimum_margin_below5mm=int(frame.minimum_margin_mm.lt(5).sum()),
        minimum_margin_below10mm=int(frame.minimum_margin_mm.lt(10).sum()),
        complete_shell5_cases=int(frame.shell5_coverage.ge(1-1e-12).sum()),
        complete_shell10_cases=int(frame.shell10_coverage.ge(1-1e-12).sum()),
        original_file_geometry_mismatches=int((~frame.raw_geometry_equal.fillna(False)).sum()),
        scanner_metadata_available={key:int(frame[key].notna().sum()) for key in
            ['scanner_manufacturer','scanner_model','kvp','slice_thickness','reconstruction_kernel','tube_current']},
        intensity_quantiles='Global raw quantiles use deterministic regular sampling<=200000 voxels; ROI/tumor/shell exact.',
        caveat='Current NIfTI spacing does not establish original acquisition resolution or original DICOM HU calibration.',
        available_fov_definition='acquired_* fields refer only to available current NIfTI image support, not verified original DICOM scan FOV.',
        physical_roi_edge_extent_mm=[64.,64.,64.],statistics=numeric,roi_comparisons=other_rois,
        elapsed_seconds=time.monotonic()-started)
    (root/'summary.json').write_text(json.dumps(summary,indent=2))
    for key,path in [('cohort',args.cohort),('preprocess_script',args.process_script),('preprocess_log',args.preprocess_log)]:
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=config['sha256'][key]:raise ValueError(f'Source changed during audit: {key}')
    (root/'completed.json').write_text(json.dumps(dict(complete=True,passed=len(failures)==0,patients=len(frame),
        read_only_sources=True,source_hashes_unchanged=True,
        source_hash_scope=['cohort_csv','preprocessing_script','preprocessing_log']),indent=2))
    print(json.dumps(summary,indent=2),flush=True)
    if len(failures):raise SystemExit('CT input audit found reconstruction/data errors; inspect local failure table before training.')


if __name__=='__main__':
    experiment=Path(__file__).resolve().parent
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cohort',default=str(experiment.parents[1]/'data/eagle/top25/all_label_roi64.csv'))
    parser.add_argument('--raw_root',default='/home/gly001/cqj/data/lung_cancer/ct')
    parser.add_argument('--preprocess_log',default='/home/gly001/cqj/data/lung_cancer/ct/processed_ct_64/preprocess_log.csv')
    parser.add_argument('--process_script',default='/home/gly001/cqj/data/lung_cancer/process_ct.py')
    parser.add_argument('--output_dir',default=str(experiment/'results/ct_input_audit'))
    parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args()
    if args.workers<1:parser.error('workers must be positive')
    run(args)
