"""Small predefined mask-assisted phenotypes from the existing64^3 CT input."""

import ast
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from radiomics import featureextractor
from scipy import ndimage


def feature_groups():
    first=['mean','std','p10','median','p90','entropy']
    texture=['glcm_contrast','glcm_idm','glcm_correlation','glrlm_sre','glrlm_lre','glrlm_glnu_normalized']
    groups=dict(Shape=['log_volume_mm3','diameter_mm','sphericity'],
        FirstOrder=['tumor_'+x for x in first],Texture=['tumor_'+x for x in texture],
        Peritumoral=['peri5_'+x for x in first+texture])
    groups['Combined']=sum(groups.values(),[])
    return groups


def recover_intensity(stored):
    x=np.asarray(stored,dtype=np.float64)
    if not np.isfinite(x).all() or x.min()<0 or x.max()>1:raise ValueError('Invalid normalized CT input')
    inverse=x*1400-1000;rounded=np.rint(inverse)
    if np.max(np.abs(inverse-rounded))>.001:raise ValueError('Stored input is not the audited clipped-integer HU grid')
    recovered=rounded.astype(np.int16)
    np.testing.assert_allclose((recovered.astype(np.float32)+1000)/1400,x,rtol=0,atol=1e-7)
    return recovered


def roi_support(size_xyz,spacing_xyz,direction,center_zyx):
    # Use header geometry only. A1-voxel dummy determines the SAME LPS permutation
    # as source preprocessing without loading any pixels outside the model ROI.
    dummy=sitk.Image([1]*3,sitk.sitkUInt8)
    dummy.SetSpacing(tuple(spacing_xyz));dummy.SetDirection(tuple(direction))
    orient=sitk.DICOMOrientImageFilter();orient.SetDesiredCoordinateOrientation('LPS');orient.Execute(dummy)
    order=list(orient.GetPermuteOrder())
    size=np.asarray(size_xyz)[order];spacing=np.asarray(spacing_xyz)[order]
    target_size=np.rint(size*spacing).astype(int);start=np.asarray(center_zyx)[::-1]-32
    z,y,x=np.ogrid[start[2]:start[2]+64,start[1]:start[1]+64,start[0]:start[0]+64]
    valid=(x>=0)&(x<target_size[0])&(y>=0)&(y<target_size[1])&(z>=0)&(z<target_size[2])
    valid=valid&(x/spacing[0]>=-.5)&(x/spacing[0]<size[0]-.5)
    valid=valid&(y/spacing[1]>=-.5)&(y/spacing[1]<size[1]-.5)
    return valid&(z/spacing[2]>=-.5)&(z/spacing[2]<size[2]-.5)


def peritumor_shell(tumor,available):
    tumor,available=np.asarray(tumor,dtype=bool),np.asarray(available,dtype=bool)
    if tumor.shape!=(64,64,64) or available.shape!=tumor.shape or not tumor.any():
        raise ValueError('Require nonempty tumor and available-data masks on64^3 grid')
    expanded=np.pad(tumor,5,constant_values=False)
    distance=ndimage.distance_transform_edt(~expanded)
    full=(distance>0)&(distance<=5)
    observed=full[5:69,5:69,5:69]&available
    if not observed.any():raise ValueError('No observed5mm peritumoral region')
    info=dict(coverage=float(observed.sum()/full.sum()),observed_voxels=int(observed.sum()),full_voxels=int(full.sum()))
    return observed,info


def region_features(values,mask):
    image=sitk.GetImageFromArray(np.asarray(values,dtype=np.float32))
    labels=sitk.GetImageFromArray(np.asarray(mask,dtype=np.uint8))
    extractor=featureextractor.RadiomicsFeatureExtractor(binWidth=25,normalize=False,
        resampledPixelSpacing=None,force2D=False,distances=[1],symmetricalGLCM=True,
        weightingNorm=None,voxelArrayShift=0,additionalInfo=False,label=1)
    extractor.disableAllFeatures()
    extractor.enableFeaturesByName(firstorder=['Mean','Variance','10Percentile','Median','90Percentile','Entropy'],
        glcm=['Contrast','Idm','Correlation'],
        glrlm=['ShortRunEmphasis','LongRunEmphasis','GrayLevelNonUniformityNormalized'])
    extracted=extractor.execute(image,labels)
    keys=dict(mean='firstorder_Mean',std='firstorder_Variance',p10='firstorder_10Percentile',
        median='firstorder_Median',p90='firstorder_90Percentile',entropy='firstorder_Entropy',
        glcm_contrast='glcm_Contrast',glcm_idm='glcm_Idm',glcm_correlation='glcm_Correlation',
        glrlm_sre='glrlm_ShortRunEmphasis',glrlm_lre='glrlm_LongRunEmphasis',
        glrlm_glnu_normalized='glrlm_GrayLevelNonUniformityNormalized')
    result={key:float(np.asarray(extracted['original_'+value]).item()) for key,value in keys.items()}
    if result['std']<0:raise ValueError('Negative radiomic variance')
    result['std']=float(np.sqrt(result['std']))
    if not np.isfinite(list(result.values())).all():raise ValueError('Nonfinite predefined radiomic feature')
    return result


def extract_case(job):
    row,log,audit=job
    case=str(row['pa_id'])+'|'+str(row['ct_id'])
    try:
        if Path(row['ct_path']).resolve()!=Path(log['processed_image_path']).resolve():
            raise ValueError('ROI/log identity mismatch')
        stored=np.load(row['ct_path']);mask=np.load(log['processed_mask_path'])
        if stored.shape!=(1,64,64,64) or mask.shape!=stored.shape:raise ValueError('Expected saved CT and mask [1,64,64,64]')
        if not np.isin(np.unique(mask),[0,1]).all():raise ValueError('Nonbinary ROI mask')
        values=recover_intensity(stored[0]);tumor=mask[0].astype(bool)
        reader=sitk.ImageFileReader();reader.SetFileName(log['image_path']);reader.ReadImageInformation()
        available=roi_support(reader.GetSize(),reader.GetSpacing(),reader.GetDirection(),ast.literal_eval(log['tumor_center_zyx']))
        np.testing.assert_allclose(available.mean(),audit['acquired_roi_fraction'],rtol=0,atol=1e-12)
        shell,coverage=peritumor_shell(tumor,available)
        np.testing.assert_allclose(coverage['coverage'],audit['shell5_coverage'],rtol=0,atol=1e-12)
        if int(tumor.sum())!=int(audit['retained_tumor_voxels']):raise ValueError('Saved mask changed since audit')
        result={**{'tumor_'+k:v for k,v in region_features(values,tumor).items()},
                **{'peri5_'+k:v for k,v in region_features(values,shell).items()}}
        return dict(case_id=case,features=result,coverage=dict(peri5_coverage=coverage['coverage'],
            peri5_observed_voxels=coverage['observed_voxels'],peri5_full_voxels=coverage['full_voxels'],
            available_roi_fraction=float(available.mean()),tumor_voxels=int(tumor.sum())),error=None)
    except Exception as exc:
        return dict(case_id=case,error=f'{type(exc).__name__}: {exc}')
