import unittest

import numpy as np
import SimpleITK as sitk


class CTInputAuditTests(unittest.TestCase):
    def test_roi_grid_preserves_physical_direction_and_air_padding(self):
        from ct_input_audit import sample_grid
        array=(np.arange(15*16*17).reshape(15,16,17)-900).astype(np.int16)
        image=sitk.GetImageFromArray(array);image.SetOrigin((10.,20.,30.))
        image.SetDirection((0.,-1.,0.,1.,0.,0.,0.,0.,1.))
        sampled,valid=sample_grid(image,[-1,3,4],[8,8,8])
        np.testing.assert_array_equal(sampled[:,:,1:],array[4:12,3:11,:7])
        np.testing.assert_array_equal(sampled[:,:,0],np.full((8,8),-1000))
        self.assertFalse(valid[:,:,0].any())
        self.assertTrue(valid[:,:,1:].all())

    def test_anisotropic_direct_roi_matches_full_resampling(self):
        from ct_input_audit import sample_grid
        rng=np.random.default_rng(123)
        array=rng.integers(-1100,800,size=(20,24,26),dtype=np.int16)
        image=sitk.GetImageFromArray(array);image.SetSpacing((.7,.7,1.5));image.SetOrigin((.3,-.2,2.1))
        size=[round(n*s) for n,s in zip(image.GetSize(),image.GetSpacing())]
        full=sitk.Resample(image,size,sitk.Transform(),sitk.sitkLinear,image.GetOrigin(),
            (1.,1.,1.),image.GetDirection(),-1000,image.GetPixelID())
        array_full=sitk.GetArrayFromImage(full)
        roi,_=sample_grid(image,[2,3,4],[10,10,10])
        self.assertLessEqual(np.max(np.abs(roi.astype(float)-array_full[4:14,3:13,2:12])),1.)

    def test_all_component_support_equals_full_nearest_resample(self):
        from ct_input_audit import resampled_tumor
        array=np.zeros((20,24,26),dtype=np.uint8);array[2:5,3:7,4:9]=1;array[12,18,20]=1
        mask=sitk.GetImageFromArray(array);mask.SetSpacing((.7,.7,1.5))
        size=[round(n*s) for n,s in zip(mask.GetSize(),mask.GetSpacing())]
        full=sitk.GetArrayFromImage(sitk.Resample(mask,size,sitk.Transform(),sitk.sitkNearestNeighbor,
            mask.GetOrigin(),(1.,1.,1.),mask.GetDirection(),0,mask.GetPixelID()))
        result=resampled_tumor(mask);positions=np.argwhere(full)
        self.assertEqual(result['total'],int(full.sum()))
        np.testing.assert_array_equal(result['min_zyx'],positions.min(0))
        np.testing.assert_array_equal(result['max_zyx'],positions.max(0))

    def test_full_tumor_retention_does_not_imply_full_peritumoral_shell(self):
        from ct_input_audit import roi_statistics
        size,halo=32,10
        mask=np.zeros((size+2*halo,)*3,dtype=bool)
        mask[halo+size//2,halo+size//2,halo]=True
        result=roi_statistics(mask,np.ones_like(mask),size,halo)
        self.assertEqual(result['tumor_voxels'],1)
        self.assertEqual(result['minimum_margin_mm'],0.)
        self.assertTrue(result['tumor_touches_roi_face'])
        self.assertLess(result['shell10_coverage'],1.)
        mask[:]=False;mask[halo+size//2,halo+size//2,halo+size//2]=True
        central=roi_statistics(mask,np.ones_like(mask),size,halo)
        self.assertEqual(central['shell10_coverage'],1.)


if __name__=='__main__':
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
    unittest.main()
