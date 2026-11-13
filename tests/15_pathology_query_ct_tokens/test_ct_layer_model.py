import sys
import unittest
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))

import torch


class CTLayerModelTests(unittest.TestCase):
    def test_layer4_exactly_matches_original_weights_forward_and_rng(self):
        from ct_layer_model import CTLayerModel
        from model.build import CT_Model
        torch.manual_seed(123);reference=CT_Model(dropout=.25).eval();rng=torch.get_rng_state()
        torch.manual_seed(123);model=CTLayerModel('layer4',dropout=.25).eval()
        torch.testing.assert_close(torch.get_rng_state(),rng,rtol=0,atol=0)
        for key,value in reference.state_dict().items():
            torch.testing.assert_close(value,model.state_dict()[key],rtol=0,atol=0)
        x=torch.rand(2,1,16,16,16)
        torch.testing.assert_close(model(x),reference(x),rtol=0,atol=0)

    def test_feature_dimensions_pooling_order_and_no_layer4_for_layer3(self):
        from ct_layer_model import CTLayerModel
        torch.manual_seed(123);reference=CTLayerModel('layer4',dropout=.25)
        rng=torch.get_rng_state()
        for source,dimension in [('layer3',256),('layer34',768)]:
            torch.manual_seed(123);model=CTLayerModel(source,dropout=.25).eval()
            torch.testing.assert_close(torch.get_rng_state(),rng,rtol=0,atol=0)
            self.assertEqual(model.ct.feature_dim,dimension)
            self.assertEqual(model.ct.fc[0].out_features,256)
            self.assertEqual(model.ct.fc[2].p,.25)
            saved={}
            model.ct.layer3.register_forward_hook(lambda m,a,o:saved.update(layer3=o))
            model.ct.layer4.register_forward_hook(lambda m,a,o:saved.update(layer4=o))
            x=torch.rand(1,1,64,64,64)
            feature=model.ct.extract_features(x)
            self.assertEqual(feature.shape,(1,dimension))
            self.assertEqual(tuple(saved['layer3'].shape),(1,256,8,8,8))
            torch.testing.assert_close(feature[:,:256],model.ct.avgpool(saved['layer3']).flatten(1),rtol=0,atol=0)
            if source=='layer3':
                self.assertIsInstance(model.ct.layer4,torch.nn.Identity)
                self.assertNotIn('layer4',saved)
            else:
                self.assertEqual(tuple(saved['layer4'].shape),(1,512,8,8,8))
                torch.testing.assert_close(feature[:,256:],model.ct.avgpool(saved['layer4']).flatten(1),rtol=0,atol=0)

    def test_gradient_bn_and_optimizer_groups(self):
        from ct_layer_model import CTLayerModel
        from utils import build_ct_optimizer
        from loss import cox_loss
        for source in ['layer3','layer34']:
            model=CTLayerModel(source,dropout=.25).train()
            before={k:v.clone() for k,v in model.ct.named_buffers()}
            optimizer=build_ct_optimizer(model,1e-4,5e-4,1e-5)
            groups={id(p):g['lr'] for g in optimizer.param_groups for p in g['params']}
            self.assertEqual(groups[id(model.ct.conv1.weight)],1e-5)
            self.assertEqual(groups[id(model.ct.fc[0].weight)],1e-4)
            self.assertTrue(all(p.requires_grad for p in model.parameters()))
            output=model(torch.rand(3,1,16,16,16))
            self.assertEqual(output.shape,(3,))
            cox_loss(output,torch.tensor([1.,2.,3.]),torch.tensor([1,1,0])).backward()
            self.assertGreater(model.ct.conv1.weight.grad.abs().sum(),0)
            for key,value in model.ct.named_buffers():
                torch.testing.assert_close(value,before[key],rtol=0,atol=0)


if __name__=='__main__':
    torch.set_num_threads(2)
    unittest.main()
