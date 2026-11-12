"""Read-only numerical/gradient diagnosis of the original test15 attention."""

import argparse
import json
import math
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from sksurv.metrics import concordance_index_censored

from dataset import Pa_CT_Dataset, build_dataloader, cv_fold_indices, cox_batch_sizes, load_path_feature
from inference_checks import cache_fusion_inputs, digest
from loss import cox_loss
from pact_train import build_pact_optimizer
from path_query_ct import PathQueryCTModel
from utils import seed_everything


def attention_parts(attention, query, context):
    """Expose the actual single-head Q/K/V and standard scaled dot products."""
    if attention.num_heads != 1 or attention.dropout != 0:
        raise ValueError('This diagnosis expects test15 single-head, zero-dropout MHA')
    d = attention.embed_dim
    w, b = attention.in_proj_weight, attention.in_proj_bias
    q = F.linear(query, w[:d], b[:d])
    k = F.linear(context, w[d:2*d], b[d:2*d])
    v = F.linear(context, w[2*d:], b[2*d:])
    logits = q @ k.transpose(-2, -1) / math.sqrt(d)
    weights = logits.softmax(-1)
    retrieved = attention.out_proj(weights @ v)
    return dict(q=q, k=k, v=v, logits=logits, weights=weights, retrieved=retrieved)


def feature_stats(x):
    x = x.detach().double()
    rms = x.square().mean().sqrt()
    within = (x - x.mean(1, keepdim=True)).square().mean().sqrt()
    u = F.normalize(x, dim=-1)
    n = x.size(1)
    cosine = ((u.sum(1).square().sum(-1) - u.square().sum((-2, -1))) / (n * (n - 1))).mean()
    return dict(norm_mean=x.norm(dim=-1).mean().item(), rms=rms.item(),
                within_token_rms=within.item(), within_token_relative_rms=(within / rms.clamp_min(1e-30)).item(),
                mean_pairwise_token_cosine=cosine.item())


def parameter_parts(model):
    for name, parameter in model.named_parameters():
        if name == 'fusion.coattn.in_proj_weight':
            for i, label in enumerate(['Wq', 'Wk', 'Wv']):
                yield label, parameter[i*256:(i+1)*256], (
                    parameter.grad[i*256:(i+1)*256] if parameter.grad is not None else None)
        elif name.endswith('weight') and (name.startswith('fusion.') or name in [
                'pa_branch.compression.0.weight', 'pa_branch.post_compression.0.weight', 'ct_backbone.conv1.weight']):
            yield name, parameter, parameter.grad


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run_dir', default=None)
    parser.add_argument('--output_dir', default=str(here / 'score_diagnosis'))
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    args = parser.parse_args()
    candidates = [p.parent for p in (here / 'results').glob('*/run_config.yaml')
                  if yaml.safe_load(p.read_text()).get('fusion_type') == 'path_query_ct_layer4'
                  and not yaml.safe_load(p.read_text()).get('alignment_weight', 0)
                  and yaml.safe_load(p.read_text()).get('pa_l2_normalize', True)
                  and not yaml.safe_load(p.read_text()).get('qk_norm', False)
                  and not yaml.safe_load(p.read_text()).get('ct_zscore', False)
                  and not yaml.safe_load(p.read_text()).get('ct_bn_adapt', False)
                  and yaml.safe_load(p.read_text()).get('fusion_control', 'crossattention') == 'crossattention'
                  and not yaml.safe_load(p.read_text()).get('all_patch_queries', False)]
    if args.run_dir is None and len(candidates) != 1:
        raise ValueError('Specify the original no-alignment run explicitly')
    run = Path(args.run_dir) if args.run_dir else candidates[0]
    config = yaml.safe_load((run / 'run_config.yaml').read_text())
    if (config.get('alignment_weight', 0) or config['lambda_ct'] or config['lambda_pa']
            or not config.get('pa_l2_normalize', True) or config.get('all_patch_queries', False)
            or config.get('qk_norm', False) or config.get('ct_zscore', False)
            or config.get('fusion_control', 'crossattention') != 'crossattention'
            or config.get('ct_bn_adapt', False)):
        raise ValueError('Only the original fused-Cox test15 run is supported')
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    torch.set_num_threads(2)
    seed_everything(config['seed'])
    initial_model = PathQueryCTModel(ct_pretrained_path=config['ct_pretrained_path'], dropout=config['dropout'])
    initial = {k: v.detach().clone() for k, v in initial_model.state_dict().items()}
    del initial_model
    dataset = Pa_CT_Dataset(config['data_dir'], roi_size=64, feature_dim=1280)
    destination = Path(args.output_dir)
    destination.mkdir(parents=True, exist_ok=False)
    log = (here / 'logs' / f'{run.name}.log').read_text()
    blocks = re.split(r'Fold (\d)/5\n=+\n', log)
    best_epochs = {}
    for i in range(1, len(blocks), 2):
        values = [(int(e), float(c)) for e, c in re.findall(
            r'Epoch (\d+)/\d+ .*?Val C-index: ([\d.]+)', blocks[i+1])]
        best_epochs[int(blocks[i])-1] = max(values, key=lambda x: x[1])[0]
    (destination / 'methods.json').write_text(json.dumps(dict(
        run=str(run), seed=config['seed'], validation='all five validation folds',
        gradient_probe='two real training risk sets of 32 patients/fold; training-mode dropout; no optimizer steps',
        decay_control='same initialization/Adam/lr/wd; zero data gradient; 18 updates per epoch',
        best_epochs_from_rounded_log=best_epochs,
        limitations='current-checkpoint gradients do not reconstruct past gradients; counterfactuals are not retrained models'), indent=2) + '\n')
    feature_rows, score_rows, gradient_rows, parameter_rows, uniform_rows = [], [], [], [], []
    for fold in range(5):
        print(f'Fold {fold}: original teacher only', flush=True)
        checkpoint = Path(config['checkpoint_root']) / f'fold_{fold}/best_model.pth'
        checkpoint_sha = digest(checkpoint)
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        model = PathQueryCTModel(dropout=config['dropout']).to(device).eval()
        model.load_state_dict(state, strict=True)
        optimizer = build_pact_optimizer(model, config['lr'], config['weight_decay'], config['ct_backbone_lr'])
        qkv = model.fusion.coattn.in_proj_weight
        groups = [g for g in optimizer.param_groups if any(p is qkv for p in g['params'])]
        assert len(groups) == 1 and groups[0]['lr'] == config['lr'] and qkv.requires_grad
        train_indices, val_indices = cv_fold_indices(dataset.samples, fold)
        features = cache_fusion_inputs(model, build_dataloader(dataset, indices=val_indices, num_workers=0), device)
        ids = features.pop('case_id')
        features = {k: v.to(device) for k, v in features.items()}
        raw_pa = torch.stack([load_path_feature(dataset.samples.iloc[i].pa_path, 1280) for i in val_indices]).to(device)
        with torch.inference_mode():
            query = model.fusion.pa_projection(features['pa_patches'])
            context = model.fusion.ct_projection(features['ct_tokens'])
            parts = attention_parts(model.fusion.coattn, query, context)
            retrieved, attention = model.fusion.coattn(query, context, context)
            torch.testing.assert_close(parts['weights'], attention, rtol=1e-4, atol=1e-6)
            torch.testing.assert_close(parts['retrieved'], retrieved, rtol=1e-4, atol=1e-6)
            for name, tensor in dict(raw_pathology=raw_pa, normalized_pathology=F.normalize(raw_pa, dim=-1),
                    compressed_pathology=features['pa_patches'], query_input=query,
                    ct_spatial=features['ct_tokens'], context_input=context,
                    internal_q=parts['q'], internal_k=parts['k'], internal_v=parts['v'],
                    spatial_centered_k=parts['k']-parts['k'].mean(1, keepdim=True)).items():
                feature_rows.append(dict(fold=fold, stage=name, **feature_stats(tensor)))
            logits = parts['logits']
            q0 = F.linear(query, initial['fusion.coattn.in_proj_weight'][:256].to(device),
                           initial['fusion.coattn.in_proj_bias'][:256].to(device))
            k0 = F.linear(context, initial['fusion.coattn.in_proj_weight'][256:512].to(device),
                           initial['fusion.coattn.in_proj_bias'][256:512].to(device))
            no_input_l2_query = model.fusion.pa_projection(model.pa_branch.compression(raw_pa))
            layernorm_parts = attention_parts(model.fusion.coattn, F.layer_norm(query, (256,)), F.layer_norm(context, (256,)))
            no_l2_parts = attention_parts(model.fusion.coattn, no_input_l2_query, context)
            score_rows.append(dict(fold=fold,
                normalized_entropy=(-attention.clamp_min(1e-30)*attention.clamp_min(1e-30).log()).sum(-1).mean().item()/math.log(512),
                spatial_logit_std=logits.std(-1, unbiased=False).mean().item(),
                unscaled_dot_product_std=logits.std(-1, unbiased=False).mean().item()*16,
                initial_q_projection_std=(q0@parts['k'].transpose(-2,-1)/16).std(-1,unbiased=False).mean().item(),
                initial_k_projection_std=(parts['q']@k0.transpose(-2,-1)/16).std(-1,unbiased=False).mean().item(),
                initial_qk_projection_std=(q0@k0.transpose(-2,-1)/16).std(-1,unbiased=False).mean().item(),
                no_patch_l2_counterfactual_std=no_l2_parts['logits'].std(-1,unbiased=False).mean().item(),
                pre_layernorm_counterfactual_std=layernorm_parts['logits'].std(-1,unbiased=False).mean().item(),
                manual_weights_max_error=(parts['weights']-attention).abs().max().item()))
            normal = model.fusion(features['pa_patches'], features['ct_tokens'], features['pa_patient'])
            uniform_ct = model.fusion.coattn.out_proj(parts['v'].mean(1))
            uniform_risk = model.fusion.head(uniform_ct, features['pa_patient'])
            np.testing.assert_allclose(normal[0].cpu(), features['risk'].cpu(), rtol=1e-4, atol=2e-5)
            reference = pd.read_csv(run / f'fold_{fold}/val_predictions.csv', dtype={'case_id': str}).set_index('case_id').loc[ids]
            np.testing.assert_allclose(normal[0].cpu(), reference.risk_score, rtol=1e-4, atol=2e-5)
            ci = lambda risk: float(concordance_index_censored(reference['dfs.status'].astype(bool), reference['dfs.month'], risk.cpu().numpy())[0])
            uniform_rows.append(dict(fold=fold, normal_cindex=ci(normal[0]), uniform_cindex=ci(uniform_risk),
                risk_rmse_over_baseline_sd=((normal[0]-uniform_risk).square().mean().sqrt()/normal[0].std(unbiased=False)).item()))
            print('  Scores:', score_rows[-1], flush=True)
            print('  Uniform-value control:', uniform_rows[-1], flush=True)
        # Isolate accumulated coupled-Adam weight decay from the supervised data gradient.
        n_steps = len(cox_batch_sizes(len(train_indices), config['cox_batch_size'])) * best_epochs[fold]
        names = ['fusion.coattn.in_proj_weight', 'fusion.pa_projection.0.weight', 'fusion.ct_projection.0.weight']
        decay_parameters = [torch.nn.Parameter(initial[n].clone()) for n in names]
        decay_optimizer = torch.optim.Adam(decay_parameters, lr=config['lr'], weight_decay=config['weight_decay'])
        for _ in range(n_steps):
            for parameter in decay_parameters:
                parameter.grad = torch.zeros_like(parameter)
            decay_optimizer.step()
        for name, decayed in zip(names, decay_parameters):
            slices = [(label, slice(i*256, (i+1)*256)) for i, label in enumerate(['Wq', 'Wk', 'Wv'])] if 'in_proj' in name else [(name, slice(None))]
            for label, index in slices:
                actual, init, decay = state[name][index], initial[name][index], decayed.detach()[index]
                parameter_rows.append(dict(fold=fold, parameter=label, updates=n_steps,
                    initial_norm=init.norm().item(), trained_norm=actual.norm().item(),
                    trained_over_initial_norm=(actual.norm()/init.norm()).item(),
                    decay_only_norm=decay.norm().item(), decay_only_relative_error=((actual-decay).norm()/actual.norm()).item()))
        # Actual training-mode forward/backward, retaining all patients in each Cox risk set.
        snapshot = {k: v.detach().clone() for k, v in model.state_dict().items()}
        rng = np.random.default_rng(np.random.SeedSequence([config['seed'], fold, 73]))
        selected = rng.choice(train_indices, 64, replace=False)
        for group in range(2):
            seed_everything(config['seed'] + 100*fold + group)
            model.train()
            model.zero_grad(set_to_none=True)
            train_scores, events, times, train_logit_stds = [], [], [], []
            def capture(module, inputs):
                with torch.no_grad():
                    p = attention_parts(module, inputs[0], inputs[1])
                    train_logit_stds.append(p['logits'].std(-1, unbiased=False).mean().item())
            hook = model.fusion.coattn.register_forward_pre_hook(capture)
            try:
                for ct, pa, event, time, _ in build_dataloader(dataset, indices=selected[group*32:(group+1)*32], num_workers=0):
                    train_scores.append(model(ct.to(device), pa.to(device))[0])
                    events.append(event.to(device)); times.append(time.to(device))
                loss = cox_loss(torch.cat(train_scores), torch.cat(times), torch.cat(events))
                loss.backward()
            finally:
                hook.remove()
            for label, parameter, gradient in parameter_parts(model):
                if gradient is None:
                    continue
                weight_norm, gradient_norm = parameter.detach().norm().item(), gradient.detach().norm().item()
                gradient_rows.append(dict(fold=fold, group=group, parameter=label, cox_loss=loss.item(),
                    events=int(torch.cat(events).sum()), train_spatial_logit_std=float(np.mean(train_logit_stds)),
                    weight_norm=weight_norm, gradient_norm=gradient_norm,
                    gradient_rms=gradient_norm/math.sqrt(parameter.numel()),
                    l2_gradient_norm=config['weight_decay']*weight_norm,
                    data_grad_over_l2_grad=gradient_norm/max(config['weight_decay']*weight_norm, 1e-30)))
            print('  Training gradient probe', group, [(r['parameter'], r['gradient_norm'], r['data_grad_over_l2_grad'])
                  for r in gradient_rows if r['fold']==fold and r['group']==group and r['parameter'] in ['Wq','Wk','Wv']], flush=True)
            model.zero_grad(set_to_none=True)
            del train_scores, events, times, loss
        for name, value in model.state_dict().items():
            assert torch.equal(value, snapshot[name]), f'Unexpected state mutation: {name}'
        assert digest(checkpoint) == checkpoint_sha
        for filename, records in [('feature_metrics.csv', feature_rows), ('score_metrics.csv', score_rows),
                ('gradient_metrics.csv', gradient_rows), ('parameter_metrics.csv', parameter_rows), ('uniform_control.csv', uniform_rows)]:
            pd.DataFrame(records).to_csv(destination / filename, index=False)
        del model, optimizer, features, snapshot, raw_pa, parts, query, context, retrieved, attention
        torch.cuda.empty_cache() if device.type == 'cuda' else None
    (destination / 'completed.json').write_text(json.dumps(dict(status='complete', folds=5,
        training_gradient_groups_per_fold=2, optimizer_steps_on_trained_models=0), indent=2) + '\n')
    print('DIAGNOSIS COMPLETE', destination, flush=True)


if __name__ == '__main__':
    main()
