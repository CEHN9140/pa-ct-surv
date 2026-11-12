"""Five-fold teacher: Top25 patient pathology, Top25/full queries, CT layer4 KV."""

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import yaml

from cox_utils import evaluate_survival
from dataset import Pa_CT_Dataset, build_dataloader, cv_splits, resolve_data_dir
from pact_train import build_pact_optimizer, train_pact
from utils import save_cv_results, save_run_config, seed_everything
from path_query_ct import PathQueryCTModel
from train_alignment import train_pact_alignment
from all_query_dataset import AllPatchQueryDataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data_dir', default=str(Path(__file__).resolve().parents[2] / 'data/eagle/top25'))
    parser.add_argument('--ct_pretrained_path', default=str(Path(__file__).resolve().parents[2] / 'model/ct_pretrain/resnet_18_23dataset.pth'))
    parser.add_argument('--num_epochs', type=int, default=30)
    parser.add_argument('--cox_batch_size', type=int, default=32)
    parser.add_argument('--dropout', type=float, default=.25)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--ct_backbone_lr', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--no_pa_l2_normalize', dest='pa_l2_normalize', action='store_false', default=True,
                        help='Use raw pathology patches in both ABMIL and the query route.')
    parser.add_argument('--all_patch_queries', action='store_true',
                        help='Keep Top25 patient pooling; use all same-grid Virchow2 patches as queries.')
    parser.add_argument('--qk_norm', action='store_true',
                        help='Top25 control: L2-normalize internal Q/K and learn the cosine score scale.')
    parser.add_argument('--fusion_control', choices=['crossattention', 'pathology_only', 'uniform_ct'],
                        default='crossattention', help='Matched zero-CT or uniform-CT fusion ablation.')
    ct_control = parser.add_mutually_exclusive_group()
    ct_control.add_argument('--ct_zscore', action='store_true',
                            help='Per-case whole-ROI population Z-score; keep BN fixed.')
    ct_control.add_argument('--ct_bn_adapt', action='store_true',
                            help='Keep original CT input; update BN only during training.')
    parser.add_argument('--all_feature_dir', default='/home/gly001/cqj/data/lung_cancer/pathology/virchow/virchow2',
                        help='Full Virchow2 H5 directory; used only with --all_patch_queries.')
    parser.add_argument('--alignment_weight', type=float, default=1.,
                        help='PAMT alignment coefficient; 1 matches its training entry, 0 disables it.')
    parser.add_argument('--alignment_topk', type=int, default=2,
                        help='Pseudo-positive CT positions per pathology query; PAMT default is 2.')
    parser.add_argument('--checkpoint_root', default=None)
    parser.add_argument('--results_root', default=None)
    args = parser.parse_args()
    if args.num_epochs < 1 or args.cox_batch_size < 1 or not 0 <= args.dropout < 1:
        parser.error('epochs and Cox batch size must be positive; dropout must be in [0,1)')
    if not math.isfinite(args.alignment_weight) or args.alignment_weight < 0:
        parser.error('alignment_weight must be finite and nonnegative')
    if not 1 <= args.alignment_topk <= 512:
        parser.error('alignment_topk must be between 1 and 512')
    if (args.fusion_control != 'crossattention'
            and (args.alignment_weight != 0 or not args.pa_l2_normalize or args.all_patch_queries
                 or args.qk_norm or args.ct_zscore or args.ct_bn_adapt)):
        parser.error('Fusion controls require Top25, patch L2, --alignment_weight 0 and original CT preprocessing/BN')
    if args.all_patch_queries and args.alignment_weight != 0:
        parser.error('The all-query control uses fused Cox only; set --alignment_weight 0')
    if ((args.qk_norm or args.ct_zscore or args.ct_bn_adapt)
            and (args.all_patch_queries or args.alignment_weight != 0 or not args.pa_l2_normalize)):
        parser.error('These attention/CT controls require Top25 queries, patch L2 and --alignment_weight 0')
    args.ct_roi_size, args.pa_feature_dim = 64, 1280
    args.pa_model, args.ct_model = 'abmil', 'resnet18'
    args.fusion_type = 'path_query_ct_layer4'
    args.fusion_l2_normalize = False
    args.survival_loss, args.lambda_ct, args.lambda_pa = 'cox', 0., 0.
    args.ct_token_source, args.ct_token_grid = 'layer4', [8, 8, 8]
    args.pa_query_tokens = 'all_available' if args.all_patch_queries else 25
    args.attention_dim, args.attention_heads = 256, 1
    args.ct_input_normalization = 'per_roi_population_zscore' if args.ct_zscore else 'window_minmax_0_1'
    args.ct_bn_stats = 'update_train_fixed_eval' if args.ct_bn_adapt else 'fixed_pretrained'
    if args.ct_zscore:
        args.ct_zscore_eps = 1e-6
    if args.qk_norm:
        args.qk_norm_scale_init = math.log2(512**2 - 512)
        args.attention_score = 'scale * cosine(internal_q, internal_k); no sqrt(d) divisor'
    else:
        args.attention_score = 'internal_q @ internal_k.T / sqrt(256); no Q/K normalization'
    if args.all_patch_queries:
        args.pa_patient_tokens = 25
        args.pa_query_order = 'original Top25 ranking first; remaining source rows once, in source order'
    args.query_aggregation, args.positional_encoding = 'attention_pool', False
    args.query_pool_hidden_dim = 128
    if args.fusion_control == 'uniform_ct':
        args.attention_score = 'fixed 1/N_CT; mean internal Value followed by original out_proj'
    elif args.fusion_control == 'pathology_only':
        args.attention_score, args.query_aggregation = 'not used; zero CT fusion slot', 'not used'
        args.ct_token_source, args.ct_token_grid, args.pa_query_tokens = 'disabled', [], 0
        args.ct_input_normalization, args.ct_bn_stats = 'not used', 'not used'
    if args.alignment_weight > 0:
        args.alignment_type = 'pamt_topk_unnormalized_target_cross_entropy'
        args.alignment_source = 'PAMT/gene_wsi_predict/train.py + utils_cox.py + vit_model_gene_wsi_concat_label.py'
    args.data_dir = resolve_data_dir(args.data_dir, args.ct_roi_size)
    tag = (f'eagle_top25_virchow2_cls_{"pal2" if args.pa_l2_normalize else "paraw"}_fusionraw_roi64'
           f'_paq{"all" if args.all_patch_queries else "25"}_ctlayer4kv512'
           f'_crossattn_attnpool_concat_dropout{args.dropout}_epochs{args.num_epochs}'
           f'_coxbs{args.cox_batch_size}_lr{args.lr:g}_blr{args.ct_backbone_lr:g}_wd{args.weight_decay:g}')
    if args.alignment_weight > 0:
        tag += f'_pamtalignk{args.alignment_topk}_w{args.alignment_weight:g}'
    if args.qk_norm:
        tag += '_qknorm'
    if args.ct_zscore:
        tag += '_ctzscore'
    elif args.ct_bn_adapt:
        tag += '_ctbnadapt'
    if args.fusion_control != 'crossattention':
        tag += '_' + args.fusion_control.replace('_', '')
    experiment = Path(__file__).resolve().parent
    args.checkpoint_root = args.checkpoint_root or str(experiment / 'checkpoints' / tag)
    args.results_root = args.results_root or str(experiment / 'results' / tag)
    for path in (Path(args.checkpoint_root), Path(args.results_root)):
        if path.exists():
            raise FileExistsError(f'Refusing to overwrite existing experiment: {path}')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed_everything(args.seed)
    excluded = ([] if args.all_patch_queries else ['all_feature_dir'])
    if args.alignment_weight == 0:
        excluded.append('alignment_topk')
    if args.fusion_control == 'crossattention':
        excluded.append('fusion_control')
    elif args.fusion_control == 'pathology_only':
        excluded.extend(['ct_pretrained_path', 'ct_backbone_lr', 'attention_dim',
                         'attention_heads', 'query_pool_hidden_dim'])
    save_run_config(args, 'pact', tag, excluded=excluded)
    print(f'Using Device: {device}')
    query_count = 'N' if args.all_patch_queries else '25'
    print(f'Experiment 15 | Fusion control: {args.fusion_control}')
    print(f'Pathology patch L2: {args.pa_l2_normalize} | Fusion L2: False | Positional encoding: False')
    print(f'CT input: {args.ct_input_normalization} | CT BN statistics: {args.ct_bn_stats}')
    if args.fusion_control == 'pathology_only':
        print('A: Top25 ABMIL patient feature -> same concat head with zero CT slot; '
              'CT encoder and query/attention modules removed; no CT forward or CT gradients')
    elif args.fusion_control == 'uniform_ct':
        print('B: own paired CT layer4 [B,512,512] -> CT projection -> internal Value -> '
              'uniform mean -> original output projection -> query pool -> concat with pathology')
        print('Attention: fixed 1/512; no learned Q/K selection; CT and PA jointly train')
    elif args.qk_norm:
        print(f'Attention: QKNorm after internal Q/K projection; trainable scale init={args.qk_norm_scale_init:.6f}; '
              'Value unchanged; no sqrt(d) divisor')
    else:
        print('Attention: standard PyTorch MHA; internal QK^T/sqrt(256); no Q/K normalization')
    if args.fusion_control != 'pathology_only':
        print(f'Pathology input: Top25; query route count: {query_count}; '
              'CT summary 256 + ABMIL patient feature 256 -> 512 -> 256 -> 128 -> 1')
    if args.alignment_weight > 0:
        print(f'Loss: fused Cox + {args.alignment_weight:g} * PAMT alignment; topK={args.alignment_topk}')
        print('Alignment: projected PA [25,256] @ CT [512,256].T; raw dot products; '
              'topK pseudo-positives, original target sum=K; no extra L2/temperature/sqrt(d)')
    else:
        print('Loss: fused Cox only')
    print('CT/PA auxiliary survival losses remain disabled; Val Loss and final metrics losses report fused Cox')
    if args.fusion_control == 'pathology_only':
        print('CT risk output is an unused zero placeholder; its logged auxiliary Cox is not a CT-model metric')
    backbone_lr_label = 'not used (CT removed)' if args.fusion_control == 'pathology_only' else f'{args.ct_backbone_lr:g}'
    print(f'Backbone LR: {backbone_lr_label} | Other LR: {args.lr:g} | '
          f'Cox batch: {args.cox_batch_size} | Seed: {args.seed} | Patience: {args.patience}')
    dataset = (AllPatchQueryDataset(args.data_dir, args.all_feature_dir)
               if args.all_patch_queries else Pa_CT_Dataset(args.data_dir, roi_size=64, feature_dim=1280))
    if args.all_patch_queries:
        print(f'Full-query input audit: {dataset.audit}')
        with (Path(args.results_root) / 'query_data_audit.yaml').open('w') as handle:
            yaml.safe_dump(dataset.audit, handle)
    print(f'Loaded {len(dataset)} paired patients')
    folds = []
    for fold, (train_indices, val_indices) in enumerate(cv_splits(dataset.samples)):
        print(f"\n{'=' * 50}\nFold {fold + 1}/5\n{'=' * 50}")
        seed_everything(args.seed)
        train_loader = build_dataloader(dataset, indices=train_indices, batch_size=1,
                                        shuffle=True, num_workers=args.num_workers)
        val_loader = build_dataloader(dataset, indices=val_indices, batch_size=1,
                                      shuffle=False, num_workers=args.num_workers)
        model = PathQueryCTModel(ct_pretrained_path=args.ct_pretrained_path, dropout=args.dropout,
                                 alignment_topk=args.alignment_topk if args.alignment_weight > 0 else None,
                                 pa_l2_normalize=args.pa_l2_normalize,
                                 all_patch_queries=args.all_patch_queries,
                                 qk_norm=args.qk_norm, ct_zscore=args.ct_zscore,
                                 ct_bn_adapt=args.ct_bn_adapt, fusion_control=args.fusion_control).to(device)
        optimizer = build_pact_optimizer(model, args.lr, args.weight_decay, args.ct_backbone_lr)
        checkpoint = Path(args.checkpoint_root) / f'fold_{fold}'
        checkpoint.mkdir(parents=True)
        trainer = train_pact_alignment if args.alignment_weight > 0 else train_pact
        model = trainer(model, train_loader, val_loader, optimizer, args, device, fold, checkpoint)
        metrics = evaluate_survival(model, train_loader, val_loader, device,
                                    save_dir=Path(args.results_root) / f'fold_{fold}')
        print(f"Fold {fold} | Train Loss: {metrics['train_loss']:.4f} | "
              f"Train C-index: {metrics['train_cindex']:.4f} | Val Loss: {metrics['val_loss']:.4f} | "
              f"Val C-index: {metrics['val_cindex']:.4f}")
        folds.append({'fold': fold, **metrics})
    save_cv_results(folds, args.results_root)


if __name__ == '__main__':
    main()
