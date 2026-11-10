"""Five-fold teacher experiment: 25 pathology Q tokens attend to CT layer4 KV."""

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from cox_utils import evaluate_survival
from dataset import Pa_CT_Dataset, build_dataloader, cv_splits, resolve_data_dir
from pact_train import build_pact_optimizer, train_pact
from utils import save_cv_results, save_run_config, seed_everything
from path_query_ct import PathQueryCTModel
from train_alignment import train_pact_alignment


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
    args.ct_roi_size, args.pa_feature_dim = 64, 1280
    args.pa_model, args.ct_model = 'abmil', 'resnet18'
    args.fusion_type = 'path_query_ct_layer4'
    args.fusion_l2_normalize = False
    args.survival_loss, args.lambda_ct, args.lambda_pa = 'cox', 0., 0.
    args.ct_token_source, args.ct_token_grid = 'layer4', [8, 8, 8]
    args.pa_query_tokens, args.attention_dim, args.attention_heads = 25, 256, 1
    args.query_aggregation, args.positional_encoding = 'attention_pool', False
    args.query_pool_hidden_dim = 128
    if args.alignment_weight > 0:
        args.alignment_type = 'pamt_topk_unnormalized_target_cross_entropy'
        args.alignment_source = 'PAMT/gene_wsi_predict/train.py + utils_cox.py + vit_model_gene_wsi_concat_label.py'
    args.data_dir = resolve_data_dir(args.data_dir, args.ct_roi_size)
    tag = (f'eagle_top25_virchow2_cls_{"pal2" if args.pa_l2_normalize else "paraw"}_fusionraw_roi64_paq25_ctlayer4kv512'
           f'_crossattn_attnpool_concat_dropout{args.dropout}_epochs{args.num_epochs}'
           f'_coxbs{args.cox_batch_size}_lr{args.lr:g}_blr{args.ct_backbone_lr:g}_wd{args.weight_decay:g}')
    if args.alignment_weight > 0:
        tag += f'_pamtalignk{args.alignment_topk}_w{args.alignment_weight:g}'
    experiment = Path(__file__).resolve().parent
    args.checkpoint_root = args.checkpoint_root or str(experiment / 'checkpoints' / tag)
    args.results_root = args.results_root or str(experiment / 'results' / tag)
    for path in (Path(args.checkpoint_root), Path(args.results_root)):
        if path.exists():
            raise FileExistsError(f'Refusing to overwrite existing experiment: {path}')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed_everything(args.seed)
    save_run_config(args, 'pact', tag, excluded=('alignment_topk',) if args.alignment_weight == 0 else ())
    print(f'Using Device: {device}')
    print('Experiment 15: pathology Q [B,25,256]; CT layer4 KV [B,512,256]')
    print(f'Pathology patch L2: {args.pa_l2_normalize} | Fusion L2: False | Positional encoding: False')
    print('25 retrieved CT tokens -> Attention pooling (256 -> 128 -> 1, softmax over 25) -> 256; '
          'concat with ABMIL patient feature -> 512 -> 256 -> 128 -> 1')
    if args.alignment_weight > 0:
        print(f'Loss: fused Cox + {args.alignment_weight:g} * PAMT alignment; topK={args.alignment_topk}')
        print('Alignment: projected PA [25,256] @ CT [512,256].T; raw dot products; '
              'topK pseudo-positives, original target sum=K; no extra L2/temperature/sqrt(d)')
    else:
        print('Loss: fused Cox only')
    print('CT/PA auxiliary survival losses remain disabled; Val Loss and final metrics losses report fused Cox')
    print(f'Backbone LR: {args.ct_backbone_lr:g} | Other LR: {args.lr:g} | '
          f'Cox batch: {args.cox_batch_size} | Seed: {args.seed} | Patience: {args.patience}')
    dataset = Pa_CT_Dataset(args.data_dir, roi_size=64, feature_dim=1280)
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
                                 pa_l2_normalize=args.pa_l2_normalize).to(device)
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
