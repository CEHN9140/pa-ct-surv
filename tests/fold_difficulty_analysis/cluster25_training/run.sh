#!/usr/bin/env bash
set -eu
cd "$(dirname "$0")/../../.."
experiment=tests/fold_difficulty_analysis/cluster25_training
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=0 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 MPLBACKEND=Agg
export MPLCONFIGDIR="$PWD/$experiment/private/matplotlib"
exec .radiomics_env/bin/python -u "$experiment/train25.py" \
 --data_dir "$PWD/data/trident/features_uni_v1" --ct_roi_size 64 --feature_dim 1024 \
 --pa_model abmil --survival_loss cox --dropout 0.25 \
 --num_epochs 30 --patience 10 --cox_batch_size 32 --lr 1e-4 --weight_decay 5e-4 \
 --checkpoint_root "$PWD/$experiment/checkpoints" --results_root "$PWD/$experiment/results"
