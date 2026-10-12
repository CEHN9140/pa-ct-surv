#!/usr/bin/env bash

BASE=/home/gly001/cqj/pa_ct_surv
PYTHON=/home/gly001/.conda/envs/UNI/bin/python

# 选择：pathology / pact / ct / ct_student
MODEL_TYPE=ct
OUTPUT_ROOT=${BASE}/vis/pact_v5

# 当前模型对应的配置和 checkpoint
CONFIG=${BASE}/results/pact_v4/ct/roi64_resnet18_dropout0.0_epochs30_bs64_lr1e-4_blr1e-5_wd5e-4_seed42/run_config.yaml
CHECKPOINT_ROOT=$(awk -F': ' '$1 == "checkpoint_root" {print $2; exit}' "${CONFIG}")
if [[ -z "${CHECKPOINT_ROOT}" ]]; then
  echo "checkpoint_root not found in ${CONFIG}" >&2
  exit 1
fi
SAMPLE_SEED=$(awk -F': ' '$1 == "seed" {print $2; exit}' "${CONFIG}")
if [[ -z "${SAMPLE_SEED}" ]]; then
  echo "seed not found in ${CONFIG}" >&2
  exit 1
fi

# 默认从当前 Fold 的验证集抽取 10 对病例；病理和 CT 使用同一随机种子保持对应
SAMPLE_N=10
SAMPLE_SPLIT=val

mkdir -p "${OUTPUT_ROOT}"

for FOLD in 0 1 2 3 4; do
  if [[ "${MODEL_TYPE}" == "pathology" || "${MODEL_TYPE}" == "pact" ]]; then
    if [[ -z "${WSI_DIR:-}" ]]; then
      echo "请先设置 WSI_DIR 环境变量" >&2
      exit 1
    fi
    "${PYTHON}" -m visualization.path_wsi_heatmap \
      --model_type "${MODEL_TYPE}" \
      --config "${CONFIG}" \
      --checkpoint_root "${CHECKPOINT_ROOT}" \
      --fold "${FOLD}" \
      --output_root "${OUTPUT_ROOT}" \
      --sample_n "${SAMPLE_N}" \
      --sample_split "${SAMPLE_SPLIT}" \
      --sample_seed "${SAMPLE_SEED}" \
      --wsi_dir "${WSI_DIR}"
  else
    "${PYTHON}" -m visualization.ct_gradcam \
      --model_type "${MODEL_TYPE}" \
      --config "${CONFIG}" \
      --checkpoint_root "${CHECKPOINT_ROOT}" \
      --fold "${FOLD}" \
      --output_root "${OUTPUT_ROOT}" \
      --sample_n "${SAMPLE_N}" \
      --sample_split "${SAMPLE_SPLIT}" \
      --sample_seed "${SAMPLE_SEED}"
  fi
done
