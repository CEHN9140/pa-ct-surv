#!/usr/bin/env bash

BASE=/home/gly001/cqj/pa_ct_surv
PYTHON=/home/gly001/.conda/envs/UNI/bin/python

# 选择：pathology / pact / ct / ct_student
MODEL_TYPE=ct
OUTPUT_ROOT=${BASE}/vis/pact_v5

# 根据模型类型自动选择对应配置；当前 CT/PACT/Student 继续使用 v4
if [[ "${MODEL_TYPE}" == "pathology" ]]; then
  CONFIG=${BASE}/results/pact_v5/pathology/roi64_abmil_dropout0.0_epochs30_coxbs64_lr1e-4_wd5e-4_seed42/run_config.yaml
elif [[ "${MODEL_TYPE}" == "pact" ]]; then
  CONFIG=${BASE}/results/pact_v4/pact/roi64_abmil_resnet18_weighted_normnone_fusion_dropout0.0_epochs30_coxbs64_lr1e-4_blr1e-5_wd5e-4_lct0.5_lpa0.3_seed42/run_config.yaml
elif [[ "${MODEL_TYPE}" == "ct_student" ]]; then
  CONFIG=${BASE}/results/pact_v4/ct_student/roi64_abmil_resnet18_weighted_normnone_fusion_dropout0.0_epochs30_coxbs64_lr1e-4_blr1e-5_wd5e-4_lct0.5_lpa0.3_seed42_resnet18_dropout0.5_mse_akd0.3_t2.0/run_config.yaml
else
  CONFIG=${BASE}/results/pact_v4/ct/roi64_resnet18_dropout0.0_epochs30_bs64_lr1e-4_blr1e-5_wd5e-4_seed42/run_config.yaml
fi

# checkpoint_root 和 seed 都从配置读取
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
  fi

  if [[ "${MODEL_TYPE}" == "ct" || "${MODEL_TYPE}" == "ct_student" || "${MODEL_TYPE}" == "pact" ]]; then
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
