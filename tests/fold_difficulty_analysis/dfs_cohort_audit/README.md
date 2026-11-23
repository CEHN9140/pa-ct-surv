# DFS cohort audit (no training)

Run from repository root:

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .radiomics_env/bin/python tests/fold_difficulty_analysis/dfs_cohort_audit/audit.py
PYTHONDONTWRITEBYTECODE=1 .radiomics_env/bin/python -m pytest -q tests/fold_difficulty_analysis/dfs_cohort_audit/test_audit.py
```

Requires local original label workbooks, canonical patient CSV, historical cohort metadata and four matched Baseline prediction sets (123, 7, 42, 2026). Writes only within this experiment. `results/` contains aggregate tables; `private/` contains patient IDs and source review lists and must not be uploaded. No original data, labels, folds or main-process code are changed.

A day-table comparison requires unique CT ID and matching pathology ID. Ambiguous CT IDs and discordant pathology IDs are counted, never silently resolved. The clinical definition of DFS cannot be inferred from numeric status columns. No neural training, Ridge-Cox repeat or new learning curve is part of this audit.

EE/EC、时间间隔、批次与IPCW后续分析：

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .radiomics_env/bin/python tests/fold_difficulty_analysis/dfs_cohort_audit/pair_followup.py
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .radiomics_env/bin/python -m pytest -q tests/fold_difficulty_analysis/dfs_cohort_audit/test_followup.py
```

结果在`followup_results/`，复用固定checkpoint的40份Train/Val预测。Uno固定tau为36/60月，删失分布仅由对应Outer Train拟合，保留所有晚于tau的比较对象。模型/Seed的重复计数不是新增独立患者。
