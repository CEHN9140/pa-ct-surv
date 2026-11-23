# 同CONCH Patch：MeanPool vs TITAN

原721例、251事件、既定五折。核心B/C是同一20×512 CONCHv1.5特征的全Patch均值与冻结TITAN公开预训练聚合器；使用相同FIT-only StandardScaler、3折内层alpha选择及Ridge-Cox。UNI仅作不同视野的历史背景，不是纯聚合对照。

模型固定版本见config.json，权重位于用户指定的`/home/gly001/cqj/pathology_foundation_model/titan/`。TITAN源代码/权重、CONCH特征、派生矩阵、患者预测与Cox模型均不上传。推理中不使用文本、DFS标签或项目训练的单模态权重。

依赖使用现有.radiomics_env；仅将缺少的`einops-exts==0.0.4`以`pip --target private/deps --no-deps`安装在本实验内。为兼容Transformers本地动态导入，将官方平铺Python模块复制到private/hf/modules，模型代码本身未修改；Tokenizer离线缓存同样在private。GPU推理需要宿主GPU权限。

从仓库根目录：

```bash
.radiomics_env/bin/python -m pip install --no-deps --target tests/fold_difficulty_analysis/titan_aggregation_comparison/private/deps einops-exts==0.0.4
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .radiomics_env/bin/python tests/fold_difficulty_analysis/titan_aggregation_comparison/extract.py --smoke
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .radiomics_env/bin/python -u tests/fold_difficulty_analysis/titan_aggregation_comparison/extract.py
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .radiomics_env/bin/python -u tests/fold_difficulty_analysis/titan_aggregation_comparison/fit.py
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .radiomics_env/bin/python -m pytest -q tests/fold_difficulty_analysis/titan_aggregation_comparison/test_extract.py tests/fold_difficulty_analysis/titan_aggregation_comparison/test_fit.py tests/fold_difficulty_analysis/titan_aggregation_comparison/test_report.py
```

extract.py每次核验完整输入并生成两套私有表示，不自动跳过缺失/非法病例。默认fit.py在CPU并行拟合两组；也支持`--arm conch_mean`或`--arm titan`单独拟合，然后运行report.py汇总。候选alpha、失败处理和内层角色沿用Stage5，完整协议见PLAN.md。

两组输入Cox前统一转换为float64，保留原缓存数值。首次刚开始的拟合因缓存精度不一致被终止并从头重跑，未依据外层成绩改变模型/候选网格；冻结TITAN提取不受影响。

## 本次完成结果

CONCH MeanPool五折Val C-index 0.7296，冻结TITAN 0.7410；平均差值+0.0114，配对名义95%区间[-0.0054,+0.0281]，仅2/5折为正，未确认稳定改善。EE均值0.5943→0.6162，>24月EE为0.6199→0.6500，但未对子组重新估计区间或开展多重检验。主对照完整报告见[results/report.md](results/report.md)。

两组均完成五折、同内层角色，全部选择alpha10000。300次内层尝试无失败，10个外层模型成功；20项保存模型/预测复算、10项正则选择复核和10项测试通过。权重位于指定本地目录，不随代码发布。
