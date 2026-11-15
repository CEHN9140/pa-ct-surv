# Stage5：MeanPool + Ridge-Cox

这是独立低容量病理诊断，不替代正式病理–CT教师/CT学生方法。
直接使用原721例、251事件、既定五折的TRIDENT UNI1024全部Raw patch，
均值聚合后仅使用折内StandardScaler和Ridge-Cox；没有MLP或Attention。

Outer Train内event分层三折、seed123，固定alpha={0.1,1,10,100,1000,10000}；
共90次内层拟合尝试+5次最终拟合，任何数值/收敛警告使该次拟合失败。
任一Inner失败则整个alpha不可选，不能平均其他折；仍尝试全部90次并记录异常。
没有完整候选或最终Outer拟合失败时中止。只保存5个最终模型。
Outer Val不参与Scaler、alpha或拟合。输出风险不经过Sigmoid。

在项目根目录运行（CPU）：

```bash
setsid nohup env OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MPLCONFIGDIR=/tmp/pact_matplotlib \
  tests/16_ct_radiomics/.runtime/bin/python -u \
  tests/fold_difficulty_analysis/pathology_stage5/run_stage5.py \
  > tests/fold_difficulty_analysis/pathology_stage5.log 2>&1 < /dev/null &
```

默认results/；已有内容时拒绝覆盖，重新运行须使用新的`--output`目录。
初次FIT-only数值失败记录存于results_failed_initial_solver/，没有任何Outer Val评估；
失败处理规则修订在查看Outer Val前完成，不改变网格或求解器。
主指标为五折Train/Val C-index、Gap，辅助36/60月训练IPCW AUC和患者对错误重合。
报告对照实验10历史ABMIL Best/Last时明确区分选模协议，不直接声称哪种方法更优。

results/private/仅本地保存患者缓存、原始特征来源哈希、5个模型、病例预测/OOF/错误档案。
公开范围仅源代码、聚合CSV、PNG/PDF和验证报告；此次不自动commit/push。
