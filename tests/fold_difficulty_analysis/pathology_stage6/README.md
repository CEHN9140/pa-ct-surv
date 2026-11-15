# Stage6：ABMIL严格Nested CV

最后补齐这一次单模态病理诊断，不自动新增结构或Stage7。
原721例/251事件/五折，全部Raw UNI1024 WSI，原ABMIL，Dropout0.25，
Cox64尾部合并、Adam lr1e-4/WD5e-4，唯一seed123，无LR调度器。

每个Outer Train内部的三折角色与Stage5 Ridge逐患者对齐。
15个Inner模型分别从头训练180次更新，在固定Step
`9,18,36,54,72,90,117,144,180`上评估Inner HELD。
按相同Step的三折均值选择一步，平分取早步。5个Outer模型再从头初始化，
恰好训练该Step，不换算Epoch；训练结束后Outer Val只评估一次。
不使用Outer Val选轮，不加载项目训练好的单模态权重初始化教师或学生。

项目根目录后台运行：

```bash
setsid nohup env CUDA_VISIBLE_DEVICES=0 OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=4 \
  MPLCONFIGDIR=/tmp/pact_matplotlib \
  tests/16_ct_radiomics/.runtime/bin/python -u \
  tests/fold_difficulty_analysis/pathology_stage6/nested_train.py \
  > tests/fold_difficulty_analysis/pathology_stage6.log 2>&1 < /dev/null &
```

默认results/，已有内容拒绝覆盖。日志一个pathology_stage6.log。
全部20次训练结束后自动生成Nested ABMIL和Stage5 Nested Ridge的五折均值±样本SD、
逐折ΔC、2000次折内配对患者Bootstrap区间、选步曲线和完整来源核验。
区间不包含重新训练、内层选择及此前多次探索的不确定性；这是内部验证。

病例预测、Inner角色与20个末步模型只保留在被忽略的fold_*/private/和根private/。
公开文件均是聚合统计、来源哈希和图。此次不自动commit或push。
