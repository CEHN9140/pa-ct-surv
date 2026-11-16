# 训练期Random-128：Raw UNI + ABMIL

这是独立的单模态病理采样诊断，不改变主流程、Stage6模型或已有WSI审查图像。
仅训练时每次无放回随机128个patch，N≤128用全部，不使用Attention/CHIEF筛选，
不做L2。完整原始UNI特征仍在CPU缓存；采样发生在模型前向入口，尚不减少原特征读取。

原ABMIL维度1024→512、Attention128、池化512→256→1，参数722178不变。
直接使用现有 `ABMIL_RandomSample(k=128)`，从头初始化，参数初始值与Full ABMIL相同。
不同更新重新采样；同一次Cox两遍VJP会重放同一采样及Dropout，且逐患者验证Risk一致。

## 匹配Stage6的Nested协议

- 原721例、251事件，`data/trident/features_uni_v1/all_label_roi64.csv`原五折。
- 每个Outer Train内同一三折角色；候选Step固定9/18/36/54/72/90/117/144/180。
- 内层使用Full WSI验证，三折平均C-index选Step，平分取早步。
- 外层从头训练所选步数，不根据Outer Val早停或挑选Checkpoint。
- Dropout0.25，Adam lr1e-4、WD5e-4，seed123；Cox64，尾部并入前组。
- 15次Inner、5次Outer，共20次训练；不重复训练已有Stage6 Full对照。
- 同患者顺序/初始化/Cox组/Step候选，不声称与Full逐神经元Dropout遮罩相同。

## 预先固定的最终评价

主比较：Random128训练→Full WSI评估，与Stage6 Full训练→Full评估比较。
Train/Val均在固定模型eval下计算，Dropout关闭，Cox Loss按事件平均。
另在同一最终模型上评价固定随机128：seed123、repeat0..29，Train/Val同规则。
repeat0单独报告；30次逐次计算指标后平均，不先平均Risk做集成，不按外层结果挑策略。
随机推理的采样与训练Torch采样流分离，且按患者ID/重复确定，不依赖读取顺序。

## 后台运行

在项目根目录运行一次，默认拒绝覆盖已有结果：

```bash
setsid nohup env CUDA_VISIBLE_DEVICES=0 OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=4 MPLCONFIGDIR=/tmp/pact_matplotlib \
  /home/gly001/cqj/pa_ct_surv/tests/16_ct_radiomics/.runtime/bin/python -u \
  /home/gly001/cqj/pa_ct_surv/tests/fold_difficulty_analysis/random128_training/train_random.py \
  > /home/gly001/cqj/pa_ct_surv/tests/fold_difficulty_analysis/random128_training/training.log 2>&1 < /dev/null &
```

```bash
tail -f tests/fold_difficulty_analysis/random128_training/training.log
```

## 输出

`results/metrics.csv`为主Full WSI逐折Train/Val C-index、Gap、36/60月IPCW AUC及Loss；
`random_metrics.csv`为150组Fold/repeat指标；`comparison_summary.csv`为四种训练/评估
组合的mean±五折样本SD，`random_repeat_variability.csv`另列随机重复SD。
`fold_comparison.csv`、`paired_comparison.json`和`report.md`自动对比Stage6。
区间为已拟合模型条件下的折内配对患者Bootstrap，不覆盖重训练、选Step和前期探索。

`inner_curves.csv`、`step_selection.csv`、`fit_audit.csv`保存选步及隔离证据，
`run_config.json`记录固定方案。`status.json`记录进度，全部完成才生成`validation.json`。
模型、内层角色和患者预测位于`fold_*/private/`与`private/`，被Git忽略；日志也不上传。
只允许代码、说明及聚合结果后续人工提交，本任务不执行commit或push。

Gap下降不等于泛化提高；应同时比较Val、AUC、逐折变化及配对区间。
既有队列已被反复探索，本次仍为内部对照，不证明组织学质量，也不保证采样有效。

## 检查

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 \
  tests/16_ct_radiomics/.runtime/bin/python -m pytest \
  tests/fold_difficulty_analysis/random128_training/test_random.py -q
```

测试覆盖采样重放、与直接联合Cox图的梯度一致性、更新间重采样、无放回、
N≤128与eval完整patch、初始化不变、固定随机评估、外层标签扰动不影响训练，
以及患者配对、差值方向、随机重复波动和缺失指标处理。
