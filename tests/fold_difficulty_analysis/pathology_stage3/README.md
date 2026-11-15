# Stage3：输入尺度与优化轨迹验证

本实验使用TRIDENT UNI1024全WSI，在原始Fold2、4上使用完整Outer Train。
每组固定60次Adam更新、Cox64且尾部并入、Dropout0.25、seed123、WD0。
不更改ABMIL/Cox，不使用验证集早停或选权重，不加载项目训练好的模型。

|组|输入|首层weight|首层weight LR|其余参数LR|
|---|---|---|---|---|
|A|Raw|W|1e-4|1e-4|
|B|/40|W|1e-4|1e-4|
|C|/40|40W|1e-4|1e-4|
|D|/40|40W|4e-3|1e-4|

首层bias不放大。四组交错训练相同批次：每折577名训练患者、144名验证患者；
每完整轮8×64+65，共9更新。逐步Dropout种子从唯一args.seed派生并共用；
原Stage2两遍重算保证联合Cox梯度及Dropout逐位复现，Adam状态持续更新。

Step0、1、5、10、20、40、60保存完整Train/Val的eval指标及病例预测；
检查A/C/D初始函数数值等价，失败则不允许训练。中途不分析后修改条件。

从项目根目录运行：

```bash
setsid nohup env CUDA_VISIBLE_DEVICES=0 OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=4 \
  tests/16_ct_radiomics/.runtime/bin/python -u \
  tests/fold_difficulty_analysis/pathology_stage3/train_scale.py \
  > tests/fold_difficulty_analysis/pathology_stage3.log 2>&1 < /dev/null &
```

默认输出results/，已有输出时拒绝覆盖；再次运行须明确提供新的`--output`目录。
日志只有一个pathology_stage3.log。status.json只有全部完成才标completed。

各fold保存metrics.csv、gradient_update.csv、function_differences.csv、
batch_schedule.csv和completed.json。全部结束后自动生成总表、report.md及PNG/PDF曲线。
private/保存各固定步的Train/Val患者预测和含Adam状态的模型；该目录禁止纳入Git。
没有best_model，不把最高Val点当作所选模型。对比仅限早期60更新机制，
不能用来解释所有后期过拟合，也不构成新的正式训练建议。
