# 03：UNI固定除以40的输入尺度对照

读取原始UNI特征后，只执行 `features / 40.0`。
训练、验证、最终评估采用同一Dataset，保持1024维、patch顺序和数量，
保留patch范数之间的相对差异。固定常数40不需要从任何数据集拟合。
原始特征文件不变，不进行逐patch L2归一化。

沿用原始UNI参考CSV中的患者、DFS标签和split，不重新划分。
复用主流程模型、Cox训练、尾部合并、早停及最终生存评估；
主流程及01、02实验代码不变。

从项目根目录运行：

```bash
bash tests/03_uni_fixed_scale_pathology/train.sh
```

脚本自带nohup后台运行和日志保存。默认ABMIL、ROI64队列、
Dropout=0.25、Cox batch=64、Adam lr=1e-4、weight_decay=5e-4、
最多30轮、patience=10、seed=123。
这与原始UNI基线和02实验最初的Dropout=0.25/BS64配置相同；
不沿用后来02脚本中的Dropout=0/BS32设置，避免同时改变多个因素。

本目录输出data清单副本、logs、checkpoints及results。
results下每次运行包括run_config.yaml、合并的metrics.csv和fold_0至fold_4。
每折保存最佳/末轮权重、训练/验证预测、KM图，计算C-index、
36/60个月时间依赖AUC、HR及Log-rank指标。
配置文件记录experiment_id=03、normalization=fixed_divisor、
feature_divisor=40.0及feature_scale=0.025。
相同配置重复运行会使用相同输出目录。

对照解释：

- 原始UNI：保留原始尺度和范数差异。
- 02 UNI L2：单位范数，去掉patch范数差异。
- 03 UNI/40：尺度约为1，保留patch范数差异。

对照同时比较验证性能、单折走势及后期退化，不能只根据平均Gap判断优劣。
