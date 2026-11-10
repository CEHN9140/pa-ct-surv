# 02：UNI逐patch L2归一化对照

仅改变输入：读取原生UNI特征后，沿每个patch的1024维特征轴进行
`F.normalize(features, p=2, dim=-1, eps=1e-12)`。
保持维度、patch顺序和数量、模型结构及可训练参数量不变。
训练、验证及最终训练集/验证集评估均通过同一Dataset执行归一化。
零向量保持为零，原始PT/H5不会被修改，不需要重新提取特征。

患者、DFS标签和五折split沿用 `data/trident/features_uni_v1/all_label_roi64.csv`，
不重新划分；本实验复用主流程模型、Cox训练及结果分析，不修改主流程。

```bash
bash tests/02_uni_l2_pathology/train.sh
```

默认ABMIL、ROI64队列、Dropout=0.25、Cox batch=64、Adam lr=1e-4、
weight_decay=5e-4、最多30轮、patience=10、seed=123。
这些设置与原始UNI ABMIL对照及01 KEEP实验一致。
修改train.sh中的PA_MODEL可切换模型。当前脚本用于固定配置的重复seed检查：
自带nohup后台运行，依次完成seed42、seed2024各自的五折训练；不重新运行seed123。
种子列表保存在repeat_seeds.py中，不在bash设置。前一次失败会终止任务，不继续下一次。

所有输出限定在本实验目录：

- `data/all_label_roi64.csv`：参考数据清单副本，仍指向原始UNI特征。
- `logs/`：训练日志。
- `checkpoints/<run>/fold_*/`：最佳/末轮权重与轮次元数据。
- `results/<run>/run_config.yaml`：参数及归一化方式。
- `results/<run>/metrics.csv`：五折及均值/标准差指标。
- `results/<run>/fold_*/`：Train/Val预测CSV、验证KM图。

与主流程一致，最佳Checkpoint由Val C-index选择；最终eval模式计算
Train/Val C-index、36/60个月时间依赖AUC、HR和Log-rank等。
对比时同时观察验证性能、单折波动、达到最佳轮次和后期退化，不能只看平均Gap。
相同配置重复运行会覆盖对应实验输出。

## 逐轮排序诊断

新运行的日志和结果标签增加`_diagnostics`，不覆盖此前无诊断的实验。
Python参数默认值如上；train.sh当前固定为ABMIL、ROI64、Dropout=0.25、Cox batch=16、
lr=1e-4、weight decay=5e-4、最多30轮、patience=10。该配置的seed123已有结果，
本次只补seed42和2024，沿用同一CSV内的五折划分。
两种模型的运行标签不同，不覆盖ABMIL结果，均保留第0轮和逐轮排序诊断。

仅测试目录内的Cox训练函数增加诊断，模型、尾部合并、Adam更新、早停和最佳选模规则与主流程一致。
第0轮是未做任何优化更新的初始化模型，额外验证恢复PyTorch CPU/CUDA随机状态，
不改变后续患者打乱或Dropout序列。第0轮不参与最佳模型选择和早停。
验证均为eval模式、完整patch，每轮对应当轮更新结束后的同一个模型状态。

每折`results/<run>/fold_*/diagnostics/`包含：

- `val_predictions_epoch_000.csv`及后续每轮CSV：患者ID、DFS时间、事件和未经Sigmoid的风险分数。
- `epoch_metrics.csv`：第0轮及实际完成轮次的验证Cox Loss、C-index、风险均值和总体标准差，
  与第0轮、第1轮、前一轮患者风险排序的Spearman相关系数，以及可比较患者对和风险同分对数量。
- `val_loss_unit_std`：每轮将验证风险分数减均值、除以其总体标准差后重新计算的Cox Loss，
  用于辅助区分分数尺度变化与排序/相对间距变化；它只做诊断，不参与优化或模型选择，也不是生存概率校准。

相关系数按患者ID对齐并检查标签一致性；无参考轮次或风险完全同分时相关系数为空。
第0轮没有第1轮及前一轮参照；零标准差时标准化Loss为空。汇总不包含未实际训练的轮次，
最终预测、KM和生存指标仍由最佳训练轮次模型生成。`--eval_only`只评估已有最佳模型，不生成逐轮诊断。

## 稳定性运行的输出

- `logs/*_diagnostics_stability.log`记录两个seed的启动、完成及运行错误。
- `logs/stability_seed42.log`和`logs/stability_seed2024.log`分别记录完整训练过程。
- 权重和结果运行名分别包含`seed42_diagnostics`、`seed2024_diagnostics`，与已有seed123区分。
- 每个seed启动及每折开始均使用该seed；数据划分不变。
- 运行完毕后，应分别比较三个seed的五折均值、各折配对表现、Gap及逐轮诊断；
  不从中挑选最高seed，也不将15折当成15次独立实验。

同配置再次执行会覆盖seed42和2024的对应输出；直接运行train.py仍默认seed123。
