# 实验10：固定验证患者的训练样本量学习曲线

目的：检验在当前UNI＋ABMIL＋Cox训练方法下，增加训练患者是否持续改善验证预测。
输入是原肺癌队列`data/trident/features_uni_v1/all_label_roi64.csv`，721例、251个DFS事件；
使用原始1024维UNI，不做L2，不更改模型、特征、标签或五折。

## 固定的比较方案

每折验证患者保持原样（144/145人），训练池为576/577人。
128、256、384人各抽3组，每次重复内三个名单嵌套；全部训练患者只训练一次。
共`5 × (3 × 3 + 1) = 50`次训练，按顺序执行。
仅在训练池内先保持event比例，再兼顾各event组内的时间四分位分布。
抽样首次不设固定seed，随后名单和源CSV/特征指纹冻结；训练seed默认123，每次重新初始化并复位。
三次重复反映抽样差异，不同时改变初始化seed。

| 每折训练人数 | 通常每轮更新次数 | 180次更新约对应轮数 |
|---|---:|---:|
|128|2|90|
|256|4|45|
|384|6|30|
|全部576/577|9|20|

所有模型均为ABMIL、Dropout0.25、Adam lr1e-4、weight_decay5e-4、事件平均Cox。
物理batch1，联合Cox风险集64人；按原规则尾部并入，577人的最后一组是65人。
每次固定180个optimizer.step，每9次更新在原验证集评估，共20个验证点，**不提前停止**。
验证时完整训练子集与验证集都使用eval模式，同一checkpoint计算Train/Val指标。
验证过程保存和恢复PyTorch RNG，不额外改变训练Dropout或患者打乱序列。
固定的是参数更新预算，不是epoch、墙钟时间或每个患者的重复曝光次数。

同时保留固定180次更新的last模型，以及相同20次验证机会中选出的best模型。
主要用last结果观察固定预算下的数据量效应，best结果作为补充；二者都是内部验证，best存在验证选模因素。
此次全量训练预算与旧主流程的30轮/早停10不同，因此不直接复用旧checkpoint。

## 运行

在项目根目录运行，脚本自带nohup和日志保存：

```bash
bash tests/10_pathology_learning_curve/run.sh
```

```bash
tail -f tests/10_pathology_learning_curve/logs/learning_curve.log
```

完整实验是50次训练，工作量大于普通五折。不要在同一output_dir同时启动；文件锁会阻止并发。
已有运行配置或checkpoint时会拒绝覆盖，包括中断的运行。重新设计配置应使用新的output_dir，并明确保留或重新准备名单。
日志追加保存。准备数据而不训练：

```bash
python tests/10_pathology_learning_curve/train.py --prepare_only
```

## 输出及解释

- `data/all_label_roi64.csv`：原CSV的字节副本。
- `data/subsets.csv`、`validation.csv`：每组训练名单及固定验证患者。
- `data/manifest.json`：名单、源CSV和特征指纹，重复准备只核对，不重新抽样。
- `results/<run_tag>/metrics.csv`：50组逐折结果；无last前缀的指标来自best，last前缀是180次更新后的结果。
- `results/<run_tag>/<人数_repeat编号>/fold_*/`：best预测、KM及20个更新点的Train/Val指标；`last/`保存固定预算模型的预测及生存分析。
- `checkpoints/<run_tag>/<人数_repeat编号>/fold_*/`：best/last模型、最佳更新步、实际更新/验证次数、患者与事件曝光次数。
- `learning_curve.csv`、`learning_curve.png`、`learning_curve.pdf`：训练人数与C-index、Gap、36/60月AUC的关系。

每个人数档先在同一fold内平均重复抽样，再对五折计算均值和样本标准差。
误差条不是置信区间，50组不是50个独立测试。任一重复的AUC不可估计时，该fold及整体相应汇总保留NaN，不挑可计算的结果取平均。
验证随训练人数反复改善，可支持当前方法受样本规模限制；有限人数范围内没有改善，不能证明病理信息上限或未来增样无效。
需要结合更新曲线判断预算是否足够，不把未充分优化误认为更多患者无益。
完整训练集Cox loss还受风险集人数影响，不依据不同人数档的Train loss绝对大小判断优劣；主要比较C-index、AUC及同一组内的变化。

完成后可单独重画：

```bash
python tests/10_pathology_learning_curve/analysis.py
```

## CPU合成检查

```bash
/home/gly001/.conda/envs/UNI/bin/python -m unittest discover -s tests/10_pathology_learning_curve -p test_experiment.py -v
```

覆盖嵌套无放回抽样、原fold隔离、名单冻结、来源变化拒绝、相同初始化、固定更新/验证次数、25组微型训练及best/last预测核验、先折内再折间聚合、NaN保留、覆盖保护和并发锁。
检查不使用真实特征训练，不改动主流程或既有实验。
