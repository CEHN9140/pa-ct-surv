# Train/Val EE–EC、事件时间间隔、收集批次与删失敏感性

复用四个Seed（123/7/42/2026）×五折的匹配Baseline最终checkpoint；这些是内层STOP选轮后Outer Train重训的最终checkpoint，不是统一训练到最后一轮；Train预测由同一checkpoint的eval()计算。无新训练、选模、标签修改或病例删除。40份预测完成成员/标签/时间核对和C-index复算，Val与上一审计20项结果一致。统计单元仍是患者，不是独立患者对或独立20个模型队列。

## 1. Train/Val EE与EC

```text
split pair_type  cindex_mean  cindex_model_sd  pair_entries    credit  pair_weighted_cindex
train        EC     0.901354         0.061165       1250908 1128028.0              0.901767
train        EE     0.675657         0.072871        400452  270566.0              0.675652
  val        EC     0.781701         0.048760         77944   61031.0              0.783011
  val        EE     0.579388         0.027053         24660   14293.0              0.579603
```

```text
 fold split       EC       EE
    0 train 0.938285 0.679472
    0   val 0.716516 0.605377
    1 train 0.853332 0.623118
    1   val 0.808131 0.566749
    2 train 0.831762 0.621617
    2   val 0.839473 0.553513
    3 train 0.904424 0.649330
    3   val 0.807749 0.609158
    4 train 0.978965 0.804750
    4   val 0.736636 0.562142
```

cindex_mean是20个模型的均值；pair_weighted_cindex为描述性的患者对加权均值，均不跨模型比较原始风险值。pair_entries重复计数训练患者/Seed，绝非独立样本量。Train表现是拟合能力，不是独立预测性能。

## 2. 仅EE：事件时间间隔

```text
split gap_bin  cindex_mean  cindex_model_sd  pair_entries   credit  pair_weighted_cindex
train     0-6     0.532677         0.027197         67676  36121.0              0.533734
train    6-12     0.599427         0.063597         58224  34969.0              0.600594
train   12-24     0.665634         0.085020         87044  57792.0              0.663940
train     >24     0.756871         0.093414        187508 141684.0              0.755616
  val     0-6     0.514668         0.038732          4268   2210.0              0.517807
  val    6-12     0.571027         0.043998          3636   2090.0              0.574807
  val   12-24     0.581936         0.034241          5360   3107.0              0.579664
  val     >24     0.598646         0.065019         11396   6886.0              0.604247
```

时间差采用(0,6]、(6,12]、(12,24]、>24个月。同时间的EE不构成可比较对。ee_gap.csv保存每折每Seed的计数、风险margin中位数、以Outer Train风险SD标准化的有符号/绝对margin中位数与risk-tie比例。更长间隔易排序并不证明模型能精确预测时间；它也可能反映病例组成差异。

## 3. 数据收集批次

```text
batch   n  events  event_fraction  event_time_median  censor_time_median  reverse_km_followup_median  at_risk36  at_risk60  survival36  survival60
  第一批 280      63        0.225000              27.50           60.000000                        61.0        217        125    0.864687    0.787677
  第二批 440     188        0.427273              21.85           76.533333                        79.0        289        225    0.709579    0.626900
   补充   1       0        0.000000                NaN           42.300000                        42.3          1          0    1.000000         NaN
```

```text
split pair_type batch_scope  cindex_mean  cindex_model_sd  pair_entries   credit  pair_weighted_cindex
train        EC       cross     0.905162         0.058592        588052 532506.0              0.905542
train        EC      within     0.897967         0.063490        662856 595522.0              0.898418
train        EE       cross     0.687431         0.070548        151224 103825.0              0.686564
train        EE      within     0.668520         0.074611        249228 166741.0              0.669030
  val        EC       cross     0.797512         0.057747         36652  29286.0              0.799029
  val        EC      within     0.767659         0.041400         41292  31745.0              0.768793
  val        EE       cross     0.586822         0.032806          9420   5533.0              0.587367
  val        EE      within     0.575100         0.026251         15240   8760.0              0.574803
```

```text
split pair_type batch_pair  pair_entries   credit  pair_weighted_cindex
train        EC   第一批->第一批        138820 125220.0              0.902031
train        EC   第一批->第二批        173360 148870.0              0.858733
train        EC    第一批->补充           532    498.0              0.936090
train        EC   第二批->第一批        412384 381420.0              0.924915
train        EC   第二批->第二批        524036 470302.0              0.897461
train        EC    第二批->补充          1776   1718.0              0.967342
train        EE   第一批->第一批         24820  17699.0              0.713094
train        EE   第一批->第二批         67892  45822.0              0.674925
train        EE   第二批->第一批         83332  58003.0              0.696047
train        EE   第二批->第二批        224408 149042.0              0.664156
  val        EC   第一批->第一批          8692   7069.0              0.813277
  val        EC   第一批->第二批         10784   7848.0              0.727745
  val        EC    第一批->补充            28     24.0              0.857143
  val        EC   第二批->第一批         25720  21322.0              0.829005
  val        EC   第二批->第二批         32600  24676.0              0.756933
  val        EC    第二批->补充           120     92.0              0.766667
  val        EE   第一批->第一批          1480    914.0              0.617568
  val        EE   第一批->第二批          4268   2452.0              0.574508
  val        EE   第二批->第一批          5152   3081.0              0.598020
  val        EE   第二批->第二批         13760   7846.0              0.570203
```

```text
split  second_vs_first_risk_auc
train                  0.636605
  val                  0.630788
```

batch_pairs.csv中的A->B表示事件锚点来自A、比较对象来自B，保留方向；补充病例单独保留，不并入任一大批次；指定时间无患者仍在随访时不外推该批次KM值。batch_risk.csv使用各模型Outer Train风险均值/SD标准化，batch_risk_association.csv报告第二批风险高于第一批的概率（Mann–Whitney AUC，ties记半分），不是DFS AUC。

批次是既有文件收集批次，不是已验证的医院、扫描仪或染色批次。原始事件比例与删失和随访长度相关；KM估计仍依赖相应删失假设。同批/跨批排序差异未经病例构成匹配，不能直接判为技术混杂或因果机制。边际风险差异也不能证明模型编码了技术批次。

## 4. 删失敏感性：预设36/60月

```text
split  tau  harrell_same_pairs      uno  max_weight  uno_minus_harrell
train 36.0            0.858363 0.857866    1.237957          -0.000498
train 60.0            0.851565 0.850192    1.621938          -0.001373
  val 36.0            0.740119 0.739336    1.207025          -0.000784
  val 60.0            0.739354 0.738976    1.620914          -0.000378
```

Uno使用对应Outer Train估计删失KM。Harrell_same_pairs与Uno使用完全相同的事件锚点t<tau，保留晚于tau的比较对象；不将样本直接删到tau前，不把完整Harrell与截断Uno的差当成删失修正效果。所有成功结果均以官方concordance_index_ipcw复算。unsupported将原样记录，不改tau寻找有利结果。

Uno采用边际删失KM，不能证明删失与影像/批次独立，亦不能修复信息性删失。这个对照针对总体排序，不能自动修复EE预测或解释时间AUC。

方法依据：[scikit-survival IPCW官方文档](https://scikit-survival.readthedocs.io/en/stable/api/generated/sksurv.metrics.concordance_index_ipcw.html)。

## 本轮结果与下一步判断

1. EE并非只在Val较弱：Train均值0.6757、Val0.5794；二者均低于EC（0.9014/0.7817）。在当前选定checkpoint下，模型同时存在较弱的训练EE区分和Train→Val迁移损失；因训练预算由内层选定，Train EE不是模型最大拟合能力，不能简单归结为‘只学会是否复发’或‘完全无法学到时间信息’。
2. EE时间越近越难：≤6月Train/Val约0.533/0.515；>24月约0.757/0.599。近时间差是部分原因，但较大间隔仍有明显泛化差距。此处不是事件时间预测误差，也不是个体事件顺序的理论上限。
3. 第二批随访中位数（reverse-KM）79月，第一批61月；原始事件比例受观察机会影响。KM36/60仍有差异，但临床病例组成未知，不能据此判定技术批次混杂。验证集同批EC0.768、跨批EC0.798；第一批内部EC约0.813、第二批约0.757（患者对加权描述性均值），不支持高EC完全由跨批比较产生。第二批风险更高与真实病例差异、删失和技术因素均可能相关，尚未分离。
4. 固定tau36/60后，Val Uno与同一患者对Harrell仅差约-0.00078/-0.00038。当前指定时间窗内边际IPCW修正没有实质改变总体排序估计；不能推断不存在信息性删失，也不能据此解释全时段指标或单独EE。
5. 这些结果尚未提供选择新Loss/编码器的充分依据。优先保留匹配内层选模与低探索成本，不依据Outer Val挑选新配置。若下一步要验证表示，应采用预先固定的公平对照；并非本轮自动启动新的模型。

## 解释边界

所有结果仍是已反复探索的同队列内部诊断。Train EE高/Val EE低支持训练拟合未迁移；不等于患者数不足的因果证明，也不是UNI信息上限证据。不可据Outer Val删除批次、筛患者或选loss并声称独立泛化改进。没有启动任何新的模型或损失函数实验。
