# Stage5 Ridge-Cox与Stage6严格Nested ABMIL：匹配排序错误

原721例、251事件、同一五折。仅复用已有严格Nested模型；不使用历史Outer Val选Best ABMIL。两者都是唯一seed123，不能混称为上一轮四Seed实验。无新训练、特征提取、模型选择、标签修改或病例剔除。

20份Train/Val预测成员、标签、float32时间及风险有限性均核对。20项C-index、5项Val Spearman与5项ΔC均复现既有Stage5/6记录。

## 全部患者对

```text
split  n_pairs  ridge_cindex  abmil_cindex  delta_cindex  delta_fold_mean  ridge_correct__abmil_correct  ridge_wrong__abmil_correct  ridge_correct__abmil_wrong  ridge_wrong__abmil_wrong  tie_involved  shared_fraction_ridge_wrong  shared_fraction_abmil_wrong
train   412840      0.749612      0.797621      0.048009         0.048000                        296947                       32343                       12523                     71027             0                     0.687114                     0.850114
  val    25651      0.720830      0.733890      0.013060         0.013495                         17319                        1506                        1171                      5655             0                     0.789694                     0.828450
```

## EE与EC

```text
split pair_type  n_pairs  ridge_cindex  abmil_cindex  delta_cindex  delta_fold_mean  ridge_correct__abmil_correct  ridge_wrong__abmil_correct  ridge_correct__abmil_wrong  ridge_wrong__abmil_wrong  tie_involved  shared_fraction_ridge_wrong  shared_fraction_abmil_wrong
train        EC   312727      0.799595      0.855526      0.055931         0.055933                        244516                       23030                        5539                     39642             0                     0.632531                     0.877404
train        EE   100113      0.593479      0.616743      0.023264         0.023252                         52431                        9313                        6984                     31385             0                     0.771168                     0.817978
  val        EC    19486      0.762958      0.778610      0.015652         0.016140                         14157                        1015                         710                      3604             0                     0.780255                     0.835420
  val        EE     6165      0.587672      0.592539      0.004866         0.004915                          3162                         491                         461                      2051             0                     0.806845                     0.816481
```

## 仅EE：按事件时间间隔

```text
split gap_bin  n_pairs  ridge_cindex  abmil_cindex  delta_cindex  delta_fold_mean  ridge_correct__abmil_correct  ridge_wrong__abmil_correct  ridge_correct__abmil_wrong  ridge_wrong__abmil_wrong  tie_involved  shared_fraction_ridge_wrong  shared_fraction_abmil_wrong
train     0-6    16919      0.522253      0.522430      0.000177         0.000401                          7396                        1443                        1440                      6640             0                     0.821477                     0.821782
train    6-12    14556      0.558533      0.567326      0.008794         0.008748                          6965                        1293                        1165                      5133             0                     0.798786                     0.815021
train   12-24    21761      0.589357      0.608060      0.018703         0.018783                         11202                        2030                        1623                      6906             0                     0.772829                     0.809708
train     >24    46877      0.631952      0.670158      0.038206         0.038278                         26868                        4547                        2756                     12706             0                     0.736452                     0.821757
  val     0-6     1067      0.531396      0.531396      0.000000         0.002430                           475                          92                          92                       408             0                     0.816000                     0.816000
  val    6-12      909      0.562156      0.574257      0.012101         0.006953                           443                          79                          68                       319             0                     0.801508                     0.824289
  val   12-24     1340      0.593284      0.604478      0.011194         0.011721                           702                         108                          93                       437             0                     0.801835                     0.824528
  val     >24     2849      0.614251      0.615655      0.001404         0.000541                          1542                         212                         208                       887             0                     0.807097                     0.810046
```

时间区间为(0,6]、(6,12]、(12,24]、>24月；同时间EE不可比较。correct=1、tie=0.5、wrong=0，风险并列容差1e-8。完整九种转移均保存在CSV；纠正/破坏只指wrong→correct/correct→wrong，不把tie强行归为错误。credit差精确重构ΔC（包括tie产生的半分）。

n_pairs是折内可比较患者对计数。Val没有跨折比较原始风险；Train患者和患者对在不同Outer Train重复出现，不能解释为独立样本。ridge_cindex/abmil_cindex/ΔC为患者对加权描述值，delta_fold_mean为五折ΔC的均值，二者不能混淆。没有基于患者对数做独立性显著性检验。

## 风险相关性

```text
 fold split  risk_spearman
    0 train       0.968104
    0   val       0.963781
    1 train       0.928114
    1   val       0.932067
    2 train       0.967909
    2   val       0.964275
    3 train       0.906251
    3   val       0.913938
    4 train       0.913202
    4   val       0.925014
```

高相关不代表模型等价；ABMIL含投影、Attention、非线性头，且优化、正则和风险集设置与Ridge不同。收益不能归因于Attention单独作用。Ridge五折都选择alpha网格上界10000，本轮不事后扩展网格，亦不将它视作最优线性模型上限。

## 大时间差且大错误margin的EE

```text
 fold split  n_large_ee  both_wrong  both_strong_wrong  ridge_strong_wrong  abmil_strong_wrong  ridge_abs_margin_z_median_common  abmil_abs_margin_z_median_common
    0 train        9201        2947                862                1362                 968                          0.909245                          0.696377
    0   val         628         169                 49                  63                  58                          0.803599                          0.732817
    1 train        9736        2501                646                1254                 794                          0.937236                          0.674428
    1   val         470         184                 49                  85                  67                          0.913460                          0.799540
    2 train        9510        2835                864                1253                1012                          0.902660                          0.724175
    2   val         562         211                 76                 106                  91                          1.006527                          0.851753
    3 train        9214        2215                511                1275                 672                          0.979569                          0.630787
    3   val         618         149                 34                  58                  51                          0.747786                          0.678356
    4 train        9216        2208                418                1190                 614                          0.943627                          0.592651
    4   val         571         174                 70                 119                  74                          1.264512                          0.790380
```

预设大时间差>24月；每个模型的margin均除以其对应Outer Train风险SD，负值表示错误方向。both_strong_wrong要求两模型均完全错误且各自margin≤-1 SD。这个阈值不是置信概率，也没有依据Outer Val选取。所有Val患者对和大margin共同错误清单仅存在private/ridge_overlap/，不发布患者ID，不据此改变标签或剔除病例。

## 本轮判断

- Val EE共6165对：双方正确3162，ABMIL纠正491、破坏461，共同错误2051。共同错误占Ridge/ABMIL各自EE错误约80.7%/81.6%。
- >24月EE：纠正212、破坏208，净增加4个正确比较；患者对加权ΔC=+0.0014，五折平均ΔC=+0.0005。共享错误887对，其中两模型均错误margin≥1训练SD的共有278对；并非独立患者数。
- EE的五折平均ΔC=+0.0049，仅2/5折为正；EC的五折平均ΔC=+0.0161。ABMIL并非没有任何增量，但本次增量主要在EC，大时间差EE没有呈现稳定超过线性基线的收益。
- Train >24月EE的患者对加权提升约0.0382，Val约0.0014。在当前内层选轮的checkpoint下，额外训练拟合未转为同等幅度的EE增益；这不是ABMIL的最大拟合能力测试。既有整体ΔC名义区间[-0.0010,+0.0282]仍跨0，本轮未重新估计EE子组区间或进行多重比较检验。
- ‘共同出错’仅定位当前两种模型都未解决的预测关系，不能区分表示缺失、共享读取偏差、有限监督或未测临床因素。本轮不支持自动换基础模型或改变Cox Loss，更不支持认定患者标签错误。

## 解释边界

共享错误支持‘当前两种读取方式未解决这些患者对’，不证明UNI缺少信息、WSI无法预测或标签错误。纠正和破坏是当前已训练模型的描述性比较，不能直接用Outer Val错误选取patch、患者或训练约束，再在同一Val声称独立提升。训练模型、内层选参、同队列长期探索及未测临床变量仍影响结论。不会据此自动启动编码器替换或新Loss实验。
