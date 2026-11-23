# 同CONCH特征：MeanPool、从头ABMIL与冻结TITAN

新增D组为原ABMIL，768→512，Attention512→128→1，预测头512→256→1；从seed123新初始化，全部CONCHv1.5 20×512 Patch，无额外L2或Patch采样。未载入项目下游单模态checkpoint作为初始化，TITAN/MeanPool/Ridge既有结果未重训或修改。原721例、251事件及5outer/3inner患者角色不变。

ABMIL沿用Stage6：Dropout0.25、Adam lr1e-4/WD5e-4、Cox64合尾风险集，联合Cox两遍VJP/Dropout重放。15个inner完整训练180次更新，在预设[9, 18, 36, 54, 72, 90, 117, 144, 180]上按3HELD均值选步；相同时选较早者。5个outer重新初始化并训练恰好选定更新数，OuterVal仅最终评价。这里控制的是更新次数，不是统一Epoch；范围不随OuterVal调整。

## 三组同特征结果

```text
     model  train_cindex  val_cindex   val_sd      gap    auc36    auc60
     abmil      0.769096    0.739938 0.046281 0.029158 0.781506 0.760606
conch_mean      0.744696    0.729569 0.057797 0.015127 0.763318 0.745088
     titan      0.758878    0.741009 0.052326 0.017868 0.781778 0.757234
```

ABMIL与Ridge两方案同时改变聚合、风险头、正则和优化，不是Attention或外部预训练的单一因果消融。比较仅表明这些完整学习流程的内部表现。MeanPool/TITAN两组Cox仍采用FIT-only StandardScaler及既定alpha网格。

## 验证EE/EC

```text
     model       EC       EE  EE_gap>24      all
     abmil 0.784911 0.596581   0.633703 0.739938
conch_mean 0.771882 0.594343   0.619945 0.729569
     titan 0.780082 0.616224   0.650009 0.741009
```

子组是逐折C-index均值，不能把可比较患者对当独立样本；EE_gap>24仅事件时间差超过24月。删失病例不是确认的永久无事件病例。

## 配对患者Bootstrap

```text
              contrast     group  delta_cindex     lower    upper  positive_folds  valid_repeats
abmil minus conch_mean       all      0.010368 -0.003214 0.024652               3           2000
abmil minus conch_mean        EE      0.002238 -0.024009 0.028059               3           2000
abmil minus conch_mean        EC      0.013029 -0.001531 0.028296               3           2000
abmil minus conch_mean EE_gap>24      0.013758 -0.030756 0.056830               2           2000
     abmil minus titan       all     -0.001072 -0.015591 0.013137               2           2000
     abmil minus titan        EE     -0.019643 -0.043794 0.005474               2           2000
     abmil minus titan        EC      0.004829 -0.012535 0.021927               4           2000
     abmil minus titan EE_gap>24     -0.016306 -0.055730 0.024037               2           2000
titan minus conch_mean       all      0.011440 -0.005364 0.028143               2           2000
titan minus conch_mean        EE      0.021881 -0.002832 0.047175               4           2000
titan minus conch_mean        EC      0.008200 -0.011792 0.027195               2           2000
titan minus conch_mean EE_gap>24      0.030064 -0.007913 0.071295               4           2000
```

每折从完整OuterVal患者集合配对抽样2000次，重复患者以各患者出现次数乘积赋予患者对权重；EE/EC/大时间差EE先固定mask，分母也重新计算。不是独立抽患者对，也不是仅抽成功或失败病例。所有contrast使用同一组随机抽样。all为平均五折差值；子组同样取五折差值均值。TITAN−MeanPool总体区间已复现之前结果。

这些名义95%区间条件于已拟合模型，不含重训练、内层选择、折间训练重叠、多次探索或本轮12项比较的多重性。EE区间是针对既有发现的后续探索，不能当作新独立验证。仅一个神经训练seed。

## 审计

20个新初始化训练；每折4个模型初态相同，outer中途HELD/Val评估调用为0；15inner各9个预设点评估，outer只最终评估。全部原CONCH向量/坐标hash逐例匹配上轮TITAN输入，未改标签/病例/划分。30项保存预测C-index复算通过，权重重载检查沿用Stage6 helper。代码/输入来源核验记录见source_validation.json。

所有checkpoint、患者预测、内层病例角色和源路径仅private，不发布。不会依据OuterVal筛Patch/患者或自动追加正则化、TITAN微调、学习曲线、教师/蒸馏实验。本诊断不改变正式病理–CT教师/CT学生的初始化约束。
