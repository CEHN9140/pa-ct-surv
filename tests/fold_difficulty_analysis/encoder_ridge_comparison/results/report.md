# 同Patch五编码器：MeanPool + Nested Ridge-Cox

721例、251事件、原五折。UNI/KEEP/GPFM/Phikon-v2/Kaiko同一TRIDENT Patch网格、逐例坐标顺序/几何一致；全部patch逐块检查finite后float64均值池化，无新增Patch L2/筛选。各编码器保留原有预处理和原生输出，维度不强制相同；比较的是固定特征提取流程，不能进一步把差异只归因于某个预训练因素。

同一Outer Train内按事件分层三折，seed123；每一患者的内层角色与Stage5一致。StandardScaler仅在FIT上拟合。统一alpha=[0.1, 1, 10, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000]；CoxPHSurvivalAnalysis/Breslow/n_iter1000/tol1e-9，警告视为拟合失败；候选必须三折全部成功。平分取强惩罚。Outer Train重新拟合后仅最终评价一次，UNI也按扩展网格重新拟合。

## 五折结果

```text
     encoder  train_cindex  val_cindex   val_sd      gap    auc36    auc60  auc36_folds  auc60_folds
        gpfm      0.769243    0.716794 0.047444 0.052450 0.759150 0.726272            5            5
kaiko-vitb16      0.772498    0.713193 0.046185 0.059305 0.754385 0.723456            5            5
        keep      0.752060    0.725248 0.055140 0.026811 0.759695 0.733985            5            5
   phikon_v2      0.774534    0.697952 0.041964 0.076583 0.740373 0.702237            5            5
      uni_v1      0.738584    0.720064 0.050773 0.018520 0.755437 0.729855            5            5
```

mean/SD为五折描述，非独立测试。AUC为36/60月累计动态IPCW，删失分布由对应Outer Train估计。任何失败保留原因，不删病例；不完整AUC不静默平均。

## 与同新协议UNI的配对差值

```text
     encoder  positive_folds  delta_cindex     lower     upper  valid_repeats
        keep               4      0.005184 -0.006338  0.016348           2000
        gpfm               2     -0.003270 -0.017111  0.010589           2000
   phikon_v2               0     -0.022112 -0.037025 -0.007139           2000
kaiko-vitb16               2     -0.006871 -0.019731  0.006353           2000
```

固定已拟合模型，每折内配对重采样患者2000次，取五折ΔC均值；不跨fold混合原始风险。这些名义95%区间不含重训练、内层选择、折间训练重叠、此前同队列探索或四组比较的多重性影响，不能仅凭最高均值选‘胜者’。

## EE/EC及大时间差EE

```text
     encoder       EC       EE  EE_gap>24
        gpfm 0.759142 0.581517   0.611223
kaiko-vitb16 0.755101 0.579516   0.605349
        keep 0.768273 0.587567   0.607078
   phikon_v2 0.740637 0.561711   0.583574
      uni_v1 0.760460 0.590447   0.613986
```

这里是逐折子组C-index均值，不是把患者对当独立样本。对应每折计数见pair_metrics.csv。删失患者不是永久无复发者。EE_gap>24指事件时间差超过24月，不代表可精确预测时间。

## alpha选择

```text
     encoder  fold       alpha  upper_boundary
      uni_v1     0 100000000.0            True
      uni_v1     1     10000.0           False
      uni_v1     2    100000.0           False
      uni_v1     3 100000000.0            True
      uni_v1     4     10000.0           False
        keep     0     10000.0           False
        keep     1     10000.0           False
        keep     2      1000.0           False
        keep     3     10000.0           False
        keep     4     10000.0           False
        gpfm     0      1000.0           False
        gpfm     1     10000.0           False
        gpfm     2     10000.0           False
        gpfm     3      1000.0           False
        gpfm     4      1000.0           False
   phikon_v2     0      1000.0           False
   phikon_v2     1     10000.0           False
   phikon_v2     2      1000.0           False
   phikon_v2     3      1000.0           False
   phikon_v2     4      1000.0           False
kaiko-vitb16     0      1000.0           False
kaiko-vitb16     1     10000.0           False
kaiko-vitb16     2      1000.0           False
kaiko-vitb16     3      1000.0           False
kaiko-vitb16     4      1000.0           False
```

到达1e8上界时只报告，不事后扩大网格，也不将其视为理论最优。旧Stage5的0.7197与本轮网格不同，不能直接当作新编码器提升的唯一参照；旧ABMIL也不与新结果直接构成公平架构比较。

## 验证与范围

完成750次内层拟合尝试和25次外层拟合。全部模型重载预测一致；50项保存Train/Val C-index复算。UNI患者均值与旧Stage5缓存逐值一致（atol1e-12）；内层角色、来源stat与代码/config/cohort检查通过。数值失败见inner_cv.csv，不静默填分。

没有微调编码器或训练新神经网络。原始H5、患者缓存/预测、joblib、病例来源信息与完整日志保持private；本实验不改变正式病理–CT教师/CT学生，也不把项目训练的单模态权重用于教师/学生初始化。
