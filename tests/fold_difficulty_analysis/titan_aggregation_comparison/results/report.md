# 同CONCHv1.5 Patch：MeanPool与冻结TITAN聚合对照

主对照B/C：同721例/251事件、原五折、同一20×512 Patch输入、同一CONCHv1.5特征和坐标。B对全部向量float64均值，C使用官方冻结TITAN encode_slide_from_patch_features(no_proj=True)；不加Patch/Slide L2，不训练CONCH/TITAN或ABMIL，不筛Patch或患者。

固定官方TITAN revision=dac6773d9961cfc75503440676ff157a2c6e8d2e，模型权重保存在用户指定目录，运行时仅离线读取。官方输入规范与接口来源：[MahmoodLab/TITAN](https://huggingface.co/MahmoodLab/TITAN)。TITAN推理为eval/inference_mode+CUDA float16 autocast；源向量只按数值转float32以匹配官方网格累加器，未归一化。Grid冲突/非法几何直接中止，不静默合并或删病例。重复真实推理核验见smoke_validation.json。

## 五折结果

```text
       arm  train_cindex  val_cindex   val_sd      gap    auc36    auc60  auc36_folds  auc60_folds
conch_mean      0.744696    0.729569 0.057797 0.015127 0.763318 0.745088            5            5
     titan      0.758878    0.741009 0.052326 0.017868 0.781778 0.757234            5            5
```

## 配对差值 TITAN − CONCH MeanPool

```json
{
  "delta_cindex": 0.011440091422274971,
  "lower": -0.005364098020464474,
  "upper": 0.028142909324294734,
  "valid_repeats": 2000,
  "positive_folds": 2
}
```

```text
 fold  conch_mean    titan  delta_cindex
    0    0.676248 0.735483      0.059235
    1    0.761986 0.759909     -0.002076
    2    0.771868 0.795246      0.023378
    3    0.779905 0.758863     -0.021043
    4    0.657840 0.655546     -0.002294
```

本次五折平均差值+0.0114，2/5折正向；该名义区间包含0，尚不能确认稳定改善。不能仅凭均值改善认定预训练聚合已解决泛化问题。

2000次每折内配对患者bootstrap，取五折ΔC均值，不跨fold混合未校准风险。名义95%区间条件于已训练模型，不包含重新训练、内层选择、训练折重叠或此前同队列探索的不确定性，不是独立外部验证。

## 验证集EE/EC

```text
       arm       EC       EE  EE_gap>24
conch_mean 0.771882 0.594343   0.619945
     titan 0.780082 0.616224   0.650009
```

子组数值为五折均值；每折患者对计数见pair_metrics.csv，患者对不是独立样本。EE_gap>24表示两个事件时间相差超过24个月。

## 下游协议与边界

两组均为FIT-only StandardScaler + Ridge-Cox，Outer Train内event分层3折、seed123，alpha=[0.1, 1, 10, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000]，Breslow/n_iter1000/tol1e-9；平分取强惩罚。每候选3折必须全成功；不完整候选不可选。两组各150次inner尝试+5次outer重新拟合，Outer Val仅最终评价。

上下文UNI参考来自上一轮同网格MeanPool+Ridge（Val C=0.7201），但其TRIDENT 20×256视野不同，只能作背景，不能把C−UNI解释成纯聚合收益。本轮主要解释B/C；二者同时改变聚合计算形式与外部预训练状态，不能将差值单独归因于空间建模或预训练某一因素。

没有自动新增学习曲线、D组ABMIL、TITAN微调或教师训练。当前项目教师/学生初始化约束不变，未将项目单模态权重用于教师/学生。代码、公开指标可发布；权重、特征、患者缓存/预测/模型、原始日志与第三方gated模型代码全部保留本地。
