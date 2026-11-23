# 同CONCH特征的从头ABMIL对照

新增ABMIL，与上一轮CONCH MeanPool+Ridge及冻结TITAN+Ridge比较。读同721例、588679个20×512 CONCHv1.5 Patch，向量及坐标逐例与TITAN输入hash一致，无额外L2/采样。只训练新初始化的ABMIL，不重训编码器或旧Ridge参照，不改主流程。

复用Stage6的3inner×5outer固定更新步选模：内层均训练180更新，在固定9个候选点取3HELD均值，选定后重新初始化Outer Train模型并训练固定预算，Outer Val最终评价。参数与完整设置见config.json/PLAN.md；不是外层Val选最佳epoch。

从仓库根目录，GPU运行：

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .radiomics_env/bin/python -u tests/fold_difficulty_analysis/conch_abmil_comparison/run.py
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .radiomics_env/bin/python -m pytest -q tests/fold_difficulty_analysis/conch_abmil_comparison/test_compare.py tests/fold_difficulty_analysis/conch_abmil_comparison/test_run.py tests/fold_difficulty_analysis/conch_abmil_comparison/test_report.py
```

结果在results/；private下所有模型、源cohort快照、内层/外层患者预测及角色禁止上传。源码依赖既有Stage6训练助手，不复制其训练实现。报告提供三组总体和EE/EC差值的配对患者Bootstrap；子组及多对比区间为探索性、未经多重比较校正，不把患者对当独立样本，也不归因到单独Attention机制。

本轮ABMIL只在固定LR/WD/Dropout及预设更新范围内选择训练预算，并未搜索其全部超参数。因此结果不能当成ABMIL的最优性能上限。TITAN使用坐标构建空间表示，ABMIL/MeanPool则为无序bag聚合；比较识别的是完整方案效果，不能隔离空间信息与预训练状态的贡献。

## 本次结果

三组Val C-index：MeanPool+Ridge 0.7296、从头ABMIL 0.7399、冻结TITAN+Ridge 0.7410。ABMIL相对MeanPool为+0.0104（名义配对95%区间[-0.0032,+0.0247]）；相对TITAN为-0.0011（[-0.0156,+0.0131]）。两项均未确认稳定差异，也不构成等效性证明。EE均值：MeanPool0.5943、ABMIL0.5966、TITAN0.6162；本轮相关子组区间均跨0。不能据此认定TITAN已解决泛化问题或从头ABMIL无法有效利用特征。

15次内层+5次外层从头训练完成；选定步数为90/72/54/72/90。20个初始状态和预算、5个选步、30项保存预测指标核对通过，7项测试通过。完整比较见[报告](results/report.md)。没有追加其他结构、损失或学习曲线训练。
