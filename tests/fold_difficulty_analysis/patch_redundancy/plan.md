# Patch冗余与Attention有效性：冻结Stage6推理诊断

按用户提供的第一部分方案实施，不新增训练，不修改模型、Cox、数据或原划分。
独立入口diagnose.py、汇总report.py、测试test_patch.py；结果在本目录results/。

- Stage6五个Raw UNI Outer最终模型，仅预测各自Outer Val；原721例/251事件。
- Full；Top-K、Bottom-K、Random-K，K固定128/512/1024；不足K全部保留。
- Top/Bottom按Full模型未归一化Attention logits排序（与Softmax排序等价），
  保留原patch顺序，对保留logits重新Softmax，再经过原pool/post/risk head。
- eval禁用所有Dropout，无梯度、无优化器；缓存逐patch投影和logits只为减少重复计算。
  测试缓存与原模型在真实删patch上的直接前向数值一致。
- Random无放回；30次预先固定重复；每个患者/重复独立种子由seed123、病例ID、
  repeat确定，病例读取顺序不影响抽样，同次重复的不同K共享同一随机排列。
- 必须先复现保存的Full患者Risk和折内C-index；核对CSV、特征、代码与权重hash。
- 每Fold/策略/K/repeat算C-index及与Full的Spearman；不将不同Fold原Risk混算。
- 报告跨折mean±样本SD；Random额外报告30次五折均值的SD，区分划分变异与抽样变异。
- Full Attention：归一化熵、1/sum(a²)有效patch数及其比例、Top-K Mass、实际保留数。
  单patch熵约定0；Spearman若风险常数则保存缺失与原因，不伪造相关系数。
- 私有病例预测与Attention明细仅留本地；公开聚合CSV、说明、来源hash和状态。
- 本次不做WSI质量审查、不对齐CHIEF、不新增Top256训练、不推断生物学/因果机制。
  这是反复探索过的Outer Val上的冻结推理干预，无法证明重训Top-K会提升泛化。

验证：先测试缺失实现，随后CPU/GPU等价/确定性/边界/汇总测试；独立代码审查；
最后后台运行五折推理，输出实际报告。测试和临床诊断结果分别记录。

## 实施核验

- CPU：7项通过、1项CUDA跳过；GPU：8项全部通过。
- 覆盖真实ABMIL缓存/直接删patch等价、参数不变、随机嵌套及病例稳定性、极端熵、
  N≤K、常数Risk、完整折内复算/只读取Outer Val、统计汇总及标签不一致拒绝。
- 独立只读审查无阻塞问题；重复登记来源hash被覆盖的审计小缺口已修正为变化即报错。
- 正式后台推理的实际状态见results/status.json；未产生新的训练模型。
- 正式五折已完成：721例、2,222,768个patch、485组折/策略/重复指标。
  Full Risk最大复现误差为0；缓存与直接删patch前向最大误差3.58e-7。
  独立重新读取全部病例预测复算485组C-index及Spearman，全部一致，来源/代码hash未变。
