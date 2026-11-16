# Random-128训练：严格匹配Stage6的容量不变对照

用户已指定Raw UNI完整特征、原ABMIL与Cox、原五折及Nested CV，只在训练时每次
随机无放回采128个patch，N≤128全部保留。不进行组织学标注，不安装HistoQC或改数据。

1. 新train_random.py：使用原ABMIL_RandomSample(k=128)，参数量/初始state与Full ABMIL
   相同（722178），从头初始化；复用Stage6 inner_roles/select_step/train_steps。
2. 原721例/251事件/五折及相同内层三折；seed123；Dropout0.25，Adam1e-4/WD5e-4，
   Cox风险集64尾部合并；固定Step[9,18,36,54,72,90,117,144,180]。
   内层仍以Full WSI评估/选Step。15个Inner+5个Fresh Outer，不用Outer Val选轮。
3. 同一次Cox VJP的两遍前向必须复现采样和Dropout，测试与直接联合图梯度一致；
   更新间种子来自原batch_schedule，重新采样。参数初始化相同不表示Dropout遮罩逐位一致。
4. 固定最终权重后，Train/Val均评估Full WSI；另用预设seed123、repeat0..29评估
   固定随机128个patch。repeat0单独列出，30次逐次计算C-index再平均，不平均Risk集成，
   不根据Outer Val挑策略/种子。eval禁用Dropout，用原patch_subset缓存规则重算Softmax。
5. report_random.py：与Stage6 Full主结果同患者配对ΔC、Gap、36/60月AUC，
   报告最终Full与随机评估（五折mean±SD、随机重复SD）；主比较仅Random训练→Full评估。
   2000次折内配对患者Bootstrap条件于已拟合模型，不覆盖重新训练/选择不确定性。
6. test_random.py：采样无放回/边界/eval全patch；两遍采样复现/梯度对照；初始化参数一致；
   固定推理重复确定；Nested隔离和自模型重载；汇总与差值方向。
7. 代码审查、CPU/GPU回归、真实完整UNI冒烟后后台启动，不等待20次完整训练。

所有新文件和结果独立保存在本目录，不改变Stage6/主流程/既有WSI图像及其hash。
病例预测、角色与模型仅存private/并Git忽略；仅代码和聚合结果白名单。此次不commit/push。

## 已核验

- GPU：本实验及Stage6回归共12项测试通过。
- 同次两遍采样与Dropout完全复现，梯度与直接联合Cox计算图一致；不同更新重采样。
- 新增报告测试覆盖配对顺序、ΔC符号、20次fit审计、随机重复SD及部分重复AUC缺失保留。
- 独立代码审查无阻塞；已修正随机重复均值跳过缺失值的报告问题。
- 真实GPU冒烟：14例、48783个完整UNI patch，3次Inner+1次Fresh Outer，
  Full/固定随机评估、选步和权重重载通过；冒烟指标不用于临床科学结论。
- 正式运行状态以results/status.json为准，正式结果须等待20次训练完成。
