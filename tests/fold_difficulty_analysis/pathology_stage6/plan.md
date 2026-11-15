# Stage6：Raw UNI ABMIL严格Nested CV

按用户指定方案，仅补齐这一个最终诊断；不新增模型/Loss/L2实验。
保留原721例、251事件、五折；每个Outer Train内部采用与Stage5逐位相同的
event分层三折(seed123)。只读Stage5角色及预测，绝不载入其训练权重初始化网络。

## 预先固定协议

Raw TRIDENT UNI1024全WSI，原ABMIL、Dropout0.25、Cox风险集64、尾部并入，
Adam lr1e-4、耦合WD5e-4、betas(0.9,0.999)、eps1e-8，无调度器。
候选Optimizer Step固定[9,18,36,54,72,90,117,144,180]。
每个Inner Fit从seed123全新初始化，完整训练到180步，仅固定点记录Inner HELD。
按相同步数的三折平均HELD C-index选一个Step，1e-12内平分取较早者。
不选择各折最佳Epoch的均值/中位数，不使用Outer Val选轮。
完整Outer Train再次从头初始化，恰好训练选定Step，保持相同更新预算，
不将内层步数换算为外层Epoch。之后仅评估一次Outer Val并输出完整Train/Val指标。
总15次内层+5次外层训练。Inner HELD全属于对应Outer Train。

复用Stage2联合Cox的两遍VJP/Dropout重算和Stage3预生成批次/逐步共同种子；
完整patch保留，避免同时存64份WSI图，不修改Cox目标。批次/Dropout流只由args.seed派生。

## 实施与核验

- [x] TDD：按三个相同Step取均值而非每折独立选轮；平分早步；内外隔离；
  固定步外层不调用验证；新初始化指纹一致；真实参数配置及模型重载。
- [x] nested_train.py：固定Step训练内核、三折角色对齐Stage5、外层一次评估、进度/审计。
- [x] nested_report.py：嵌套ABMIL与嵌套Ridge逐折指标、均值±样本SD；
  基于同患者五折配对的ΔC及2000次患者Bootstrap，保持Fold并条件于已训练模型。
  报告风险尺度不同，不能将跨Fold OOF风险直接混算C-index。
- [x] 完整源哈希、私有病例/模型边界、独立代码审查、CPU/GPU测试及真实特征冒烟。
  Stage6+Stage2 GPU联合15项通过；14份真实完整UNI WSI的三内层+新外层短程验证、
  选步/仅一次Outer Val/模型重载通过。15个真实180步Inner计划最少每风险集12个事件。
- [x] 后台启动20次训练（PID2993830）；已确认Fold0 Inner0的Step9 HELD评估成功，
  FIT384/HELD192，仅使用Outer Train内部数据，Outer Val尚未使用。
- [ ] 等待完整15次内层+5次外层训练完成，自动输出配对比较及来源复核。

## 结果边界

results/保存公开inner_curves.csv、step_selection.csv、metrics.csv、comparison_summary.csv、
paired_comparison.json、report.md、validation/status及PNG/PDF；private/保存20个末步模型、
内层候选预测、最终Train/Val及OOF预测和患者/来源角色，不发布大文件或患者ID。
禁止覆盖已有目录；不自动commit/push，不修改Stage1–5或正式模型/训练。
主比较仅当前嵌套Ridge，不以历史Outer Val选轮ABMIL作公平对照。
Bootstrap不包含重训练/选参/此前探索不确定性，仍属内部验证。
Ridge全部选择网格上界10000是当前参照的边界，本次不为此扩展网格。
无论结果怎样，本轮均不自动增加Stage7。
