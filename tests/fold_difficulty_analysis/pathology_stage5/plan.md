# Stage5：Raw UNI MeanPool + Ridge-Cox

按用户完整方案执行：仅新增这一种低容量模型，原721例/251事件/五折保持不变。
不调用带MLP的model/meanpool_cox.py，不重新提特征，不训练新的ABMIL，不触及教师/学生。

## 固定设计

全部1024维Raw UNI patch在float64中分块求均值，生成独立721×1024患者矩阵。
不做patch L2、PCA、patch筛选或监督特征选择。原始时间转为既有训练采用的
float32表示后在Cox中使用float64，确保与历史ABMIL的可比较患者对一致。
Outer Train内部event分层三折，seed=123；每次只在Inner Train拟合StandardScaler。
候选alpha=[0.1,1,10,100,1000,10000]，按三折C-index均值选；
均值在1e-12内相同则取更强惩罚。CoxPHSurvivalAnalysis显式Breslow，
n_iter=1000、tol=1e-9；任何收敛/数值警告使该拟合失败。
在Fold0首个FIT、alpha0.1溢出后、未查看任何Outer Val结果时修订失败处理：
所有90个Inner拟合仍尝试并记录，某alpha任一折失败则整体不可选，只在完整三折中选；
无完整候选或最终Outer拟合失败则中止。保留原网格、划分与求解器，未填补失败分数。
完整Outer Train重新拟合Scaler+Cox，保存5个最终模型及Train/Val、OOF预测。

软件目标为[−sum_event log partial likelihood + alpha/2*||beta||²]/n_fit。
这保留Cox目标，但相比深度模型还改变了池化、非线性容量、优化器和全风险集；
不是单独针对Attention的因果消融。历史Best/Last只作为明确标注协议的参照。

## 实现与验收

- [x] 先写测试：全部patch均值、折内Scaler、选参tie、警告失败、Outer Val标签隔离、模型重载。
- [x] ridge.py实现分块均值、固定三折选择、95次拟合（90内层+5最终），仅5个最终模型。
- [x] run_stage5.py严格核对患者、原标签/划分/历史预测；只读源文件，独立缓存与输出拒绝覆盖。
- [x] 对每折记录Train/Val C-index、Gap、36/60月IPCW AUC（训练估计删失分布）；
  保存五折mean±样本SD，AUC不可估计时注明原因，不筛除患者或填补数值。
- [x] 复用Stage4的pair_changes对照ABMIL Best/Last：risk Spearman、错误Jaccard、
  双方共享/独有错误、纠正与破坏，以及event时间×患者对类型、患者级共同错误。
  shared wrong为双方credit=0；ties另列，ΔC使用完整半分规则复算。
- [x] report_stage5.py输出协议说明、逐折及均值表、错误重合/特征信息解释边界和PNG/PDF。
- [x] 独立代码审查通过；Stage5初版9项测试、与Stage4联合17项测试通过。
- [x] 初次运行PID2681025因Fold0内层alpha0.1数值溢出正常中止，未产生任何Outer Val结果。
- [x] FIT-only诊断确认其余5个预设alpha在首个同一FIT可收敛；修订失败候选规则后11项测试通过。
- [x] 修订规则经独立审查；与Stage4联合19项测试通过。重启完整运行PID2684751，
  Fold0的alpha0.1三折失败均被记录并整体排除，进程继续拟合其他原定候选。
- [x] 全量2,222,768 patch生成721×1024均值，与实验07独立全patch缓存逐位相同。
- [ ] 等待真实五折训练完成并复核模型、预测、95次拟合与全部原始特征哈希。

## 输出与边界

默认results/中保存公开聚合CSV、report.md、validation.json和图。
private/保存缓存、特征来源哈希、患者角色、模型joblib、Train/Val及OOF预测和病例错误表，
均加入Git忽略；正式模型、原始特征、旧实验及Stage1–4不覆盖。此次不commit/push。
不把OOF风险直接跨Fold合并计算总体C-index（折间风险尺度/基准不同）；主要报告五折指标。
不从相关风险/共享错误推断病理缺乏信息，也不因结果不理想扩展alpha网格或增加新模型。
