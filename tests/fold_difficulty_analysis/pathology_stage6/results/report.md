# Stage6：严格Nested ABMIL与Nested Ridge-Cox

原721例/251事件、同一五折、Raw UNI1024全WSI；三折Inner患者角色与Stage5逐位相同。
ABMIL结构、Dropout0.25、Cox64合尾、Adam lr1e-4/WD5e-4/betas(0.9,0.999)/eps1e-8不变。
每次网络从seed123独立初始化，未载入任何项目训练模型作初始化。
内层候选Step预先固定[9,18,36,54,72,90,117,144,180]，所有15个Inner训练到180步。
按每个相同Step的三折HELD平均C-index选一步，1e-12内平分取较早者。
Outer Train从头训练恰好选定的Step，不换算Epoch、不早停、没有调度器；结束后仅评估Outer Val一次。
完整原始patch保留，使用已验证的两遍联合Cox VJP；批次与Dropout随机流仅从唯一seed派生。

## 同外层患者、内层选参的五折结果

|模型|Train C-index|Val C-index|Gap|36月AUC|60月AUC|
|---|---|---|---|---|---|
|ABMIL Nested|0.7976 ± 0.0225|0.7332 ± 0.0456|0.0644 ± 0.0582|0.7703 ± 0.0238|0.7439 ± 0.0514|
|MeanPool Ridge Nested|0.7496 ± 0.0124|0.7197 ± 0.0543|0.0299 ± 0.0665|0.7559 ± 0.0409|0.7273 ± 0.0742|

均值±样本SD跨5折。AUC采用对应Outer Train估计删失分布，失败原因保留在metrics.csv，不删除病例或填补。

|Fold|选定Step|ABMIL Val|Ridge Val|ΔC|
|---|---|---|---|---|
|0|54|0.6716|0.6574|+0.0142|
|1|72|0.7492|0.7508|-0.0017|
|2|54|0.7769|0.7641|+0.0129|
|3|90|0.7685|0.7625|+0.0061|
|4|90|0.6997|0.6637|+0.0361|

平均ΔC(ABMIL−Ridge)=+0.0135，配对患者Bootstrap名义95%区间[-0.0010,+0.0282]，4/5折为正。

## 如何判断

本轮比较消除了用Outer Val选择ABMIL训练轮次的直接乐观偏差；历史0.7452不作为本次公平比较结果。
若出现跨折一致且区间支持的正差值，支持当前可训练ABMIL整体流程有额外预测收益；
但它同时包含Patch投影、Attention和非线性头，不能将收益单独归因于Attention。
若差值区间跨0，不能据此证明两者等效；只能说当前样本未确认增益。
Ridge五折均选原网格上界10000，比较仅针对Stage5已完成的方案，不代表所有线性模型的最优性能。
两类模型均在内部Outer Train选参，但模型类别、惩罚、优化与风险集大小仍不同。

## 不确定性和边界

Bootstrap仅在每个Outer Val内配对重采样患者，然后取五折ΔC均值；不跨Fold混合未校准风险。
区间条件于已训练模型，不包含重训练、内层选择、折间训练重叠及此前同队列多次探索的全部不确定性。
严格Nested CV仍是内部验证，不是独立外部测试。没有根据Outer Val调整候选范围、学习率或模型。
本轮结束后不自动新增结构或Stage7诊断。正式病理–CT教师和CT-only学生方法未被改变。

## 文件

inner_curves.csv/step_selection.csv记录全部135个Inner固定点评估和45个候选均值；
fit_audit.csv记录20次新初始化、预算与评估调用；metrics.csv记录外层唯一评估。
comparison_summary.csv、fold_comparison.csv、paired_comparison.json及图记录嵌套比较。
模型、内层HELD预测、最终Train/Val及OOF病例预测均本地保留，不发布。
