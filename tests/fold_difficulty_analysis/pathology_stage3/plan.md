# Stage3：输入尺度与优化轨迹验证

用户已指定完整实验设计；本计划落实到独立诊断目录，不修改主流程或Stage2。
仅Fold2、4，原721例/251事件保存划分，完整Outer Train及全WSI UNI1024。

## 固定协议

A=Raw/W/lr1e-4；B=/40/W/lr1e-4；C=/40/40W/lr1e-4；
D=/40/40W，仅compression.0.weight学习率4e-3。
首层bias不放大、学习率1e-4。其余网络初始化逐位一致，seed123。
原ABMIL、Dropout0.25、原事件平均Breslow Cox、64风险集、尾部并入，
Adam WD0、betas(0.9,0.999)、eps1e-8；关闭早停及调度器。
每组恰好60更新，不因验证结果改变训练预算。每折9更新/完整epoch。

## 实现与验收

- [x] 先写测试：四组初始化/优化器分组；共享顺序和合尾；评估不扰动RNG；
  原始全图和Stage2两遍重算的多步Adam状态一致；固定点预测及完成状态。
- [x] train_scale.py：加载完整病理特征至只读CPU缓存，四组交错训练同一个Cox风险集。
  使用Stage2的joint_gradients和compensated_model，逐步种子由唯一seed123派生，
  每步所有患者使用连续Dropout序列，重算时逐位复现。
  预先生成患者顺序及种子，仅用于共同随机数控制，不复用历史训练的RNG流。
- [x] scale_eval.py：原ABMIL forward hooks统计Attention归一化熵、两级表示方差；
  完整Train/Val全部风险组成单一Cox，仅eval模式，保存固定点患者预测。
- [x] scale_analysis.py：所有条件结束后，比较A/B/C/D固定点C-index、Cox、Risk SD、
  表征方差与函数RMSE/Spearman；按用户预定判断规则解读，不用Val挑选模型。
  Python绘制每折Train/Val四项曲线及函数差异，粗线、大字体英文PNG/PDF。
- [x] 源数据/模型状态完整性检查、合成小数据CPU/GPU测试、独立代码审查。
  Stage2+3 GPU共16项通过。6份真实完整UNI WSI的4组单步、权重重载及出图通过。
- [x] 后台启动真实实验（PID2649683），状态running。
- [x] 完整训练池Fold2的Step0 Train/Val初始等价与四组首步更新通过。
- [ ] 等待两折各组全部60步完成，自动统一分析。

## 文件与输出

train_scale.py负责共享批次及训练；scale_eval.py负责指标/差值；scale_analysis.py负责完成后报告/图；
test_stage3.py验证关键协议。results/拒绝覆盖，状态先running，全部成功才completed。
metrics.csv、gradient_update.csv、function_differences.csv、batch_schedule.csv只含汇总。
private/保存患者级预测与每个固定步checkpoint，不发布。manifest.json记录源哈希、
CPU/GPU配置与初始权重哈希；原始文件读取前后SHA256校验。

输出的Train指标均为固定eval，与在线训练指标分开；batch_loss只用于训练诊断。
60更新是早期机制诊断，不能声称解释后期过拟合。此次不训练多模态、不commit/push。
