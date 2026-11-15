# 实验16子实验：Raw UNI轻量ABMIL

用户要求将Hidden512→128、Post256→64，使用原始UNI全WSI、不做L2、不用EAGLE Top25，
Cox目标保持不变，检验约80%参数缩减能否改善泛化。独立置于实验16，不改原模型或旧结果。

用户已明确选择：Attention128→32，并沿用Stage6严格Nested CV公平对照（20次训练）。
实现仍可由--attention_dim保留128，
参数722178→143682，减少80.104%；Attention保留128时156162，减少78.376%。
不把后者写成精确80%。输入1024维不改变。全部原始patch在Train/Val都保留。

## 协议

默认采用Stage6同一严格Nested：原721例/251事件/五折，同样event分层内层三折，
seed123，Raw UNI1024、Dropout0.25、Cox64合尾、Adam1e-4/WD5e-4。
候选Step固定[9,18,36,54,72,90,117,144,180]，按相同Step三折平均Inner C-index选步，
平分取较早步；外层从头初始化恰好训练选定步，只评估一次Outer Val。
共15个Inner+5个Outer模型，不使用已训练完整ABMIL初始化。
与Stage6完整Raw UNI ABMIL作同协议对照，另记录配对患者Bootstrap区间。
不将v7 EAGLE1280结果作为参数容量的严格对照，也不新增完整ABMIL训练。

## 实现

- [x] light_model.py只使用原ABMIL构造参数，固定1024→128、Attention32、池化128→64→1。
- [x] train_light.py复用Stage6 train_steps、inner_roles、select_step；独立外层编排，
  同一批次顺序及CV角色，训练参数不变，保存两类模型的参数量/真实维度。
- [x] compare_light.py对齐Stage6完整模型患者与指标，逐折差值、五折mean±SD、
  36/60月AUC、Gap及条件配对区间；不将输入/选轮差异冒充容量收益。
- [x] TDD验证维度/参数数量/全部patch、Dropout eval、Cox反传、无Outer Val选参及恢复权重。
- [x] 独立审查：最终复核无阻塞问题，共享代码hash/config记录等修正已落实。
- [x] GPU回归：本子实验与Stage6共13项测试通过；CPU为5通过、1因无GPU跳过。
- [x] 真实数据冒烟：14例、48,783个完整UNI patch；3次Inner+1次Outer，
  GPU反传、选步、最终评估和重载一致性全部通过（冒烟指标不作为科学结果）。
- [x] 正式五折20次训练已后台启动；PID与运行状态保存于results/status.json。
- [ ] 完成20次训练，自动生成与完整ABMIL的对照报告。

结果/日志/Checkpoint都在本子目录独立保存；private/不上传。
现有实验16 Radiomics/教师数据、主流程model/path_train/train_path.sh保持不变。
此次不commit/push。改变架构会改变Dropout张量维度：相同seed/患者批次，
不声称不同宽度网络的神经元Dropout遮罩逐位相同。
