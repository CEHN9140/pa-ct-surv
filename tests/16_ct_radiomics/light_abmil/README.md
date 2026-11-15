# 实验16：轻量ABMIL容量对照

使用原始完整UNI1024维WSI特征，不做L2、不使用EAGLE筛选。仅缩小原ABMIL的宽度，
主流程模型和实验16原有Radiomics/教师结果不变。

| 部分 | 完整ABMIL（Stage6） | 轻量ABMIL |
|---|---|---|
| Patch投影 | 1024→512 | 1024→128 |
| Attention | 512→128→1 | 128→32→1 |
| Attention加权池化 | 512维 | 128维 |
| 池化后投影 | 512→256 | 128→64 |
| 风险头 | 256→1 | 64→1 |
| 可训练参数 | 722,178 | 143,682（减少80.104%） |

投影及池化后隐藏层为ReLU、Dropout；Attention为Tanh、Dropout后输出权重。
Dropout统一0.25，风险输出不加Sigmoid。仍调用原ABMIL和事件平均Cox损失。
原始全部patch在训练和评估中均保留，改变的是网络容量。

## 严格匹配的Nested CV

沿用Stage6的721例、251事件及原五折，数据为
`data/trident/features_uni_v1/all_label_roi64.csv`。每个外层训练集内部做相同的
三折分层CV，在预先固定的Step `[9,18,36,54,72,90,117,144,180]` 选择平均
内层C-index最高的Step，平分取早步。从头初始化，在完整Outer Train训练所选
步数，最后评估Outer Val一次。共15次Inner和5次Outer训练。

Adam lr=1e-4、weight_decay=5e-4、seed=123、Cox风险集64人，尾部合并。
复用Stage6的两遍前向/反向、患者顺序和折内角色，不使用已训练完整ABMIL初始化。
不同网络宽度的Dropout张量维度不同，不能声称神经元遮罩逐位一致。

参照为已完成的 `tests/fold_difficulty_analysis/pathology_stage6/results/`。
训练前校验参照配置、共享代码、患者、标签、划分和原始特征哈希。
v7的EAGLE特征日志不作为本次容量的严格对照。

## 后台运行

在项目根目录运行一次，默认已有结果时拒绝覆盖：

```bash
setsid nohup env CUDA_VISIBLE_DEVICES=0 OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=4 MPLCONFIGDIR=/tmp/pact_matplotlib \
  /home/gly001/cqj/pa_ct_surv/tests/16_ct_radiomics/.runtime/bin/python -u \
  /home/gly001/cqj/pa_ct_surv/tests/16_ct_radiomics/light_abmil/train_light.py \
  > /home/gly001/cqj/pa_ct_surv/tests/16_ct_radiomics/light_abmil/training.log 2>&1 < /dev/null &
```

```bash
tail -f tests/16_ct_radiomics/light_abmil/training.log
```

## 输出与判断

`results/metrics.csv` 保存逐折结果；`comparison_summary.csv` 保存轻量与完整模型的
五折mean±样本SD，含Train/Val C-index、Gap和36/60月IPCW AUC。
`fold_comparison.csv`、`paired_comparison.json` 保存轻量减完整模型的配对差值，
`report.md`、`capacity_comparison.png/pdf` 自动生成最终汇总。
`inner_curves.csv`、`step_selection.csv`、`fit_audit.csv` 记录选步及隔离审计；
`status.json` 记录运行状态，完成后才生成 `validation.json`。

病例预测、CV角色和模型保存在 `results/fold_*/private/` 与 `results/private/`，
均被Git忽略，日志也不上传。代码和不含患者明细的汇总结果允许后续人工提交。
本任务不自动commit或push。

Gap缩小本身不能证明泛化改善，需同时看Val、AUC、逐折一致性及配对区间。
患者Bootstrap条件于已训练模型，不包含重新训练、选步及前期探索的不确定性；
这是内部交叉验证，不是独立外部验证。

## 检查

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 \
  tests/16_ct_radiomics/.runtime/bin/python -m pytest \
  tests/16_ct_radiomics/light_abmil/test_light.py -q
```

测试覆盖真实维度/参数量、Raw输入直达投影、完整patch、eval Dropout、Cox梯度、
恢复权重、外层标签扰动不改变选步/训练权重，以及五折汇总与配对差值方向。
