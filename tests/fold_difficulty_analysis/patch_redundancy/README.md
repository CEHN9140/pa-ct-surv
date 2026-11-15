# 冻结模型：Patch冗余与Attention有效性

仅使用Stage6已完成的5个Raw UNI ABMIL最终Outer模型，每个模型只预测对应的
Outer Val。保持721例、251事件与原五折，不重新训练、不修改主模型或数据。

固定Full、Top-K、Bottom-K、Random-K，K为128/512/1024。若N≤K，保留全部。
Top/Bottom由Full Attention logits排序；与Softmax权重的数学排序一致，避免
权重下溢导致的人工并列。相同logits按原索引稳定排序，选择后恢复原patch顺序。
Random无放回，30次固定重复；每例/重复使用独立的seed123+病例ID+repeat随机流，
不依赖读取顺序、DFS标签；同一重复不同K共享随机排列。不能反复挑K或重复种子。

## 推理与验证

所有模型始终eval、requires_grad=False、无优化器。原ABMIL逐patch压缩和打分，
因此缓存H与logits后，对保留logits重新Softmax、加权池化，再通过原post/head，
等价于直接把这些patch送入同一模型。CPU/GPU测试及每Fold首例实际子集均核对
缓存与直接前向；不通过Full权重截断后省略重新归一化。

每例Full Risk先复现Stage6存档，折内Full C-index也必须一致。
核对原CSV、特征及权重hash；结束时确认模型参数和所有来源未变。
原始特征保持1024维全部patch，不L2、不使用EAGLE/CHIEF评分。

## 后台运行

在项目根目录运行一次，已有结果不覆盖：

```bash
setsid nohup env CUDA_VISIBLE_DEVICES=0 OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=4 \
  /home/gly001/cqj/pa_ct_surv/tests/16_ct_radiomics/.runtime/bin/python -u \
  /home/gly001/cqj/pa_ct_surv/tests/fold_difficulty_analysis/patch_redundancy/diagnose.py \
  > /home/gly001/cqj/pa_ct_surv/tests/fold_difficulty_analysis/patch_redundancy/inference.log 2>&1 < /dev/null &
```

```bash
tail -f tests/fold_difficulty_analysis/patch_redundancy/inference.log
```

## 输出

- `results/report.md`、`summary.csv`：各策略的五折Val C-index和与Full的Spearman。
- `fold_metrics.csv`：每Fold/策略/K/repeat指标；Random先逐次算C-index，再平均，
  不使用先平均Risk的集成模型。不同Fold的原Risk不混算。
- `strategy_contrasts.csv`：Top/Bottom减Random的平均ΔC、逐折变化及正增益Fold数。
- `attention_fold_summary.csv`、`attention_summary.csv`：Full归一化熵、
  N_eff=1/sum(a²)、N_eff/N、Top-K Mass、patch数及N≤K比例；单patch熵约定0。
- `replay_audit.csv`、`validation.json`：历史Full复算、直接前向等价及冻结状态。
- `private/fold_*_predictions.csv`、`private/fold_*_attention.csv`：病例明细仅留本地。

跨折SD与随机抽样SD分别报告：前者在每Fold对30次结果取均值后计算五折样本SD；
后者对每次重复的五折均值计算30次样本SD。随机抽样SD不是患者置信区间。
如果某组Risk为常数，Spearman标为未定义，不伪造数值。

Random稳定只能支持预测层面冗余，不等于组织无用；Top优于Random只能说明
当前冻结模型的筛选行为，不能证明Top重训会改善泛化。此处Outer Val此前已被
探索，不作为独立验证。Patch质量判断仍需图像/病理审查，本次不开展第二部分。

## 测试

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 \
  tests/16_ct_radiomics/.runtime/bin/python -m pytest \
  tests/fold_difficulty_analysis/patch_redundancy/test_patch.py -q
```

仅代码、说明和小体积聚合结果允许后续人工提交；日志、患者数据和模型不上传。
本任务不执行commit或push。
