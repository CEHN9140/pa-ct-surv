# 实验08：GL队列单模态病理生存预测

## 当前默认：154例，事件＋时间分层五折

已补齐之前缺特征的1例。源表158例中，只排除4例status=2；纳入154例：67个event=1、87个event=0。按patient_id与TRIDENT UNI H5文件名精确匹配，单模态病理不要求CT存在。status=1按事件、0按删失使用，临床终点尚不标记为DFS或OS。

survival_time始终保持原始数值，单位未确认，暂不计算月份AUC。预测导出survival_time和event，KM标注原始时间单位。

## 新划分规则

先按event=0/1分组，在各组内依据观察时间25%、50%、75%分位点划分时间区间。每个区间内随机打乱患者，依次轮流分配到五折，并跨区间延续轮转位置。

- 五折总人数、每类event人数、每个event×时间区间的人数，最大差均为1。
- 不固定划分seed；本次只随机生成一次，保存后训练与再次准备均复用。
- 不读取模型预测或验证成绩来筛选划分。
- 分期单独汇总，不作为这次分层约束；时间区间平衡也不保证均值、中位数或全部特征分布完全一致。
- 这里的全队列时间分位点仅用于划分。NLL的时间切点仍单独依据每折训练事件计算。

本次固定时间切点：event=0为478、598、793；event=1为350、528、739.5，均为原始单位。

| Fold | 验证人数 | 验证事件数 | 事件时间中位数 | 删失时间中位数 |
|---|---:|---:|---:|---:|
| 0 | 31 | 13 | 528 | 655.5 |
| 1 | 31 | 13 | 579 | 591 |
| 2 | 31 | 14 | 550.5 | 596 |
| 3 | 31 | 14 | 466 | 567 |
| 4 | 30 | 13 | 500 | 644 |

当前数据目录为data_154_eventtime，本实验仅保留154例事件＋时间分层五折及对应结果。

## 训练设置

默认ABMIL、1024维UNI、Dropout0.25、Adam lr1e-4、weight_decay5e-4、30 epochs、patience10、Cox batch16。训练seed默认123，在启动及每折开始时设置，不在bash中设置。

每折训练123或124名患者，尾部并入后风险集为16×6＋27或16×6＋28，每epoch更新7次。全部patch参与ABMIL，沿用主流程的训练与eval模式，不自动启动训练。

原始UNI与L2共用同一份154例CSV及五折：

- train.sh使用原始UNI。
- train_l2.sh对每个patch特征沿最后一维计算normalize(p=2, dim=-1, eps=1e-12)。训练、验证和最终评估均应用；原H5不变。
- 新run tag以gl154_eventtime开头，L2额外包含uni_v1_l2；两种输入的输出分别保存。

后台启动原始UNI：

```bash
bash tests/08_gl_pathology/train.sh
```

后台启动L2对照：

```bash
bash tests/08_gl_pathology/train_l2.sh
```

分别查看日志：

```bash
tail -f tests/08_gl_pathology/logs/gl154_eventtime_uni_v1_abmil_cox_dropout0.25_epochs30_coxbs16_lr1e-4_wd5e-4.log
```

```bash
tail -f tests/08_gl_pathology/logs/gl154_eventtime_uni_v1_l2_abmil_cox_dropout0.25_epochs30_coxbs16_lr1e-4_wd5e-4.log
```

准备数据的入口默认指向新目录。重复执行会校验并复用保存的划分：

```bash
python tests/08_gl_pathology/prepare.py
```

train.py仍支持--survival_loss nll或cox_rank。重载L2模型时同时传入--eval_only --l2_normalize。

## 输出与核对

- data_154_eventtime/all_label.csv：原表字段、特征路径、原始event/time、time_group及冻结五折。
- data_audit.csv、excluded_cases.csv：158条源记录审计；本次仅排除4个非二值status。
- preparation.json：原始文件、队列哈希、特征指纹、划分规则和时间分位点。
- fold_distributions.csv、time_stratum_counts.csv、stage_counts.csv：逐折时间/事件、分层计数及分期汇总。
- logs/、checkpoints/、results/下的运行均带gl154_eventtime标签。
- results/<run_tag>/保存配置、数据快照和含逐折/mean±样本标准差的metrics.csv；fold_*/保存原始单位预测、KM、HR和log-rank。风险分组阈值取训练风险中位数，不输出月份AUC。

数值与合成CPU回归检查：

```bash
python tests/08_gl_pathology/test_experiment.py
```

检查覆盖事件/时间分层计数、病例与标签、L2和零向量、三种损失及L2 Cox的五折训练/重载、NLL训练内切点、输出隔离及provenance保护。没有启动真实队列的新划分训练，也未验证实际GPU显存需求。

五折验证集参与最佳epoch选择，因此成绩属于内部交叉验证，不能视为独立测试性能；训练与验证差距缩小也不代表过拟合已完全解决。

## KEEP特征对照

KEEP使用`/home/gly001/cqj/data/yixianai/GL/pathology/trident/20x_256px_0px_overlap/features_keep/`，
原始输入为每例`[N,768]`，不额外做L2归一化，不通过1024维中间适配器。
ABMIL自动使用768→512投影，后续Attention与风险头及Dropout设置与UNI实验相同。

`data_154_eventtime_keep/`从已有`data_154_eventtime/`复制患者、事件、原始时间和fold，
只替换实际读取的pa_path及feature_dim；准备过程逐例核对UNI和KEEP的patch数量、坐标与顺序。
不重新划分、不补入参考队列之外的病例，也不静默排除缺失KEEP的患者。
准备配置记录原参考CSV哈希，训练时会核对参考名单仍未改变。

本机数据准备完成后，直接在项目根目录执行：

```bash
bash tests/08_gl_pathology/train_keep.sh
```

脚本内含nohup和日志保存，默认ABMIL＋事件平均Cox、Cox batch16、Dropout0.25、Adam lr1e-4、
weight_decay5e-4、最多30轮、早停10，Python训练seed默认123。
新标签为`gl154_eventtime_keep_abmil_cox...`，不覆盖UNI/UNI L2的日志、checkpoint及结果。
仍输出原始时间单位的Train/Val loss、C-index、KM、HR、log-rank及五折mean±样本标准差，不计算月份AUC。

重新部署时先准备KEEP数据表（只准备，不训练）：

```bash
python tests/08_gl_pathology/prepare_keep.py
```

完整合成CPU检查：

```bash
/home/gly001/.conda/envs/UNI/bin/python -m unittest discover -s tests/08_gl_pathology -p 'test_*.py' -v
```
