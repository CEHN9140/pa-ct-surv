# 实验09：CPTAC-LUAD单WSI病理OS预测

## 已确认的设计

- 一名患者随机选择一张WSI，保存后冻结；UNI、UNI L2、KEEP使用同一名单，不按模型结果或生存结局选片。
- 患者105名、死亡事件23名，使用OS_event（1事件、0删失）和OS_days（天），不用多类别OS列作事件标签。
- 保留PathoBench TSV的患者级五折，将其test成员用作本实验验证集。沿用逐轮验证、最佳checkpoint及早停10，性质为内部交叉验证，不宣称官方独立测试结果。
- 统一ABMIL＋Cox，UNI/UNI L2为1024维，KEEP为768维；Cox batch16、Dropout0.25、Adam lr1e-4、weight_decay5e-4、最多30轮；训练seed默认123，每折复位。
- 物理batch为一名患者，Cox风险集累计16人，尾部并入。每折84人时为16、16、16、16、20。
- UNI L2对每个patch沿特征维归一化，训练/验证一致。KEEP保留编码器原始输出。
- 输出逐折Train/Val loss、C-index、KM、HR、log-rank及24/36月时间AUC。训练及预测时间保留天；仅在AUC计算中按365.25/12天每月换算。60月不作为本实验指标。

## 实施计划

1. 检查患者、切片、TSV三表标签完全一致，所有切片两个编码器及坐标均可读取；从每名患者的候选切片中随机选择一张，保存选择及五折。
2. 单独实现数据层，冻结源文件哈希与所选特征指纹；重复准备、训练及eval_only均复用该名单，读取时不重新抽样。
3. 复用现有Pa_Model、train_path、DataLoader及生存指标实现，适配OS字段/单位，原模型与其他实验不改动。
4. 提供单组和三组并行后台脚本。用临时合成数据验证选择、患者分组、三种输入及五折训练/重载、结果隔离和失败状态。
5. 准备真实数据并做只读检查，真实GPU训练由用户启动。

## 实际数据准备结果

从313张候选WSI中，为105名患者各随机选定一张。105张共777,622个patch，中位数4,750，范围749～21,932。所选UNI和KEEP特征均已完整读取，无非有限值；每例坐标在两种编码器中顺序一致。五折真实病例的UNI L2向量范数抽查约为1。

| Fold | 训练患者/事件 | 验证患者/事件 |
|---|---|---|
| 0 | 84 / 18 | 21 / 5 |
| 1 | 84 / 18 | 21 / 5 |
| 2 | 84 / 18 | 21 / 5 |
| 3 | 84 / 19 | 21 / 4 |
| 4 | 84 / 19 | 21 / 4 |

选片未使用OS_event、OS_days或模型结果；不保证随机选中的切片是最具肿瘤代表性的切片。准备阶段不设seed，保存的CSV提供可复现的固定输入；训练seed不改变选片和划分。

## 运行

当前数据已准备好。在项目根目录执行一次下面的命令，即可在后台通过三个独立进程**并行**运行UNI、UNI L2、KEEP，默认共用GPU 0，每组各做五折。一组报错时其他组继续，汇总等待全部结束后以非零状态退出；不自动重试或换切片：

```bash
bash tests/09_cptac_luad_pathology/run_all.sh
```

查看整体进度：

```bash
tail -f tests/09_cptac_luad_pathology/logs/run_all.log
```

每组完整训练日志另存于logs/，文件名由Python按实际参数生成。调整所有组共同的参数时，修改run_all.sh中的普通变量即可；seed仅在Python中设置，默认123。

## 早停5轮对照

原run_all.sh继续使用早停10轮。新增脚本将连续无验证C-index提升的容忍轮数设为5，最多仍训练30轮，其余设置、选片和五折保持一致：

```bash
bash tests/09_cptac_luad_pathology/run_all_patience5.sh
```

```bash
tail -f tests/09_cptac_luad_pathology/logs/run_all_patience5.log
```

三组同时启动。各自日志、checkpoint和results的运行标签增加_patience5，保留原早停10轮目录；run_config.yaml记录实际patience。验证C-index连续5轮无提升时停止，例如第1轮最佳、之后始终持平，则在第6轮结束。修改后的并行方式仅对新启动的任务生效，不改变已经运行的顺序进程。

## Cox batch32和64对照

两组均沿用早停5轮、最多30轮和相同的105例选片/五折，其余超参数不变。每个脚本内部均并行运行UNI、UNI L2、KEEP三组。

Cox batch32：

```bash
bash tests/09_cptac_luad_pathology/run_all_coxbs32_patience5.sh
```

```bash
tail -f tests/09_cptac_luad_pathology/logs/run_all_coxbs32_patience5.log
```

Cox batch64：

```bash
bash tests/09_cptac_luad_pathology/run_all_coxbs64_patience5.sh
```

```bash
tail -f tests/09_cptac_luad_pathology/logs/run_all_coxbs64_patience5.log
```

调节的是累计计算Cox损失的风险集大小，物理batch仍为1名患者。按每折84名训练患者和尾部并入规则，实际更新为：

| Cox batch | 实际风险集分组 | 每epoch更新次数 |
|---|---|---:|
| 16 | 16、16、16、16、20 | 5 |
| 32 | 32、52 | 2 |
| 64 | 84 | 1 |

因此这些对照同时改变风险集大小与每轮优化次数，不能把差异全归因于风险集更完整。日志、checkpoint和results由coxbs32/coxbs64及patience5分别命名，不覆盖batch16配置。

单独运行一组时，修改train.sh中的FEATURE_SOURCE=uni、uni_l2或keep，然后执行：

```bash
bash tests/09_cptac_luad_pathology/train.sh
```

重新部署时先运行准备入口。若已有选片与划分，会严格核对并复用，不重新随机抽样：

```bash
python tests/09_cptac_luad_pathology/prepare.py
```

## 输出

- data/all_label.csv：每名患者选中的slide_id、三类输入共用的特征路径、原始event/time及五折。
- data/slide_selection_audit.csv：313张候选切片、各自所属患者、是否选中及患者折号。
- data/preparation.json、fold_counts.csv：原始三表哈希、冻结名单哈希、所选特征指纹及逐折人数/事件。
- checkpoints/<run_tag>/fold_*/：best_model.pth、last_model.pth、checkpoint_metadata.yaml。
- results/<run_tag>/：有效参数run_config.yaml、相同选片快照data.csv和metrics.csv（逐折及mean±样本标准差）。
- results/<run_tag>/fold_*/：Train/Val预测（case_id、OS_days、OS_event、risk_score），验证集分组、KM、评估说明。风险分组阈值由训练风险中位数确定。

24/36月AUC以训练数据估计删失分布，单独评估各时间点。若无病例/对照或IPCW支持不足，记NaN并在evaluation_notes.json记录原因，不用0代替。每折仅4～5个事件，AUC/KM/HR属于小样本描述；不把相关患者对当作独立患者数，也不将内部验证成绩宣称为独立测试。

重载已有模型时以相同参数运行train.py并加--eval_only；选择不同feature_source会定位到各自的checkpoint。重载前检查全部五折checkpoint及原配置/数据，不覆盖训练provenance。

## 检查

```bash
python tests/09_cptac_luad_pathology/test_experiment.py
```

检查仅使用临时合成数据，覆盖患者标签与切片对齐、跨切片划分冲突、选片冻结、三种特征、逐patch L2与零向量、三组五折训练/重载、OS字段/时间及C-index、缺少checkpoint的保护、patience5停止时点，以及三个真实轻量子进程的并行屏障和部分失败后其他进程完成。没有由本次修改启动真实CPTAC训练，三组并发的实际GPU显存尚未实测。

## CPTAC EAGLE Top25＋MeanPool对照

输入为`/home/gly001/cqj/EAGLE/test/cptac_luad/eagle/features/`的Virchow2-CLS patch特征。
继续使用本实验已经固定的105名患者、105张WSI及原五折；这些切片在EAGLE的313张输出中均存在。
不使用该目录另行计算的多WSI患者均值或CoxNet结果，也不重新选片。
数据层核对EAGLE输入表中的患者、切片、OS标签、折号，以及selection_manifest中的Top25和特征路径。

网络为`[B,25,1280] → 原始patch均值[B,1280] → Linear(1280,512) → ReLU → Dropout(0.25) → Linear(512,256) → ReLU → Dropout(0.25) → Linear(256,1)`。
不在均值前做投影或Attention，不额外做L2归一化。标量风险不经Sigmoid，损失仍为Cox。

```bash
bash tests/09_cptac_luad_pathology/train_eagle_meanpool.sh
```

脚本自带nohup和日志保存；默认真实患者batch16，Adam lr1e-4、weight_decay5e-4、最多30轮、早停10，Python训练seed默认123。
每折84名训练患者的风险集为16、16、16、16、20；每个batch一次联合Cox损失和更新，不跨batch累计。
这里`--cox_batch_size`就是患者batch大小，配置另行记录`patient_batch_size`；复用train_path时的内部iteration分组设为1，不写入有效患者batch配置。
验证及最终生存指标沿用原OS流程：时间保留天，计算24/36月AUC、KM、HR，属于带验证早停的内部五折。

本组运行标签包含`single_eagle_top25_meanpool_cox`，日志、checkpoint与results仍写入实验09，与原UNI/UNI L2/KEEP ABMIL结果区分。
本组`data.csv`保存实际EAGLE路径及n_patches=25，配置记录该快照哈希、原固定名单哈希、输入表哈希和实际特征指纹；eval_only会验证这些内容。
原始`data/all_label.csv`与已有结果不变。原三种特征继续默认ABMIL，原并行脚本不增加此实验。

相较原UNI/KEEP实验，这个对照同时改变了编码器、组织尺度、patch筛选及聚合方法，不能把结果差异仅归因于Attention。

## EAGLE Top25＋MeanPool＋单层Linear容量对照

此对照与上面的EAGLE MeanPool MLP使用相同105例、相同单WSI、25×1280特征、OS标签和五折。
网络仅为`[B,25,1280] → mean(dim=1) → [B,1280] → Linear(1280,1)`，共1281个参数（含bias），
没有投影、隐藏层、ReLU、Dropout或Sigmoid。返回原始标量风险，继续使用Cox Loss。
本地模型位于`meanpool_linear.py`，不修改主流程MeanPool或已有MLP模型。

```bash
bash tests/09_cptac_luad_pathology/train_eagle_linear.sh
```

脚本内含nohup和日志保存。仍用患者batch16、Adam lr1e-4、weight_decay5e-4、最多30轮、早停10，训练seed默认123。
`--pa_model meanpool_linear --feature_source eagle_top25`启用此对照。
标签包含`single_eagle_top25_meanpool_linear_cox`，分别保存日志、checkpoint和结果；此模型不使用Dropout，因此脚本、运行标签和run_config不记录Dropout参数。
验证、最佳checkpoint选择和OS的C-index、24/36月AUC、KM、HR评估沿用同一流程。
已有MLP对照仍使用Dropout0.25；线性结构移除了非线性和Dropout，不能把差异单独归因于参数数量。

## UNI、UNI L2、KEEP：Cox按患者数归一化对照

该对照已完成，三种特征未取得一致改善。按用户决定，后续继续使用原本按事件数归一化的Cox。
主流程、实验09默认值和常规启动脚本均保持原定义；以下病例数归一化脚本及其结果仅保留作历史对照。

```bash
bash tests/09_cptac_luad_pathology/run_all_patientmean.sh
```

脚本内含nohup，三个独立进程并行训练UNI、UNI L2和KEEP，每组完整五折。
沿用固定105名患者、105张单WSI、OS标签和五折；仍为ABMIL、Dropout0.25、Cox batch16、Adam lr1e-4、
weight_decay5e-4、最多30轮、早停10、默认训练seed123。

唯一干预是`--cox_reduction patients`：Breslow Cox负对数部分似然总和除以当前实际患者数。
普通风险集16人除以16，合并尾部20人除以20，不使用配置中的16代替实际人数；无事件时仍返回可反传的0。
模型、风险集组成、患者顺序、事件编码和早停规则保持一致。
验证及最终Train/Val loss也采用患者数归一化，各自使用整个评估队列的人数；C-index、24/36月AUC、KM、HR计算不变。
不同归一化方式下的损失数值不能直接用于判断预测性能改善。

本地损失在`patient_cox_loss.py`，复用共享Cox的风险集与Breslow ties，不修改共享损失。
共享`train_path`仅增加可选criterion接口，未传入时保持原有Cox/CoxRank/NLL行为；只有本对照传入患者数归一化损失。
默认`--cox_reduction events`保留原来的运行标签、配置和结果，旧脚本行为不变。
新标签增加`cox_patientmean`，配置记录`cox_reduction: patients`及训练/评估分母；eval_only检查同一设置和数据来源。

```bash
tail -f tests/09_cptac_luad_pathology/logs/run_all_patientmean.log
```

各组完整训练日志另存于logs/，文件名含各自特征名和`cox_patientmean`。

包含新增EAGLE检查的完整CPU合成测试：

```bash
/home/gly001/.conda/envs/UNI/bin/python -m unittest discover -s tests/09_cptac_luad_pathology -p 'test_*.py' -v
```
