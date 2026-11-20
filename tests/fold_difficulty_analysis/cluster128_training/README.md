# Raw UNI Cluster-balanced128：主流程Epoch五折训练

本次按用户更正，训练协议直接复用`path_train.py`，不再进行Nested CV或固定Step选轮。
读取`data/trident/features_uni_v1/all_label_roi64.csv`的原始五折；每折用其余四折训练，当前折逐Epoch验证，按Val C-index保存best及早停。

配置对标当前`train_path.sh`：ABMIL、Cox、Dropout0.25、30Epoch、patience10、Cox batch32（尾部并入）、Adam lr1e-4、WD5e-4、seed123。
特征依用户实验要求改为完整Raw UNI1024维；每轮训练访问患者时，按无监督簇均衡无放回采样128个Patch。
验证和最佳Checkpoint的最终Train/Val评估使用完整WSI。每轮Train C-index和Train Loss采用主流程在线统计；最终重新eval评估。
不对模型输入做L2、不加载项目已训练权重、不改主流程文件。

聚类复用已有缓存：每WSI独立MiniBatchKMeans8簇，seed123、batch1024、n_init3，仅对聚类副本做L2，标签不涉及DFS。
缓存必须通过来源SHA256及参数校验。模型输入仍是原始UNI。
采样用独立(seed123,epoch,patient)随机流；每Epoch改变子集，不影响患者shuffle和Dropout随机流。
极小簇可能含背景或伪影，均衡采样不等于组织质量筛选。

- `training.log`：原生主流程格式 `Epoch | Train Loss | Train C-index | Val Loss | Val C-index`。
- `status.json`：当前Fold与完成状态。
- `results/`：run_config、各Fold预测、KM、生存指标和五折mean±SD，均由主流程输出。
- `checkpoints/`：各Fold best、last及训练轮数记录。
- `cache/`：721例聚类缓存，原始UNI文件不变。
- `visualizations/<patient>/overview.png`、`clusters.png`：全部簇覆盖及分簇WSI图。
- `epoch_training/`：独立入口与测试。仅进程内替换数据适配，不修改主流程源码。
- `validation/epoch_smoke/`：真实GPU两Epoch训练及完整推理导出验证。
- `nested_stopped/`：按用户要求停止的旧固定Step/Nested实验半成品，仅历史记录，不用于本轮性能比较。

运行：`bash tests/fold_difficulty_analysis/cluster128_training/run.sh`。重复运行拒绝覆盖。
本轮逐Epoch用原五折验证集选最佳模型，属于常规内部CV；与历史严格Nested结果评估协议不同，不能直接归因性能差异。
所有新输出均在本实验内，不自动提交或push。

## GitHub publication

The published subset contains the completed Epoch run and aggregate results. Local-only cache, visualizations, patient predictions, model weights, smoke outputs and stopped Nested artifacts are intentionally excluded. See `results/report.md`.
