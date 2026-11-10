# 实验05：EAGLE top25 单模态病理 ABMIL / MeanPool

- 数据：`data/eagle/top25/all_label_roi64.csv`，读取已有五折，不重新划分。
- 每例原始特征为 `[25, 1280]`，不额外归一化、采样或改变 patch 顺序。
- 网络：1280→512（ReLU、Dropout），Attention 512→128→1，softmax 加权池化，512→256（ReLU、Dropout），256→1 风险分数。Attention 中的 Dropout 与主流程一致。
- 默认 Dropout=0.25、Adam lr=1e-4、weight decay=5e-4，最多30轮、早停10轮、seed=123，每折重设同一训练种子。
- 真实患者 batch=64：输入 `[B,25,1280]`，每个 batch 一次前向、一个联合 Cox 损失、一次更新。不跨 batch 累积梯度。末尾不足64人并入前一个 batch；ROI64 中训练576人是9×64，训练577人是8×64+65。
- 验证、最终训练集及验证集评估均使用 eval 模式；C-index、验证 Cox loss 在完整患者集合上计算，不对各 batch 指标取平均。
- 复用 `path_train.train_path` 时，内部 `cox_batch_size=1` 指一个 DataLoader iteration（该 iteration 已包含一个完整患者 batch），不是单患者风险集。实际患者 batch size 记录在配置的 `batch_size` 中。
- 最终加载最佳验证 C-index checkpoint，导出与主流程一致的预测、KM、36/60月AUC、HR及 `metrics.csv`，五折以均值±样本标准差汇总。

运行（脚本内含 nohup，自动后台执行）：

```bash
bash tests/05_eagle_top25_pathology/train.sh
```

日志、checkpoint及结果全部写入本实验目录下 `logs/`、`checkpoints/`、`results/`。每次结果目录内的 `data.csv` 保存所用数据表快照。

该实验同时更换特征编码器、patch筛选/数量及输入组织尺度，因此与原UNI结果的差异不能单独归因于特征维度或batch前向方式。

MeanPool对照直接在原始25个1280维向量上求均值，不使用patch投影或Attention，再接现有MeanPool风险头：
`[B,25,1280] → mean(dim=1) → [B,1280] → Linear(1280,512) → ReLU → Dropout(0.25) → Linear(512,256) → ReLU → Dropout(0.25) → Linear(256,1)`。
输出为未经Sigmoid的标量风险，继续使用Cox Loss。

```bash
bash tests/05_eagle_top25_pathology/train_meanpool.sh
```

该脚本内含nohup，使用batch64，其余训练设置与原ABMIL batch64对照一致。沿用原患者及五折划分，不新增L2归一化。
`train.py --pa_model meanpool`启用此模型；不传`--pa_model`仍使用ABMIL。
MeanPool的日志、checkpoint及结果目录名含`meanpool`，与原ABMIL输出区分。
该风险头仍含非线性MLP，约79万参数；此实验不能当作纯线性风险头实验，也不能仅凭结果变化证明Attention是过拟合的唯一原因。

CPU合成数据检查（不读取真实特征、不运行真实训练）：

```bash
/home/gly001/.conda/envs/UNI/bin/python -m unittest discover -s tests/05_eagle_top25_pathology -p test_experiment.py -v
```
