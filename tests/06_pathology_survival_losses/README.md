# 实验06：三种病理输入 × 两种生存损失

所有实验均为单模态 ABMIL。只读取现有五折，核对病理ID、CT ID、事件、时间和split与UNI参考表一致，并将患者顺序对齐。代码、日志、模型和结果都位于本目录，主流程与实验01–05不改动。

| feature_source | 特征 | 额外归一化 | 训练输入 |
|---|---|---|---|
| uni_l2 | TRIDENT UNI，1024维 | 每个patch做L2，训练/验证一致 | 变长WSI，逐患者前向 |
| keep | TRIDENT KEEP，768维 | 保留编码器原输出 | 变长WSI，逐患者前向 |
| eagle | EAGLE top25 Virchow2-CLS，1280维 | 无 | 真实患者batch，特征[B,25,1280] |

- 网络沿用ABMIL：输入→512→Attention(128)→池化→256→生存输出；dropout=0.25。
- 两种损失：`nll`、`cox_rank`，共六组。后者为Cox＋平均患者对softplus排序，`--rank_weight`默认1.0。
- NLL：仅当前训练折事件时间四分位点分箱，通常四个区间；输出hazard与生存概率。变长WSI每患者更新一次；EAGLE每个真实batch更新一次。两者优化步数不同。
- Cox/rank：变长WSI合并64人的风险图后更新；EAGLE真实batch=64，不跨batch累计；尾部并入前一组。
- NLL验证逐患者计算，避免小尾批次改变验证损失权重。最终评估均为患者加权均值；Cox/rank在完整评估患者集合上计算。
- 默认ROI64、Adam lr=1e-4、weight decay=5e-4、最多30轮、早停10轮；训练seed=123，每折重设。固定既有split，不重新划分。
- 输出：每组`run_config.yaml`与`data.csv`；各fold最佳/最后checkpoint、预测CSV、KM、36/60月AUC、HR；`metrics.csv`记录各折及均值±样本标准差。只保存实际使用的参数。
- `--eval_only`从各折checkpoint恢复NLL切点，重新导出最佳模型结果；保留原训练配置与数据快照，配置或队列不一致时会报错。

一次后台顺序运行六组（同一GPU依次运行，单组失败会停止并返回非零状态）：

```bash
bash tests/06_pathology_survival_losses/run_all.sh
tail -f tests/06_pathology_survival_losses/logs/run_all.log
```

每组详细日志为`logs/uni_l2_nll.log`、`uni_l2_cox_rank.log`、`keep_nll.log`、`keep_cox_rank.log`、`eagle_nll.log`、`eagle_cox_rank.log`。

单独运行：编辑`train.sh`中的`FEATURE_SOURCE`、`SURVIVAL_LOSS`及需要的参数，再执行：

```bash
bash tests/06_pathology_survival_losses/train.sh
```

两个sh内已经包含nohup。结果名称包含特征来源、损失和有效batch/排序权重，六组互不覆盖。
