# 既有病理特征实验结果索引

本目录补充发布本地已有的KEEP、UNI（原始/L2/除40）、EAGLE Top25相关五折汇总指标。不包含特征、患者名单/预测、图像、checkpoint或新的训练。

- `summary.csv`：每个历史run五折C-index均值/样本SD、36/60月AUC均值与可用折数，完整原run目录名和来源路径。
- `metric_tables/run_XXX.csv`：原metrics.csv逐字节副本，保留每折原指标及原有mean/std行；summary仅从0–4折重新计算，绝不把mean/std当作额外折；`sources.json`提供来源与SHA256。run_id只是按来源路径排列的索引，不是效果排名。
- `validation.json`：发布数量、完成情况及拷贝校验；当前只收721例/251事件、五折完整指标。

## 必须区分的协议

大部分历史run使用Outer Val最佳checkpoint，具有选轮偏倚；同队列上还探索过多种batch/dropout/学习率/loss/seed。严格Nested Stage5/6单独标记，不能将其较低分数解释为模型劣化。

不同run可能只改归一化、batch、模型或优化参数，而非换基础模型。summary保留run_tag与source供逐项核对，未按最高分筛选赢家。UNI L2中带diagnostics/不同seed的结果也全部保留；部分普通/diagnostics数值重复，不是独立复现，不能重复加权得到模型平均成绩。

KEEP是768维不同编码器输出，不等同于UNI加L2；UNI是1024维完整WSI patch。EAGLE本身是CHIEF评分/Top25选择流程，本项目配合Virchow2 CLS得到25×1280维特征，不是一个独立‘EAGLE编码器’。它同时改变编码器、patch选择与bag范围，与全WSI UNI/KEEP不能作为纯编码器单因素比较。EAGLE测试目录末尾_l2表示额外L2；v7 run_tag含cls_raw/cls_l2明确区分。

`MeanPool`的历史网络可能含MLP，只有Stage5明确是线性Ridge-Cox。不能将多个run按原始C-index排序后推断某种特征理论上更优。本目录仅发布结果，不改训练流程。GL/CPTAC/TCGA等外部队列不混入721例队列汇总。ROI96/128的历史划分亦未与ROI64混合。

## 结果目录索引

|Run|特征|Train C|Val C|选择协议|每折表|
|---|---|---:|---:|---|---|
|run_001|UNI|0.7970|0.7444|Outer Val选轮|[run_001](metric_tables/run_001.csv)|
|run_002|UNI|0.8285|0.7479|Outer Val选轮|[run_002](metric_tables/run_002.csv)|
|run_003|UNI|0.8368|0.7458|Outer Val选轮|[run_003](metric_tables/run_003.csv)|
|run_004|UNI|0.8269|0.7450|Outer Val选轮|[run_004](metric_tables/run_004.csv)|
|run_005|UNI|0.8301|0.7465|Outer Val选轮|[run_005](metric_tables/run_005.csv)|
|run_006|UNI|0.7679|0.7328|Outer Val选轮|[run_006](metric_tables/run_006.csv)|
|run_007|UNI|0.8304|0.7459|Outer Val选轮|[run_007](metric_tables/run_007.csv)|
|run_008|UNI|0.8722|0.7618|Outer Val选轮|[run_008](metric_tables/run_008.csv)|
|run_009|UNI|0.8319|0.7412|Outer Val选轮|[run_009](metric_tables/run_009.csv)|
|run_010|UNI|0.7570|0.7261|Outer Val选轮|[run_010](metric_tables/run_010.csv)|
|run_011|UNI|0.7915|0.7326|Outer Val选轮|[run_011](metric_tables/run_011.csv)|
|run_012|EAGLE CHIEF Top25 + Virchow2 CLS|0.8063|0.7544|Outer Val选轮|[run_012](metric_tables/run_012.csv)|
|run_013|EAGLE CHIEF Top25 + Virchow2 CLS|0.7905|0.7528|Outer Val选轮|[run_013](metric_tables/run_013.csv)|
|run_014|EAGLE CHIEF Top25 + Virchow2 CLS|0.8402|0.7585|Outer Val选轮|[run_014](metric_tables/run_014.csv)|
|run_015|EAGLE CHIEF Top25 + Virchow2 CLS|0.8497|0.7576|Outer Val选轮|[run_015](metric_tables/run_015.csv)|
|run_016|KEEP|0.7496|0.7371|Outer Val选轮|[run_016](metric_tables/run_016.csv)|
|run_017|UNI|0.7997|0.7393|Outer Val选轮|[run_017](metric_tables/run_017.csv)|
|run_018|UNI|0.7997|0.7393|Outer Val选轮|[run_018](metric_tables/run_018.csv)|
|run_019|UNI|0.7997|0.7403|Outer Val选轮|[run_019](metric_tables/run_019.csv)|
|run_020|UNI|0.7760|0.7364|Outer Val选轮|[run_020](metric_tables/run_020.csv)|
|run_021|UNI|0.7760|0.7364|Outer Val选轮|[run_021](metric_tables/run_021.csv)|
|run_022|UNI|0.7885|0.7399|Outer Val选轮|[run_022](metric_tables/run_022.csv)|
|run_023|UNI|0.7456|0.7274|Outer Val选轮|[run_023](metric_tables/run_023.csv)|
|run_024|UNI|0.7456|0.7274|Outer Val选轮|[run_024](metric_tables/run_024.csv)|
|run_025|UNI|0.7384|0.7286|Outer Val选轮|[run_025](metric_tables/run_025.csv)|
|run_026|UNI|0.7709|0.7392|Outer Val选轮|[run_026](metric_tables/run_026.csv)|
|run_027|UNI|0.7914|0.7396|Outer Val选轮|[run_027](metric_tables/run_027.csv)|
|run_028|UNI|0.7789|0.7403|Outer Val选轮|[run_028](metric_tables/run_028.csv)|
|run_029|UNI|0.7636|0.7361|Outer Val选轮|[run_029](metric_tables/run_029.csv)|
|run_030|UNI|0.7539|0.7287|Outer Val选轮|[run_030](metric_tables/run_030.csv)|
|run_031|UNI|0.7539|0.7287|Outer Val选轮|[run_031](metric_tables/run_031.csv)|
|run_032|UNI|0.8164|0.7281|Outer Val选轮|[run_032](metric_tables/run_032.csv)|
|run_033|UNI|0.7470|0.7280|Outer Val选轮|[run_033](metric_tables/run_033.csv)|
|run_034|EAGLE CHIEF Top25 + Virchow2 CLS|0.8351|0.7501|Outer Val选轮|[run_034](metric_tables/run_034.csv)|
|run_035|EAGLE CHIEF Top25 + Virchow2 CLS|0.8039|0.7522|Outer Val选轮|[run_035](metric_tables/run_035.csv)|
|run_036|EAGLE CHIEF Top25 + Virchow2 CLS|0.8263|0.7556|Outer Val选轮|[run_036](metric_tables/run_036.csv)|
|run_037|EAGLE CHIEF Top25 + Virchow2 CLS|0.7900|0.7507|Outer Val选轮|[run_037](metric_tables/run_037.csv)|
|run_038|EAGLE CHIEF Top25 + Virchow2 CLS|0.7990|0.7497|Outer Val选轮|[run_038](metric_tables/run_038.csv)|
|run_039|UNI MeanPool + linear Ridge-Cox|0.7496|0.7197|Nested|[run_039](metric_tables/run_039.csv)|
|run_040|UNI ABMIL|0.7976|0.7332|Nested|[run_040](metric_tables/run_040.csv)|
