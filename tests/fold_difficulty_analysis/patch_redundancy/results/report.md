# Patch冗余与Attention有效性：冻结模型推理诊断

仅复用Stage6五个Raw UNI最终Outer模型；721例、251事件；无网络训练、无新模型选择。
所有patch保留策略均在eval模式下对剩余patch重新Softmax；不足K的患者保留全部。
K=[128, 512, 1024]；Random每K重复30次，seed=123，无放回。

|策略|K|Val C-index（五折mean±SD）|Random重复SD|与Full Spearman（五折mean±SD）|ΔC vs Full|
|---|---:|---|---|---|---|
|full|All|0.7332 ± 0.0456|—|1.0000 ± 0.0000|+0.0000|
|top|128|0.7384 ± 0.0294|—|0.9590 ± 0.0180|+0.0052|
|top|512|0.7372 ± 0.0330|—|0.9788 ± 0.0118|+0.0040|
|top|1024|0.7352 ± 0.0362|—|0.9879 ± 0.0084|+0.0021|
|bottom|128|0.6406 ± 0.0177|—|0.5123 ± 0.0637|-0.0926|
|bottom|512|0.6611 ± 0.0340|—|0.6541 ± 0.0687|-0.0721|
|bottom|1024|0.6751 ± 0.0446|—|0.7400 ± 0.0675|-0.0581|
|random|128|0.7319 ± 0.0460|0.0026|0.9935 ± 0.0019|-0.0013|
|random|512|0.7329 ± 0.0457|0.0011|0.9983 ± 0.0005|-0.0003|
|random|1024|0.7330 ± 0.0457|0.0007|0.9992 ± 0.0002|-0.0001|

Random的五折SD：先对每折30次结果取均值，再计算五折样本SD。
Random重复SD：每次重复先取五折C-index均值，再计算30次样本SD；它不是患者Bootstrap置信区间。
各重复的C-index分别计算后取平均，没有先平均30次风险来制造集成模型。
Spearman逐折计算，不跨Fold混合风险；常数风险时不伪造相关系数。

|对照|K|平均ΔC|正增益Fold|
|---|---:|---:|---|
|top_minus_random|128|+0.0065|2/5|
|bottom_minus_random|128|-0.0912|0/5|
|top_minus_random|512|+0.0043|3/5|
|bottom_minus_random|512|-0.0718|0/5|
|top_minus_random|1024|+0.0022|3/5|
|bottom_minus_random|1024|-0.0580|0/5|

## Attention与实际删减幅度

attention_fold_summary.csv与attention_summary.csv记录Full Attention归一化熵、有效patch数/比例、
Top-K Attention Mass、patch数及N≤K患者比例。指标按患者等权汇总，避免大WSI主导统计。
N_eff=1/sum(a²)，不与exp(entropy)混用；单patch归一化熵约定0。

## 解释边界

Full预测必须重现Stage6存档。筛选来自当前ABMIL自身Attention，不是CHIEF评分。
预测对随机删减稳定仅支持预测层面的冗余，不证明被删组织无用或质量差。
Top-K优于Random-K只能支持本冻结模型的Attention筛选优势，不能证明重新训练Top-K会改善泛化。
Bottom-K表现良好不等于Attention失效，因为重新归一化改变输入分布，低权重patch仍可能有信息。
所有比较为同队列反复探索后的机制诊断；不按Outer Val挑K，不宣称独立验证或生物学对应关系。
本次不含WSI组织质量的人工审查、CHIEF坐标映射或任何新的训练实验。

## Full Attention统计（五折患者均值的mean±SD）

|指标|结果|
|---|---|
|effective_patches|1690.1118 ± 488.0140|
|effective_ratio|0.5539 ± 0.1577|
|fraction_N_le_1024|0.0291 ± 0.0124|
|fraction_N_le_128|0.0014 ± 0.0031|
|fraction_N_le_512|0.0111 ± 0.0079|
|normalized_entropy|0.9518 ± 0.0233|
|patch_count|3082.7841 ± 124.4028|
|top_mass_1024|0.6908 ± 0.0777|
|top_mass_128|0.1918 ± 0.0579|
|top_mass_512|0.4779 ± 0.0886|

## 本次结果解读

- Full Val C-index为0.7332，与Stage6严格Nested最终模型一致，未使用历史外层选轮结果替代。
  721例Full Risk复现误差全部为0；缓存子集与直接前向最大误差3.58e-7。
- Random-512为0.7329，与Full只差−0.0003，Risk Spearman为0.9983；
  Random-128也有0.7319、Spearman0.9935。支持当前冻结预测存在较高的patch冗余。
- Bottom-128/512/1024降至0.6406/0.6611/0.6751，五折都低于Full和对应Random均值。
  因此不能说Attention排序完全没有区分作用；高低Attention区域对当前预测的贡献不同。
- Top-K相对Random均值提高0.0022～0.0065，但只有2～3/5折改善；
  未证明Top-K具有稳定泛化优势，更不能据此选K并承诺重新训练会提升。
- 归一化熵0.9518、N_eff/N约0.5539提示整体Attention较广泛，
  同时Top-128仍占约19.18%的权重。高归一化熵不等于完全均匀或毫无筛选作用。
- 这些结果不能确定patch组织质量，也不能证明patch数量本身是训练过拟合的原因。
  未开展第二部分组织审查或任何后续重训。
