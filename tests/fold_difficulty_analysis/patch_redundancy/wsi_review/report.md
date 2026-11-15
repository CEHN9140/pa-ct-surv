# WSI组织审查材料准备完成

25例：每个Outer Val按patch数10/30/50/70/90百分位秩选择5例，与Risk/结局无关。
四组各16个，共1600个展示patch；组间自然重叠允许。
25例UNI特征/坐标逐行一致，CHIEF索引、坐标及UNI值完全对应，视野为level0 512px。
使用Stage6冻结模型，Full Risk复现通过，未训练网络、未判读组织标签。

本地reviewer/为去病例标识、隐藏策略的原图位置和拼图及空白审查表；private/为解盲键和来源。
**材料生成完成不代表组织学审查完成；不能据此断言高Attention落在肿瘤或低Attention为伪影。**

技术复核：Full Risk最大误差0；所有病例实际read level=1/downsample=2.0。
独立抽查全部100组的首张patch，拼图中原尺寸像素与WSI对应位置直接读取逐像素一致。
共250张图片、1600行空白审查记录；Reviewer表不包含病例ID、结局、风险或策略信息。
已打开两个不同Fold的病例总览作显示检查。首例缩略图中的黑色块在原始WSI
level0/1/5同位置均存在，非本次标注或透明背景引入，保留原状以供审查。

先阅读本地 [盲法审查说明与病例入口](reviewer/REVIEW_GUIDE.md)。审查完成前不要将
private/group_key.csv和private/selection_key.csv提供给审查者。
