# WSI组织审查准备：冻结模型，不重训

执行用户附件中的下一步建议，仍置于patch_redundancy目录，保留既有推理结果。
本次只准备原图及盲法审查材料，不自动判读肿瘤/间质/正常/坏死，不启动Patch重训。

## 事先固定的取样

- 仍用Stage6五个Raw UNI最终Outer模型，每例只用其对应验证Fold模型。
- 每折按Full patch数、病例ID排序，取10/30/50/70/90百分位秩对应5例，共25例。
  不根据Risk、DFS事件、模型预测正确与否或Top-K性能挑病例。
- 每例4组，各16个patch：最高Attention、最低Attention、固定种子的随机无放回、
  已有冻结CHIEF Top25中评分最高16个。各组允许自然重叠，不排除重复以免改变策略。
- 高低Attention由当前ABMIL，CHIEF由已有独立预训练评分；二者不混称。

## 坐标与视野

- 必须验证features.h5内coords与coord_path全部逐行相同、feature行数匹配。
- CHIEF selected_indices必须唯一、在界内；其coords及features必须逐值等于原UNI相应行，
  其source_uni/source_coords/source_wsi必须对应同例，FOV元数据必须相同。
- 使用H5 patch_size_level0/level0_dimensions/目标倍率而非固定256 level0读图。
  已预检25例均为level0 512px视野，对应名义20x的256px patch。
- 读取与TRIDENT相同的金字塔层选择与目标patch尺寸，无额外染色/强度增强。
  缩略图坐标分别按真实宽高缩放，记录MPP但不把名义倍率当精确物理尺度。

## 输出、盲法和隐私

- 新review_wsi.py及test_review_wsi.py，独立输出wsi_review/，不覆盖原结果。
- reviewer/只含R001等匿名编号、A/B/C/D分组和tile01..16；不显示病例ID、Fold、
  DFS、Risk、Attention、策略名称；每例独立随机打乱组标签及组内展示顺序。
- 每组原图位置图+4×4原色patch拼图，每例总览；空白review.csv及审查说明。
- private/保存病例清单、分组解盲键、索引/坐标/Attention、来源hash和WSI size/mtime。
- 公共仅代码、说明及不含病例的完成/QC汇总；WSI图像、审查表、解盲键全部Git忽略。

## 验证

先测试取样不依赖结局、行错序/FOV/CHIEF来源错误拒绝、坐标宽高投影、盲法确定性、
实际patch读图视野与拼图布局，再实施。每例Full Risk复现已有存档，参数保持冻结；
检查输入/源码未变、25例及1600个展示patch完成，并打开实际图片做技术质量核对。
本次不进行诊断性组织标签，不把可视化材料的完成写成病理审查已完成。

## 完成记录

- 新增9项测试通过，连同原Patch诊断GPU回归共17项通过；独立只读审查无阻塞问题。
- 25例已全部完成；100组、1600次patch展示、250张位置/拼图/预览图片。
- 所有病例均为实际level1/downsample2.0，Full Risk复现最大误差0，模型/数据来源不变。
- 独立复查100组的首张patch：PNG拼图原尺寸像素与WSI level1直接读取逐像素一致。
- 空白审查表1600行，Reviewer列不含病例ID/Fold/DFS/Risk/Attention/策略。
- 实际打开R001、R021总览核查拼图、标号及地图；原图R001的黑色块在原生level0/1/5
  同位置均存在、alpha255，非新可视化叠加或透明背景引入，未擅自修补原图。
- 尚未进行组织学审查，不产生肿瘤/正常/伪影的真值标签，不启动后续Nested训练。
