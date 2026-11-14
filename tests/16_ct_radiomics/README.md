# 实验16：单模态CT Radiomics基线

原`train.py`仅做独立比较基线，不读取病理特征、不使用神经网络或Checkpoint，不改动
主流程、其他实验和原始数据。此实验不替代正式的病理–CT教师与CT学生方法。

## 固定方案

- 患者与标签：`data/eagle/top25/all_label_roi64.csv`的721例、251个事件及
  原有五折。这里仅借用患者ID、time/event和split，不读取EAGLE特征。
- 输入：医生提供的`ct/image/<ct_id>.nii.gz`和`ct/mask/<ct_id>.nii.gz`。
  在完整三维肿瘤内提取，不使用64³归一化数组，也不限制为64³输入。
- PyRadiomics 3.0.1：Original三维107项非弃用特征；不含shape2D、Wavelet、LoG。
  目标体素间距1 mm，CT B-spline、Mask最近邻；固定binWidth=25；不做额外
  截窗、强度归一化或强度重分割，voxelArrayShift=0。物理坐标不一致时由
  PyRadiomics校正Mask并记录；无效Mask、提取失败或非有限值会阻止建模，
  不静默删除患者。
- 训练折内去除精确常数特征，然后按特征名排序、贪心去冗余：与已保留特征
  的绝对Spearman相关系数>0.90时剔除。顺序固定，不依赖DFS标签或验证指标。
  随后按训练均值/标准差做逐列Z-score，不做患者向量L2。
- Ridge-Cox，Breslow并列时间处理。每个外层训练集内部，使用事件分层三折
  （seed=123）选择正则系数[0.01,0.1,1,10,100,1000]；每个内层训练折重新
  拟合过滤、标准化及模型。按内层平均C-index选择，精确平分选更强正则。
  最后在完整外层训练集重新拟合，外层验证只评价一次。
- 不额外加入单因素P值筛选、mRMR、LASSO、PCA、ComBat或ICC筛选。
  没有重复勾画/扫描数据时，不声称已验证特征重测稳定性。

注意：原NIfTI上游处理未知，审计发现大多数图像上限315。binWidth=25指
当前输入强度单位，不宣称原始HU完整性已验证。本实验是Mask辅助的影像组学，
输入条件不同于仅从CT学习的深度编码器。结果属于当前固定队列的内部验证。

## 运行

当前已建立实验目录内的`.runtime`，复用UNI已安装的依赖，并在局部固定
NumPy1.26.4，避免NumPy2与PyRadiomics二进制扩展不兼容；UNI不受影响。
新机器可以按下列方式重建：

```bash
/home/gly001/.conda/envs/UNI/bin/python -m venv --system-site-packages tests/16_ct_radiomics/.runtime
tests/16_ct_radiomics/.runtime/bin/python -m pip install --no-deps numpy==1.26.4
```

后台完成提取和五折建模（CPU，不占GPU）：

```bash
bash tests/16_ct_radiomics/run.sh
```

查看日志：

```bash
tail -f tests/16_ct_radiomics/logs/radiomics_cox.log
```

需要分开执行时，在`train.py`指令后使用`--stage extract`或`--stage fit`。
默认`--stage all`。提取缓存需满足原患者、标签、参数以及源文件size/mtime校验；
已存在的建模结果不自动覆盖。修改方案应使用新的`--output_dir`。

## 输出

全部保存在`tests/16_ct_radiomics/output/`：

- `features.csv`：患者、标签、原Fold与107项候选特征。
- `extraction_qc.csv`、`failed_cases.json`：几何、强度范围与提取质量记录。
- `extraction_manifest.json`、`run_config.yaml`：输入及实际参数；源文件校验
  是路径/大小/mtime，不是全部NIfTI的内容哈希。
- `results/fold_0/`至`fold_4/`：拟合模型、内层调参结果、特征筛选明细、
  标准化特征系数、Train/Val患者预测、KM图。
- `results/metrics.csv`：每折Train/Val C-index、Cox NLL、36/60月IPCW AUC、
  Gap、HR、log-rank等；含五折均值及样本标准差。HR和P值保留逐折值。
- `results/oof_predictions.csv`：每位患者恰好一次外层验证预测。不同折风险
  尺度可不同，因此主要报告折内指标的均值，不直接将拼接风险作为统一校准风险。

Cox NLL在完整Train/Val风险集分别计算、除以事件数，不包含Ridge惩罚项；
不产生逐epoch训练曲线。KM使用训练风险中位数阈值；AUC沿用项目删失适配评价。
复现joblib模型时，需要将此目录加入Python搜索路径以导入`radiomics_ct`。

## 检查

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 MPLCONFIGDIR=/tmp/pact_matplotlib OPENBLAS_NUM_THREADS=2 \
tests/16_ct_radiomics/.runtime/bin/python -m pytest -q tests/16_ct_radiomics/test_radiomics_ct.py
```

测试另外需要pytest；这不是运行基线的依赖。真实病例冒烟验证和完整实验结果
应区别报告：代码通过检查不等于721例提取与五折建模已经完成。

## 已完成结果

721例、251个事件的完整五折已完成；以下为五折均值±样本标准差。

| 指标 | 结果 |
|---|---:|
| Train C-index | 0.7399 ± 0.0128 |
| Val C-index | 0.7336 ± 0.0608 |
| 36月IPCW AUC | 0.7773 ± 0.0490 |
| 60月IPCW AUC | 0.7420 ± 0.0886 |

各折从107项候选特征保留39/39/39/39/38项；本次没有常数项，删除均来自
训练折内高相关去冗余。五折都选择了预定义候选网格上限ridge=1000，这不是
对更强正则化的最优性证明。Fold4验证C-index为0.6384，折间差异仍明显。
完整验证记录见`validation.json`；7项测试包含CSV浮点往返精度回归检查。

## GitHub发布范围

仅发布代码、脚本、说明、参数和不含患者记录的小体积汇总结果，包括
`metrics.csv`、各折筛选/系数/调参表与KM图。
`.runtime/`、日志、模型、患者级特征/预测、提取QC和源文件清单均保留本地，
由本目录`.gitignore`白名单控制，不随提交上传。原始CT、Mask及标签表也不发布。

## 新增：病理Query / CT Radiomics Key–Value教师

独立入口`teacher_train.py`，复用721例原始五折和完整107项提取缓存。
原来的CT-only Ridge-Cox结果不变，也不加载它的模型或筛选器。
病理与CT的可学习模块均重新初始化并联合训练，不加载项目单模态权重。

Radiomics按以下7类组织：Shape、First-order、GLCM、GLRLM、GLSZM、GLDM、NGTDM。
每次实际训练FIT患者内重新去常数、高相关去冗余及Z-score，然后逐类独立
`Linear(保留维度,256) → ReLU → Dropout`。它们是**特征类别token**，不是
CT空间位置；某类被筛空时用Attention mask屏蔽，不生成有效的空组token。

数据通路（省略Batch维）：

| 路径 | 维度与操作 |
|---|---|
| 病理输入 | EAGLE Top25 Virchow2-CLS，25×1280；每个patch做L2 |
| 共享病理投影 | 1280→512，ReLU、Dropout，得到25×512 |
| Query | 512→256，ReLU、Dropout，得到25×256 |
| CT Key/Value输入 | 7组分别编码，得到7×256；之后MHA内部再做Q/K/V线性映射 |
| Cross-Attention | 标准单头softmax(QKᵀ/√256)V，输出25×256 |
| 交互表示 | 可学习Attention pooling，25×256→256 |
| 独立病理表示 | 原ABMIL Attention pooling，25×512→512→256 |
| 风险预测 | Concat为512→256→128→1；隐藏层LayerNorm、ReLU、Dropout |

投影、ABMIL、交互池化和风险头的Dropout统一0.25；MHA的**权重Dropout为0**，
沿用test15标准Attention设置。无QKNorm、无额外融合入口L2、无辅助分支Cox。
只优化融合风险的Cox损失；Adam lr=1e-4、weight_decay=5e-4、Python seed=123。
患者前向batch=1，32名患者联合计算一次Cox，尾部并入前一个风险集，不丢样本。
可选`--fusion_type concat`是同一套协议下的无Cross-Attention对照：有效CT类别
token均值与ABMIL表示拼接。默认仅运行`crossattention`。

验证使用原外层5折，每个外层训练集再做事件分层3折。在每个内层，剩余患者
按85% FIT/15% STOP划分；另有独立HELD，此教师选轮流程不使用其标签或预测。
过滤和标准化只拟合FIT，STOP C-index选最佳轮数，patience=10、最多30轮。
取3个内层最佳轮数中位数，在全部外层训练患者上重新拟合预处理并从新初始化
训练固定轮数，最后评价外层验证一次。每种融合共15次内层、5次外层拟合。
该协议与历史使用外层早停的教师不同，历史均值只能作参考，不能直接归因于融合改进。

启动后台训练（本脚本自行nohup，不要重复包一层）：

```bash
bash tests/16_ct_radiomics/run_teacher.sh
tail -f tests/16_ct_radiomics/logs/teacher_crossattention.log
```

仅审计病例、原始特征、所有内外层筛选分组，不训练：

```bash
tests/16_ct_radiomics/.runtime/bin/python tests/16_ct_radiomics/teacher_train.py --device cpu --audit_only
```

输出分开保存在：

- `teacher_results/crossattention/metrics.csv`：逐折指标、均值±样本标准差；
  每折Train/Val评估、36/60月IPCW AUC、HR、KM、筛选明细和固定轮数记录。
- `teacher_checkpoints/crossattention/fold_*/`：内层与外层权重、过滤器、标准化、
  类别布局、训练历史、患者划分和元数据。`teacher_train.load_teacher`可重载外层目录。
- 逐轮Train指标来自训练模式的在线风险；最终表中Train/Val均为保存模型在
  `eval()`下重新评估；最终Cox NLL使用各自完整患者风险集，按事件数平均。

本教师使用从完整肿瘤CT及Mask提取的Radiomics，与64³深度CT输入的覆盖范围不同；
Attention是类别权重，不能解释为CT空间定位。推理仍依赖相同Radiomics提取及Mask。
此处尚未实现学生的Radiomics教师特征适配或蒸馏，不能据此宣称CT-only学生已完成。
教师大文件、患者级表和日志仍被Git忽略。新功能核查记录见`teacher_validation.json`。

### Cross-Attention五折已经完成

721例、251事件，15次内层及5次外层训练完成。五折均值±样本标准差：

| Train C-index | Val C-index | 36月AUC | 60月AUC |
|---:|---:|---:|---:|
| 0.8017±0.0426 | 0.7476±0.0456 | 0.7862±0.0343 | 0.7677±0.0478 |

Fold4 Train/Val为0.8751/0.6874，仍有明显泛化差距。该结果不能单独证明
Cross-Attention优于Concat或病理-only。`teacher_validation.json`现在将原始
冒烟记录与已完成的五折验证分开；不能用其中原始冒烟阶段的状态推断当前进度。
小体积汇总metrics、完成标记、选轮记录及KM图可随Git提交；患者级表、模型、
含患者路径的run_config、原始特征和日志继续忽略。此次未自动commit/push。

### 匹配Concat与PA-only对照

保持Cross-Attention结果不变，两个新对照均从新初始化开始，分别独立执行相同
内层选轮及外层评价，不引用外层最佳分数选择轮数。运行参数与已完成教师一致。

- `concat`：七类CT token的有效类别均值256维，与ABMIL病理256维拼接，
  接相同512→256→128→1风险头；没有Cross-Attention和Query pooling。
- `pa_only`：相同Patch L2、ABMIL投影和池化，风险头256→256→128→1，
  相同LayerNorm/ReLU/Dropout。仅首层输入宽度由512改为256。病理编码器在
  同一seed下初始化一致；不同宽度风险头采用原生初始化，不声称所有参数或
  Dropout序列逐位相同。没有CT参数，预测可仅传病理；不拟合或保存CT过滤器、
  标准化。使用同一配对队列做完整性核验，CT缓存不进入其风险计算。

这组PA-only仅为比较基线，不能用它的训练权重初始化教师或学生。尚未加入
Radiomics风险级融合，也未修改Shape监督或主流程。

后台并行运行两组（脚本已含nohup，拒绝覆盖已有对照）：

```bash
bash tests/16_ct_radiomics/run_teacher_controls.sh
```

脚本使用setsid独立会话，防止执行工具会话结束时回收后台进程。

日志分别为`logs/teacher_concat.log`、`logs/teacher_pa_only.log`；结果分别为
`teacher_results/concat/`、`teacher_results/pa_only/`；权重在同名
`teacher_checkpoints/`子目录。两组各包含15次内层及5次外层拟合。
