# 实验16：单模态CT Radiomics基线

仅做独立比较基线，不读取病理特征、不使用神经网络或Checkpoint，不改动
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
