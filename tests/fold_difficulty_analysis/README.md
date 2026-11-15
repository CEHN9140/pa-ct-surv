# 病理训练与五折差异诊断

本目录发布已有实验的汇总分析、图表与验证记录，不包含训练模型或患者级数据。
这属于比较基线和机制诊断，不替代项目的病理–CT教师及CT学生方法。

## 阅读入口

- [第一阶段完整报告](pathology_stage1/report.md)：Raw UNI、UNI/40、UNI L2的
  配置核对、逐轮排序、风险尺度及真实过拟合证据。
- [同配置最终指标](pathology_stage1/matched_final_summary.csv)：五折均值和
  样本标准差。研究对象为1024维UNI全WSI，不是1280维EAGLE Top25。
- [历史五折差异分析](analysis_public.md)：随访组成、患者对错误和折间差异；
  公开版已移除具体病例段落，完整病例复核资料保留在本地。

## 第一阶段诊断图

第三阶段为[输入尺度与60步优化轨迹验证](pathology_stage3/README.md)：
仅Fold2、4，Raw、/40及两种首层权重补偿条件，共同随机数和固定更新预算。
Train/Val均在固定eval模型上记录，全部结束后统一生成对比报告。

第二阶段为[同初始化、同64名训练患者的单步诊断](pathology_stage2/plan.md)，
包含完整UNI特征下的激活、联合Cox梯度及Adam首步。代码独立于正式训练；
运行结束后生成该目录的report.md及三张汇总CSV，不保存病例清单或模型。

- [三种输入的训练曲线](pathology_stage1/matched_input_training_curves.png)
- [L2排序与风险尺度](pathology_stage1/l2_ranking_and_scale.png)
- [固定配置的Seed对照](pathology_stage1/l2_fixed_config_seed_stability.png)
- [Raw固定模型eval动态](pathology_stage1/raw_fixed_model_eval_dynamics.png)

每张图同时保存PDF。图中的数值来自真实日志或已保存的患者预测；没有平滑、
插值、提前停止后补值或补造缺失的Epoch。详细可用性见
[data_availability.csv](pathology_stage1/data_availability.csv)，复核记录见
[validation.json](pathology_stage1/validation.json)。

Raw和UNI/40没有逐轮患者风险，因此不能恢复其逐轮Risk SD和Spearman。
三输入对照中的Train曲线是在线训练指标；实验10的独立Raw曲线才是每次更新点
固定模型eval下的Train/Val，二者不可拼接。验证集unit-SD Cox仅用于诊断。

## 较早的UNI/L2/KEEP对照

[curves_uni_l2_keep](curves_uni_l2_keep)保留四项日志指标、五折曲线及原始作图表。
其配置与指标定义见该目录的[说明](curves_uni_l2_keep/README.txt)。KEEP是不同
的特征编码器，不能当作只改变L2的对照。

## 发布范围

`.gitignore`使用明确白名单，仅收录不含患者记录的小体积汇总、PNG/PDF图、
参数和来源验证记录。队列表、患者错误档案、病例复核名单及带患者ID的排序
清单不上传；原始含病例报告也只保留本地。模型权重和原始病理/CT文件不在
发布范围内。诊断结果仍属同队列内部探索，不能直接证明特征信息上限、临床
标签错误或某个机制的唯一因果解释。
