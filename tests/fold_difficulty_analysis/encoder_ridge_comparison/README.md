# 同Patch病理编码器对照（独立实验）

见PLAN.md/config.json。只读取既有721例UNI、KEEP、GPFM、Phikon-v2和Kaiko特征；无图像重提取、无新神经训练，不改主流程。CPU拟合五编码器并行，每进程BLAS单线程。

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .radiomics_env/bin/python -u tests/fold_difficulty_analysis/encoder_ridge_comparison/run.py
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .radiomics_env/bin/python -m pytest -q tests/fold_difficulty_analysis/encoder_ridge_comparison/test_run.py
```

结果在results/；每编码器private/含完整患者缓存、来源清单、模型及预测，禁止上传。完成后results/report.md汇总同协议比较。重运行只允许校验通过的同config/代码/来源MeanPool缓存，拟合均重新进行；不混用历史网格模型。

本次完成：750次内层尝试（31次数值失败，按预设规则使对应alpha不可选）、25次成功外层拟合。五折平均Val C-index：UNI 0.7201、KEEP 0.7252、GPFM 0.7168、Kaiko 0.7132、Phikon-v2 0.6980。KEEP相对新网格UNI为+0.0052，条件配对名义95%区间[-0.0063,+0.0163]，尚未确认优势。Phikon-v2在本次读取协议下五折均低于UNI；不能外推成该基础模型普遍较差。

EE仍约0.56–0.59；KEEP总体略高但没有改善EE或>24月EE。UNI的Fold0、3选择1e8上界，本轮未事后扩网格。当前结果不支持仅换这几套同Patch固定表示并以MeanPool读取即可明确突破瓶颈，也不证明WSI信息上限或临床结局不可预测。完整指标与限制见[报告](results/report.md)。
