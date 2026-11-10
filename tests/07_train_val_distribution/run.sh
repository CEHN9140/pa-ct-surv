BASE=/home/gly001/cqj/pa_ct_surv

nohup python -u "${BASE}/tests/07_train_val_distribution/analyze.py" > "${BASE}/tests/07_train_val_distribution/analysis.log" 2>&1 &
