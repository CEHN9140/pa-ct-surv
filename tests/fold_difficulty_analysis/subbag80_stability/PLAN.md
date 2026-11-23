# Fixed-model80% subbag diagnostic

Approved: no new training. Original721 patients, five best/last checkpoints, Raw UNI complete features.30 deterministic80% subsets; same subsets for both checkpoints. Eval-only cached patch projection/logits with subset softmax matches direct forwards. Separate Train/Val risk SD, checkpoint-specific full-Train normalized noise, Spearman, C-index and comparable/all-pair rank flips. Checkpoint scales/mean shifts and best-selection bias are interpretation limits. Do not automatically train consistency loss or tune retention/lambda.

Tests and real-case GPU smoke before source-pinned background run; independently reproduce metrics from all saved matrices. User requested push after completion: exact code/aggregate whitelist only, exclude private predictions/weights. No unrelated changes or force push.
