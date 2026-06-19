"""Shared risk stratification threshold for the Ridge-RF screening model."""

# 预测值 >= RISK_THRESHOLD 视为有问题（高风险）；< RISK_THRESHOLD 视为没问题（低风险）。
RISK_THRESHOLD = 0.4438

# 三分支 QFR 判定阈值（Y1=LAD, Y2=LCX, Y3=RCA）；预测值 >= 对应阈值标记为正常。
BRANCH_QFR_THRESHOLDS = {
    "lad": 0.8393,
    "lcx": 0.9017,
    "rca": 0.9022,
}
