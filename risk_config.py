"""Shared risk stratification threshold for the Ridge-RF screening model."""

# Predicted value >= RISK_THRESHOLD is high risk; < RISK_THRESHOLD is low risk.
RISK_THRESHOLD = 0.5

# Branch QFR thresholds (Y1=LAD, Y2=LCX, Y3=RCA); predicted value >= threshold is Normal.
BRANCH_QFR_THRESHOLDS = {
    "lad": 0.8,
    "lcx": 0.8,
    "rca": 0.8,
}
