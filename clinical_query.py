"""Build retrieval queries from model output and risk tier."""

from risk_config import RISK_THRESHOLD


def _is_high_risk(risk) -> bool:
    return "high" in risk.get("label_en", "").lower()


def build_clinical_query(prediction, risk, patient_context=None):
    is_high_risk = _is_high_risk(risk)

    if is_high_risk:
        management_terms = (
            "high-risk acute coronary syndrome NSTE-ACS chest pain "
            "early invasive strategy guideline-directed management "
            "serial troponin ECG monitoring antiplatelet anticoagulation "
            "urgent evaluation coronary angiography when indicated"
        )
        screening_note = (
            "noninvasive screening model suggests higher concern; "
            "does not replace coronary angiography or definitive diagnosis"
        )
    else:
        management_terms = (
            "low-risk acute coronary syndrome chest pain "
            "conservative strategy observation discharge criteria "
            "outpatient follow-up noninvasive testing shared decision-making "
            "guideline-directed medical therapy"
        )
        screening_note = (
            "noninvasive screening model suggests lower concern; "
            "continue clinical assessment and guideline-based evaluation"
        )

    parts = [
        "Acute coronary syndrome risk stratification after noninvasive hyperspectral screening.",
        f"Model predicted score {float(prediction):.6f}; threshold {RISK_THRESHOLD}; "
        f"classified as {risk.get('label_en', '')}.",
        management_terms,
        screening_note,
    ]

    if patient_context:
        parts.append(str(patient_context).strip())

    return " ".join(parts)


def build_risk_query(risk):
    """Backward-compatible query builder used by legacy retriever tests."""
    base_terms = (
        "acute coronary syndrome ACS guideline management chest pain "
        "risk stratification clinical evaluation"
    )

    if _is_high_risk(risk):
        return (
            f"high risk {base_terms} invasive evaluation early invasive "
            "urgent emergency reperfusion antiplatelet anticoagulation "
            "guideline-directed management"
        )

    return (
        f"low risk {base_terms} discharge follow-up outpatient observation "
        "serial troponin noninvasive testing shared decision-making"
    )
