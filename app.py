import os

from flask import Flask, render_template, request

from ai_analysis import DeepSeekAnalyzer
from model_registry import (
    BRANCH_MODEL_SPECS,
    FEATURE_SPECS,
    OVERALL_FEATURE_NAMES,
    build_feature_fields,
    build_feature_groups,
    finalize_branch_panel,
    load_branch_services,
)
from model_service import FeatureShapeError, PredictionService
from llm_pipeline import JUDGMENT_LABELS
from rag_store import get_default_corpus_store
from report_export import build_analysis_report
from risk_config import BRANCH_QFR_THRESHOLDS, RISK_THRESHOLD


class _NullShapRuntime:
    def explain_all(self, feature_map):
        return []


def _load_shap_runtime():
    if os.environ.get("SHAP_ENABLED", "1").strip().lower() in {"0", "false", "no", "off"}:
        print("[riskpredict] SHAP disabled via SHAP_ENABLED")
        return _NullShapRuntime()
    try:
        from shap_service import build_shap_runtime

        return build_shap_runtime()
    except ImportError as exc:
        print(f"[riskpredict] SHAP unavailable, skipping explanations: {exc}")
        return _NullShapRuntime()
    except Exception as exc:
        print(f"[riskpredict] SHAP init failed: {exc}")
        return _NullShapRuntime()


def _shap_is_active(runtime) -> bool:
    return not isinstance(runtime, _NullShapRuntime)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "20260610_most_powerful", "y_Ridge.dat")
RAG_CORPUS_DIR = os.path.join(BASE_DIR, "rag_corpus")

FEATURE_FIELDS = build_feature_fields()
FEATURE_GROUPS = build_feature_groups(FEATURE_FIELDS)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024

prediction_service = PredictionService.from_shelve(
    MODEL_PATH,
    feature_names=OVERALL_FEATURE_NAMES,
)
branch_services = load_branch_services()
shap_runtime = _load_shap_runtime()
rag_corpus_store = get_default_corpus_store()
try:
    rag_auto_build = os.environ.get("RAG_AUTO_BUILD", "true").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }
    rag_corpus_store.ensure_index(auto_build=rag_auto_build)
except Exception as exc:
    print(f"[riskpredict] RAG index not ready: {exc}")

ai_analyzer = DeepSeekAnalyzer(
    api_key=os.environ.get("DEEPSEEK_API_KEY"),
    model=os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-pro"),
    verify_ssl=os.environ.get("DEEPSEEK_SSL_VERIFY", "true").lower() != "false",
    corpus_store=rag_corpus_store,
    rag_mode=os.environ.get("LLM_RAG_MODE", "rag"),
)


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "GET":
        return _render_index()

    if request.form.get("mode") != "manual":
        return _render_error("Invalid submission mode.")

    return _handle_manual_prediction()


def _handle_manual_prediction():
    feature_map, form_values, error = _parse_manual_feature_map()
    if error:
        return _render_error(error, values=form_values)

    try:
        prediction = prediction_service.predict_values_by_names(feature_map)
        branch_predictions = _predict_all_branches(feature_map, prediction)
        shap_explanations = [
            item.to_dict() for item in shap_runtime.explain_all(feature_map)
        ]
    except FeatureShapeError as exc:
        return _render_error(str(exc), values=form_values)
    except KeyError as exc:
        return _render_error(str(exc), values=form_values)
    except Exception as exc:
        return _render_error(f"Prediction failed: {exc}", values=form_values)

    risk = classify_risk(prediction)
    values = [feature_map[field["column"]] for field in FEATURE_FIELDS]
    judgment_mode = "combined"
    ai_analysis = ai_analyzer.analyze(
        fields=FEATURE_FIELDS,
        values=values,
        prediction=prediction,
        risk=risk,
        judgment_mode=judgment_mode,
    )
    ai_analysis = _attach_judgment_metadata(ai_analysis, judgment_mode)

    analysis_report = None
    if ai_analysis and (ai_analysis.get("content") or ai_analysis.get("error")):
        analysis_report = build_analysis_report(
            result=prediction,
            risk=risk,
            ai_analysis=ai_analysis,
            judgment_mode=judgment_mode,
            judgment_labels=JUDGMENT_LABELS,
        )

    return _render_index(
        result=prediction,
        risk=risk,
        branch_predictions=branch_predictions,
        shap_explanations=shap_explanations,
        ai_analysis=ai_analysis,
        analysis_report=analysis_report,
        values=form_values,
        judgment_mode=judgment_mode,
    )


def _parse_manual_feature_map():
    feature_map = {}
    form_values = {}
    for field in FEATURE_FIELDS:
        raw_value = request.form.get(field["name"], "").strip()
        form_values[field["name"]] = raw_value
        if raw_value == "":
            return (
                None,
                form_values,
                f"Please enter {field['label_en']}.",
            )

        try:
            feature_map[field["column"]] = float(raw_value)
        except ValueError:
            return (
                None,
                form_values,
                f"{field['label_en']} must be a number.",
            )

    return feature_map, form_values, None


def _predict_all_branches(feature_map, screening_score):
    raw = finalize_branch_panel(feature_map, branch_services, screening_score)

    results = []
    for spec in BRANCH_MODEL_SPECS:
        qfr = raw[spec["id"]]
        results.append(
            {
                "id": spec["id"],
                "label_zh": spec["label_zh"],
                "label_en": spec["label_en"],
                "qfr": qfr,
                "status": classify_branch_qfr(qfr, BRANCH_QFR_THRESHOLDS[spec["id"]]),
            }
        )
    return results


def _attach_judgment_metadata(ai_analysis, judgment_mode="combined"):
    payload = dict(ai_analysis or {})
    mode = payload.get("judgment_mode") or judgment_mode
    payload["judgment_mode"] = mode
    payload["judgment_label"] = JUDGMENT_LABELS.get(mode, mode)
    return payload


def classify_risk(value):
    if value >= RISK_THRESHOLD:
        return {
            "label_zh": "高风险",
            "label_en": "High Risk",
            "class_name": "high-risk",
        }

    return {
        "label_zh": "低风险",
        "label_en": "Low Risk",
        "class_name": "low-risk",
    }


def classify_branch_qfr(value, threshold):
    if value >= threshold:
        return {
            "label_zh": "正常",
            "label_en": "Normal",
            "class_name": "branch-normal",
        }

    return {
        "label_zh": "关注",
        "label_en": "Attention",
        "class_name": "branch-attention",
    }


def _render_index(
    error=None,
    result=None,
    risk=None,
    branch_predictions=None,
    shap_explanations=None,
    ai_analysis=None,
    analysis_report=None,
    values=None,
    judgment_mode="combined",
    status=200,
):
    return (
        render_template(
            "index.html",
            error=error,
            result=result,
            risk=risk,
            branch_predictions=branch_predictions or [],
            shap_explanations=shap_explanations or [],
            ai_analysis=ai_analysis,
            analysis_report=analysis_report,
            values=values or {},
            feature_fields=FEATURE_FIELDS,
            feature_groups=FEATURE_GROUPS,
            feature_count=len(prediction_service.feature_indexes),
            full_feature_count=len(FEATURE_SPECS),
            rag_status=rag_corpus_store.status(),
            judgment_mode=judgment_mode,
            judgment_labels=JUDGMENT_LABELS,
            shap_enabled=_shap_is_active(shap_runtime),
        ),
        status,
    )


def _render_error(message, values=None):
    return _render_index(error=message, values=values, status=400)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
