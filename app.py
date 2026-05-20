import os
from datetime import datetime
from io import BytesIO

from flask import Flask, render_template, request, send_file
from werkzeug.utils import secure_filename

from ai_analysis import DeepSeekAnalyzer
from model_service import FeatureShapeError, PredictionService
from llm_pipeline import JUDGMENT_LABELS, normalize_judgment_mode
from rag_store import get_default_corpus_store
from report_export import build_analysis_report


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(
    BASE_DIR,
    "SVR_StandardScaler_RRMSE_2026-05-17_23-17-32.666985.dat",
)
RAG_CORPUS_DIR = os.path.join(BASE_DIR, "rag_corpus")
FEATURE_LABELS = [
    ("面部光谱均值1", "Face spectrum mean 1"),
    ("面部纹理均值30", "Face texture mean 30"),
    ("面部纹理均值78", "Face texture mean 78"),
    ("面部纹理均值94", "Face texture mean 94"),
    ("面部纹理均值102", "Face texture mean 102"),
    ("面部纹理均值126", "Face texture mean 126"),
    ("面部纹理均值166", "Face texture mean 166"),
    ("左耳纹理均值218", "Left ear texture mean 218"),
    ("左耳纹理均值1438", "Left ear texture mean 1438"),
    ("右耳纹理均值1574", "Right ear texture mean 1574"),
    ("右耳纹理方差1608", "Right ear texture variance 1608"),
    ("右耳纹理均值1200", "Right ear texture mean 1200"),
    ("面部纹理方差1326", "Face texture variance 1326"),
    ("面部纹理方差846", "Face texture variance 846"),
    ("左耳纹理均值739", "Left ear texture mean 739"),
    ("面部纹理均值6", "Face texture mean 6"),
    ("面部纹理均值14", "Face texture mean 14"),
    ("面部纹理均值54", "Face texture mean 54"),
    ("面部纹理均值46", "Face texture mean 46"),
    ("面部纹理方差611", "Face texture variance 611"),
    ("面部纹理方差1559", "Face texture variance 1559"),
    ("面部纹理方差875", "Face texture variance 875"),
    ("右耳纹理均值71", "Right ear texture mean 71"),
]
FEATURE_FIELDS = [
    {
        "name": f"feature_{index}",
        "index": index + 1,
        "label_zh": label_zh,
        "label_en": label_en,
    }
    for index, (label_zh, label_en) in enumerate(FEATURE_LABELS)
]
FEATURE_GROUP_DEFINITIONS = [
    ("face_spectrum", "面部光谱指标", "Face Spectrum Features", [0]),
    (
        "face_texture",
        "面部纹理指标（均值与方差）",
        "Face Texture Features (Mean & Variance)",
        [1, 2, 3, 4, 5, 6, 12, 13, 15, 16, 17, 18, 19, 20, 21],
    ),
    ("left_ear_texture", "左耳纹理指标", "Left Ear Texture Features", [7, 8, 14]),
    ("right_ear_texture", "右耳纹理指标", "Right Ear Texture Features", [9, 10, 11, 22]),
]
FEATURE_GROUPS = [
    {
        "id": group_id,
        "title_zh": title_zh,
        "title_en": title_en,
        "fields": [FEATURE_FIELDS[index] for index in indexes],
    }
    for group_id, title_zh, title_en, indexes in FEATURE_GROUP_DEFINITIONS
]

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024

prediction_service = PredictionService.from_shelve(MODEL_PATH)
rag_corpus_store = get_default_corpus_store()
try:
    rag_corpus_store.ensure_index(auto_build=True)
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

    mode = request.form.get("mode")
    if mode == "manual":
        return _handle_manual_prediction()

    file = request.files.get("file")
    if not file or not file.filename:
        return _render_error("请先选择一个 Excel 文件。")

    if not file.filename.lower().endswith((".xlsx", ".xls")):
        return _render_error("只支持上传 .xlsx 或 .xls 文件。")

    filename = secure_filename(file.filename) or "input.xlsx"
    uploaded_file = BytesIO(file.read())
    uploaded_file.seek(0)

    try:
        output = prediction_service.predict_excel(uploaded_file)
    except FeatureShapeError as exc:
        return _render_error(str(exc))
    except Exception as exc:
        return _render_error(f"预测失败：{exc}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    download_name = f"prediction_result_{timestamp}.xlsx"
    return send_file(
        output,
        as_attachment=True,
        download_name=download_name,
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


def _handle_manual_prediction():
    values = []
    form_values = {}
    for field in FEATURE_FIELDS:
        raw_value = request.form.get(field["name"], "").strip()
        form_values[field["name"]] = raw_value
        if raw_value == "":
            return _render_error(
                f"请填写 {field['label_zh']} / {field['label_en']}。",
                values=form_values,
            )

        try:
            values.append(float(raw_value))
        except ValueError:
            return _render_error(
                f"{field['label_zh']} / {field['label_en']} 必须是数字。",
                values=form_values,
            )

    try:
        prediction = prediction_service.predict_values(values)
    except FeatureShapeError as exc:
        return _render_error(str(exc), values=form_values)
    except Exception as exc:
        return _render_error(f"预测失败：{exc}", values=form_values)

    risk = classify_risk(prediction)
    judgment_mode = normalize_judgment_mode(request.form.get("ai_judgment_mode"))
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
        ai_analysis=ai_analysis,
        analysis_report=analysis_report,
        values=form_values,
        judgment_mode=judgment_mode,
    )


def _attach_judgment_metadata(ai_analysis, judgment_mode):
    payload = dict(ai_analysis or {})
    mode = normalize_judgment_mode(
        payload.get("judgment_mode") or judgment_mode
    )
    payload["judgment_mode"] = mode
    payload["judgment_label"] = JUDGMENT_LABELS.get(mode, mode)
    return payload


def classify_risk(value):
    if value >= 0.8:
        return {
            "label_zh": "低风险",
            "label_en": "Low Risk",
            "class_name": "low-risk",
        }

    return {
        "label_zh": "高风险",
        "label_en": "High Risk",
        "class_name": "high-risk",
    }


def _render_index(
    error=None,
    result=None,
    risk=None,
    ai_analysis=None,
    analysis_report=None,
    values=None,
    judgment_mode="rag_only",
    status=200,
):
    return (
        render_template(
            "index.html",
            error=error,
            result=result,
            risk=risk,
            ai_analysis=ai_analysis,
            analysis_report=analysis_report,
            values=values or {},
            feature_fields=FEATURE_FIELDS,
            feature_groups=FEATURE_GROUPS,
            feature_count=len(prediction_service.feature_indexes),
            full_feature_count=max(prediction_service.feature_indexes) + 1,
            rag_status=rag_corpus_store.status(),
            judgment_mode=judgment_mode,
            judgment_labels=JUDGMENT_LABELS,
        ),
        status,
    )


def _render_error(message, values=None):
    return _render_index(error=message, values=values, status=400)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
