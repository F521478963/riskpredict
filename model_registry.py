import os

from feature_name_map import feature_alias
from model_service import PredictionService

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "20260610_most_powerful")

FEATURE_SPECS = [
    ("face guangpu mean1", "面部光谱均值1", "Face spectrum mean 1", "face_spectrum"),
    ("face wenli mean30", "面部纹理均值30", "Face texture mean 30", "face_texture"),
    ("face wenli mean78", "面部纹理均值78", "Face texture mean 78", "face_texture"),
    ("face wenli mean94", "面部纹理均值94", "Face texture mean 94", "face_texture"),
    ("face wenli mean102", "面部纹理均值102", "Face texture mean 102", "face_texture"),
    ("face wenli mean126", "面部纹理均值126", "Face texture mean 126", "face_texture"),
    ("face wenli mean166", "面部纹理均值166", "Face texture mean 166", "face_texture"),
    ("left ear wenli mean218", "左耳纹理均值218", "Left ear texture mean 218", "left_ear_texture"),
    ("left ear wenli mean1438", "左耳纹理均值1438", "Left ear texture mean 1438", "left_ear_texture"),
    ("right ear wenli mean1574", "右耳纹理均值1574", "Right ear texture mean 1574", "right_ear_texture"),
    ("right ear wenli variance1608", "右耳纹理方差1608", "Right ear texture variance 1608", "right_ear_texture"),
    ("face wenli variance951", "面部纹理方差951", "Face texture variance 951", "face_texture"),
    ("left ear wenli variance246", "左耳纹理方差246", "Left ear texture variance 246", "left_ear_texture"),
    ("left ear wenli mean214", "左耳纹理均值214", "Left ear texture mean 214", "left_ear_texture"),
    ("right ear wenli mean1184", "右耳纹理均值1184", "Right ear texture mean 1184", "right_ear_texture"),
    ("face wenli variance455", "面部纹理方差455", "Face texture variance 455", "face_texture"),
    ("face wenli variance1007", "面部纹理方差1007", "Face texture variance 1007", "face_texture"),
    ("left ear wenli variance894", "左耳纹理方差894", "Left ear texture variance 894", "left_ear_texture"),
    ("left ear wenli mean217", "左耳纹理均值217", "Left ear texture mean 217", "left_ear_texture"),
    ("right ear wenli mean200", "右耳纹理均值200", "Right ear texture mean 200", "right_ear_texture"),
    ("left ear wenli variance1088", "左耳纹理方差1088", "Left ear texture variance 1088", "left_ear_texture"),
    ("left ear wenli variance262", "左耳纹理方差262", "Left ear texture variance 262", "left_ear_texture"),
    ("left ear wenli mean904", "左耳纹理均值904", "Left ear texture mean 904", "left_ear_texture"),
    ("face wenli variance535", "面部纹理方差535", "Face texture variance 535", "face_texture"),
    ("left ear wenli variance872", "左耳纹理方差872", "Left ear texture variance 872", "left_ear_texture"),
    ("left ear wenli variance848", "左耳纹理方差848", "Left ear texture variance 848", "left_ear_texture"),
    ("face wenli variance1167", "面部纹理方差1167", "Face texture variance 1167", "face_texture"),
    ("right ear wenli variance998", "右耳纹理方差998", "Right ear texture variance 998", "right_ear_texture"),
    ("face wenli variance559", "面部纹理方差559", "Face texture variance 559", "face_texture"),
    ("right ear wenli mean928", "右耳纹理均值928", "Right ear texture mean 928", "right_ear_texture"),
    ("face wenli variance1454", "面部纹理方差1454", "Face texture variance 1454", "face_texture"),
    ("face wenli variance935", "面部纹理方差935", "Face texture variance 935", "face_texture"),
    ("face wenli variance863", "面部纹理方差863", "Face texture variance 863", "face_texture"),
]

OVERALL_FEATURE_NAMES = [spec[0] for spec in FEATURE_SPECS[:15]]

BRANCH_MODEL_SPECS = [
    {
        "id": "lad",
        "model_file": "y1_Ridge.dat",
        "label_zh": "左前降支",
        "label_en": "LAD (Left Anterior Descending)",
        "feature_names": [
            "face guangpu mean1",
            "face wenli mean30",
            "face wenli mean78",
            "face wenli mean94",
            "face wenli mean102",
            "face wenli mean126",
            "face wenli mean166",
            "left ear wenli mean218",
            "left ear wenli mean1438",
            "right ear wenli mean1574",
            "right ear wenli variance1608",
            "face wenli variance455",
            "face wenli variance1007",
            "left ear wenli variance894",
            "left ear wenli mean217",
            "right ear wenli mean200",
        ],
    },
    {
        "id": "lcx",
        "model_file": "y2_Ridge.dat",
        "label_zh": "左回旋支",
        "label_en": "LCX (Left Circumflex)",
        "feature_names": [
            "face guangpu mean1",
            "face wenli mean30",
            "face wenli mean78",
            "face wenli mean94",
            "face wenli mean102",
            "face wenli mean126",
            "face wenli mean166",
            "left ear wenli mean218",
            "left ear wenli mean1438",
            "right ear wenli mean1574",
            "right ear wenli variance1608",
            "left ear wenli variance1088",
            "left ear wenli variance262",
            "left ear wenli mean904",
            "face wenli variance535",
            "left ear wenli variance872",
            "left ear wenli variance848",
            "face wenli variance1167",
        ],
    },
    {
        "id": "rca",
        "model_file": "y3_Ridge.dat",
        "label_zh": "右冠状动脉",
        "label_en": "RCA (Right Coronary Artery)",
        "feature_names": [
            "face guangpu mean1",
            "face wenli mean30",
            "face wenli mean78",
            "face wenli mean94",
            "face wenli mean102",
            "face wenli mean126",
            "face wenli mean166",
            "left ear wenli mean218",
            "left ear wenli mean1438",
            "right ear wenli mean1574",
            "right ear wenli variance1608",
            "right ear wenli variance998",
            "face wenli variance559",
            "right ear wenli mean928",
            "face wenli variance1454",
            "face wenli variance935",
            "face wenli variance863",
        ],
    },
]

FEATURE_GROUP_DEFINITIONS = [
    ("face_spectrum", "面部光谱指标", "Face Spectrum Features"),
    ("face_texture", "面部纹理指标（均值与方差）", "Face Texture Features (Mean & Variance)"),
    ("left_ear_texture", "左耳纹理指标", "Left Ear Texture Features"),
    ("right_ear_texture", "右耳纹理指标", "Right Ear Texture Features"),
]


def build_feature_fields():
    return [
        {
            "name": f"feature_{index}",
            "column": column,
            "index": index + 1,
            "alias": feature_alias(column),
            "label_zh": feature_alias(column),
            "label_en": feature_alias(column),
            "group_id": group_id,
        }
        for index, (column, _label_zh, _label_en, group_id) in enumerate(FEATURE_SPECS)
    ]


def build_feature_groups(feature_fields):
    grouped = {group_id: [] for group_id, _, _ in FEATURE_GROUP_DEFINITIONS}
    for field in feature_fields:
        grouped[field["group_id"]].append(field)

    return [
        {
            "id": group_id,
            "title_zh": title_zh,
            "title_en": title_en,
            "fields": grouped[group_id],
        }
        for group_id, title_zh, title_en in FEATURE_GROUP_DEFINITIONS
    ]


def load_branch_services():
    services = {}
    for spec in BRANCH_MODEL_SPECS:
        path = os.path.join(MODEL_DIR, spec["model_file"])
        services[spec["id"]] = PredictionService.from_shelve(
            path,
            feature_names=spec["feature_names"],
        )
    return services


def predict_branch_qfr(service, feature_map):
    values = [feature_map[name] for name in service.feature_names]
    raw = service.predict_values(values)
    return abs(float(raw))
