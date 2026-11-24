import json
from dataclasses import dataclass
from pathlib import Path

import joblib

# ai-server 폴더 위치 (이 파일 기준으로 한 단계 위)
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"


@dataclass
class ModelBundle:
    """추론에 필요한 것을 한 묶음으로 보관"""
    model: object
    feature_columns: list
    training_info: dict
    project_config: dict


def load_model_bundle() -> ModelBundle:
    model = joblib.load(MODEL_DIR / "rf_action_baseline.joblib")

    with open(MODEL_DIR / "rf_feature_columns.json", encoding="utf-8") as f:
        feature_columns = json.load(f)
    with open(MODEL_DIR / "rf_training_info.json", encoding="utf-8") as f:
        training_info = json.load(f)
    with open(BASE_DIR / "project.json", encoding="utf-8") as f:
        project_config = json.load(f)

    # 검사: 세 파일의 Feature 수가 모두 같아야 함
    n_model = model.n_features_in_
    n_cols = len(feature_columns)
    n_info = training_info["feature_count"]
    if not (n_model == n_cols == n_info):
        raise ValueError(f"Feature 수 불일치: model={n_model}, columns={n_cols}, info={n_info}")

    # 검사: 모델의 Class와 학습 정보의 Class가 같아야 함
    if list(model.classes_) != training_info["classes"]:
        raise ValueError("model.classes_와 training_info의 classes가 다릅니다")

    return ModelBundle(model, feature_columns, training_info, project_config)
