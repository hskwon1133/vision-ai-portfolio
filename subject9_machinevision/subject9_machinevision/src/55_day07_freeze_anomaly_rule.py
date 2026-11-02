
# ==================================================
# ① 필요한 라이브러리 불러오기
#    → json   : 판정 규칙을 JSON 파일로 저장
#    → Path   : 입력 및 출력 파일 경로 관리
#    → NumPy  : Percentile을 이용한 Threshold 계산
#    → Pandas : Validation Score CSV 읽기
# ==================================================
import json
from pathlib import Path

import numpy as np
import pandas as pd


# ==================================================
# ② 입력 및 출력 파일 경로 설정
#    → 입력 : 53번에서 계산한 Validation Score
#    → 출력 : 확정한 Threshold와 판정 규칙
# ==================================================

# 정상 Validation 이미지 42장의 Score 파일
VALIDATION_CSV = Path(
    "reports/day07_validation_scores.csv"
)

# 최종 판정 규칙을 저장할 JSON 파일
RULE_PATH = Path(
    "configs/day07_anomaly_rules.json"
)

# ==================================================
# ③ 사용할 Threshold 기준 선택
#    → 54번에서 90 / 95 / 99 Percentile 비교
#    → 이번 Baseline에서는 95 Percentile 선택
#
#    주의!
#    95는 정확도 95%를 의미하지 않습니다.
#    정상 Score 분포의 95백분위수를 의미합니다.
# ==================================================
SELECTED_PERCENTILE = 95


# ==================================================
# ④ 정상 Validation Score 불러오기
#    → CSV 파일 존재 여부 확인
#    → anomaly_score 컬럼 확인
#    → 정상 이미지의 Score를 NumPy 배열로 변환
# ==================================================

# ----------------------------------------------
# ④-1. Validation Score 파일 확인
# ----------------------------------------------
if not VALIDATION_CSV.exists():
    raise FileNotFoundError(
        f"Validation Score 파일을 확인하세요: {VALIDATION_CSV}"
    )


# ----------------------------------------------
# ④-2. CSV 파일 불러오기
# ----------------------------------------------
df = pd.read_csv(
    VALIDATION_CSV
)


# ----------------------------------------------
# ④-3. anomaly_score 컬럼 존재 여부 확인
# ----------------------------------------------
if "anomaly_score" not in df.columns:
    raise RuntimeError(
        "anomaly_score 컬럼이 없습니다."
    )


# ----------------------------------------------
# ④-4. Anomaly Score만 NumPy 배열로 변환
# ----------------------------------------------
scores = df[
    "anomaly_score"
].to_numpy()


# ----------------------------------------------
# ④-5. Score가 비어 있는지 확인
# ----------------------------------------------
if len(scores) == 0:
    raise RuntimeError(
        "Validation Score가 없습니다."
    )


# ==================================================
# ⑤ 95 Percentile Threshold 계산
#    → 정상 이미지 Score의 95백분위수 계산
#    → 소수점 6자리로 반올림하여 저장
#
#    예: 0.067469
#    실제 값은 Validation CSV에 따라 결정됩니다.
# ==================================================
threshold = round(
    float(
        np.percentile(
            scores,
            SELECTED_PERCENTILE,
        )
    ),
    6,
)


# ==================================================
# ⑥ Baseline 판정 규칙 구성
#    → 어떤 모델과 정상 기준을 사용하는지 기록
#    → 선택한 Percentile과 Threshold 기록
#    → 정상 / 이상 판정 조건 기록
#
#    Score <= Threshold → Normal
#    Score >  Threshold → Anomaly
# ==================================================
rules = {

    # Baseline 버전
    "baseline_version": "day07-anomaly-v1.0",

    # 사용 데이터셋
    "dataset": "MVTec AD bottle",

    # 이미지 특징 추출에 사용할 모델
    "backbone": "resnet18_imagenet_pretrained",

    # ResNet18 특징 추출 위치
    "feature_layer": "layer3",

    # 정상 기준 생성 방법
    "reference": "mean_normal_feature",

    # 52번 코드에서 저장한 정상 기준 파일
    "reference_path": "artifacts/day07/normal_reference.npz",

    # ResNet18 입력 이미지 크기
    "image_size": 224, ## ResNet18 입력 이미지 크기 / 이미지 줄여서 계산

    # 선택한 Percentile
    "selected_percentile": SELECTED_PERCENTILE,

    # Threshold 계산 방법
    "threshold_method": "normal_validation_95_percentile",

    # 실제 계산된 Threshold 값
    "anomaly_threshold": threshold,

    # 이상 판정 규칙
    "decision_rule": "score > threshold => Anomaly",
}


# ==================================================
# ⑦ 확정한 Baseline Rule을 JSON 파일로 저장
#    → 저장 폴더가 없으면 생성
#    → 판정 기준과 설정 정보를 JSON으로 저장
#
#    Rule Freeze란?
#    → 이후 Test에서 동일한 기준을 사용하도록
#      현재 선택한 판정 규칙을 파일로 고정하는 것
# ==================================================

# ----------------------------------------------
# ⑦-1. configs 폴더가 없으면 생성
# ----------------------------------------------
RULE_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)


# ----------------------------------------------
# ⑦-2. 판정 규칙을 JSON으로 저장
#       → indent=2 : 읽기 쉽게 들여쓰기
#       → ensure_ascii=False : 한글 그대로 저장
# ----------------------------------------------
RULE_PATH.write_text(
    json.dumps(
        rules,
        indent=2,
        ensure_ascii=False,
    ),
    encoding="utf-8",
)


# ==================================================
# ⑧ 실행 결과 확인
#    → Baseline 버전 출력
#    → 선택한 Percentile 출력
#    → 확정된 Threshold 출력
#    → 판정 규칙 및 저장 위치 출력
# ==================================================

# ----------------------------------------------
# ⑧-1. Rule Freeze 완료 메시지
# ----------------------------------------------
print("=" * 70)
print("DAY07 RULE FREEZE")
print("=" * 70)


# ----------------------------------------------
# ⑧-2. Baseline 버전 출력
# ----------------------------------------------
print(
    "Version   :",
    rules["baseline_version"],
)


# ----------------------------------------------
# ⑧-3. 선택한 Percentile 출력
# ----------------------------------------------
print(
    "Percentile:",
    rules["selected_percentile"],
)


# ----------------------------------------------
# ⑧-4. 확정된 Threshold 출력
# ----------------------------------------------
print(
    "Threshold :",
    rules["anomaly_threshold"],
)


# ----------------------------------------------
# ⑧-5. 이상 판정 규칙 출력
# ----------------------------------------------
print(
    "Rule      :",
    rules["decision_rule"],
)


# ----------------------------------------------
# ⑧-6. JSON 저장 경로 출력
# ----------------------------------------------
print(
    "Saved     :",
    RULE_PATH,
)