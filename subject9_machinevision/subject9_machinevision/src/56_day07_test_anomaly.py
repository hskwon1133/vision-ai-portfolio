
# ==================================================
# ① 필요한 라이브러리와 공통 함수 불러오기
#    → csv  : 검사 결과를 CSV로 저장
#    → json : 55번에서 확정한 판정 규칙 읽기
#    → Path : 이미지 및 파일 경로 관리
#    → 공통 함수 : 특징 추출, 정상 기준 비교
# ==================================================
import csv
import json
from pathlib import Path

from day07_anomaly_core import (
    anomaly_score,
    create_backbone,
    extract_features,
    load_reference,
)


# ==================================================
# ② 입력 및 출력 경로 설정
#    → 정상 Test 이미지와 이상 Test 이미지 사용
#    → 52번의 정상 기준, 55번의 판정 규칙 사용
#    → 검사 결과는 CSV 파일로 저장
# ==================================================

# 정상 Test 이미지 폴더
TEST_NORMAL_DIR = Path(
    "samples/day07/test/normal"
)

# 이상 Test 이미지 폴더
TEST_ANOMALY_DIR = Path(
    "samples/day07/test/anomaly"
)

# 52번에서 생성한 Normal Reference
REFERENCE_PATH = Path(
    "artifacts/day07/normal_reference.npz"
)

# 55번에서 확정한 Threshold와 판정 규칙
RULE_PATH = Path(
    "configs/day07_anomaly_rules.json"
)

# Test 이미지별 검사 결과 저장 파일
OUTPUT_CSV = Path(
    "reports/day07_anomaly_scores.csv"
)


# ==================================================
# ③ 검사에 필요한 파일과 폴더 확인
#    → Test 데이터가 있는지 확인
#    → 정상 기준과 판정 규칙 파일이 있는지 확인
#    → 없으면 오류를 발생시켜 검사 중단
# ==================================================
required_paths = [
    TEST_NORMAL_DIR,
    TEST_ANOMALY_DIR,
    REFERENCE_PATH,
    RULE_PATH,
]

for path in required_paths:

    if not path.exists():
        raise FileNotFoundError(
            f"필요한 파일 또는 폴더를 확인하세요: {path}"
        )


# ==================================================
# ④ 55번에서 확정한 Rule 불러오기
#    → JSON 파일에서 anomaly_threshold 읽기
#    → Test 데이터로 Threshold를 다시 계산하지 않음
#
#    예: Threshold = 0.067469
# ==================================================

# ----------------------------------------------
# ④-1. 저장된 JSON 판정 규칙 읽기
# ----------------------------------------------
rules = json.loads(
    RULE_PATH.read_text(
        encoding="utf-8"
    )
)

# ----------------------------------------------
# ④-2. 확정된 Threshold 가져오기
# ----------------------------------------------
threshold = float(
    rules["anomaly_threshold"]
)


# ==================================================
# ⑤ Normal Reference와 ResNet18 준비
#    → 정상 기준 Feature Vector 불러오기
#    → 사전학습된 ResNet18 특징 추출기 준비
#    → 검사 결과를 기록할 빈 리스트 생성
# ==================================================

# ----------------------------------------------
# ⑤-1. 정상 기준 Feature Vector 불러오기
#       → 이번 검사는 이미지 전체 특징 비교
#       → reference_map은 사용하지 않음
# ----------------------------------------------
reference_vector, _ = load_reference(
    REFERENCE_PATH
)

# ----------------------------------------------
# ⑤-2. Feature 추출용 ResNet18 준비
# ----------------------------------------------
model = create_backbone()

# ----------------------------------------------
# ⑤-3. 검사 결과를 저장할 리스트 준비
# ----------------------------------------------
rows = []


# ==================================================
# ⑥ 이미지 한 장을 검사하는 함수 작성
#    → 병 이미지의 Feature Vector 추출
#    → 정상 기준과 거리 계산
#    → Threshold로 정상 / 이상 판정
#    → 실제 정답과 비교하여 TP/TN/FP/FN 기록
# ==================================================
def inspect_image(
    image_path,
    actual,
    defect_type,
):

    # ----------------------------------------------
    # ⑥-1. 검사 이미지의 Feature Vector 추출
    #       → 공통 함수 extract_features() 사용
    # ----------------------------------------------
    vector, _ = extract_features(
        model,
        image_path,
    )

    # ----------------------------------------------
    # ⑥-2. 정상 기준과 유클리드 거리 계산
    #       → Anomaly Score 생성
    #
    #       Score 낮음 : 정상 기준과 비슷함
    #       Score 높음 : 정상 기준과 많이 다름
    # ----------------------------------------------
    score = anomaly_score(
        vector,
        reference_vector,
    )

    # ----------------------------------------------
    # ⑥-3. Threshold와 비교하여 판정
    #
    #       Score >  Threshold → Anomaly
    #       Score <= Threshold → Normal
    # ----------------------------------------------
    predicted = (
        "Anomaly"
        if score > threshold
        else "Normal"
    )

    # ----------------------------------------------
    # ⑥-4. 실제 정답과 예측 결과 비교
    #
    #       TP : 실제 이상 → 이상으로 판정 (정답)
    #       TN : 실제 정상 → 정상으로 판정 (정답)
    #       FP : 실제 정상 → 이상으로 판정 (오탐)
    #       FN : 실제 이상 → 정상으로 판정 (미검출)
    #
    #       여기서는 Anomaly를 Positive로 취급
    # ----------------------------------------------

    # 실제 이상을 이상으로 정확히 판정
    if (
        actual == "Anomaly"
        and predicted == "Anomaly"
    ):
        error_type = "TP"

    # 실제 정상을 정상으로 정확히 판정
    elif (
        actual == "Normal"
        and predicted == "Normal"
    ):
        error_type = "TN"

    # 정상 병을 이상으로 잘못 판정
    elif (
        actual == "Normal"
        and predicted == "Anomaly"
    ):
        error_type = "FP"

    # 이상 병을 정상으로 잘못 판정
    else:
        error_type = "FN"

    # ----------------------------------------------
    # ⑥-5. 검사 결과를 리스트에 기록
    #
    #       file          : 이미지 경로
    #       actual        : 실제 정답
    #       defect_type   : 결함 종류
    #       anomaly_score : 계산된 이상 점수
    #       threshold     : 확정된 임계값
    #       predicted     : 모델의 판정 결과
    #       error_type    : TP / TN / FP / FN
    #       correct       : 판정 정답 여부
    # ----------------------------------------------
    rows.append(
        {
            "file": str(image_path),
            "actual": actual,
            "defect_type": defect_type,
            "anomaly_score": round(
                score,
                6,
            ),
            "threshold": round(
                threshold,
                6,
            ),
            "predicted": predicted,
            "error_type": error_type,
            "correct": actual == predicted,
        }
    )


# ==================================================
# ⑦ Normal Test 이미지 검사
#    → test/normal 폴더의 정상 이미지 검사
#    → 실제 정답은 모두 Normal
#    → defect_type은 good으로 기록
# ==================================================
for image_path in sorted(
    TEST_NORMAL_DIR.glob("*.png")
):

    inspect_image(
        image_path,
        actual="Normal",
        defect_type="good",
    )


# ==================================================
# ⑧ Anomaly Test 이미지 검사
#    → test/anomaly 아래 결함 종류별 폴더 순회
#    → broken_large / broken_small / contamination
#    → 실제 정답은 모두 Anomaly
# ==================================================

# ----------------------------------------------
# ⑧-1. 이상 종류별 폴더 순회
# ----------------------------------------------
for defect_dir in sorted(
    TEST_ANOMALY_DIR.iterdir()
):

    # 폴더가 아니면 건너뛰기
    if not defect_dir.is_dir():
        continue

    # ------------------------------------------
    # ⑧-2. 결함 이미지 한 장씩 검사
    #       → 결함 폴더명을 defect_type으로 기록
    # ------------------------------------------
    for image_path in sorted(
        defect_dir.glob("*.png")
    ):

        inspect_image(
            image_path,
            actual="Anomaly",
            defect_type=defect_dir.name,
        )


# ==================================================
# ⑨ 검사 결과 확인 및 CSV 저장
#    → 검사 결과가 비어 있는지 확인
#    → 이미지별 판정 결과를 CSV에 저장
#    → 이후 57번 성능 평가에서 사용
# ==================================================

# ----------------------------------------------
# ⑨-1. 검사 결과 존재 여부 확인
# ----------------------------------------------
if not rows:
    raise RuntimeError(
        "검사할 Test 이미지가 없습니다."
    )


# ----------------------------------------------
# ⑨-2. CSV 저장 폴더 생성
# ----------------------------------------------
OUTPUT_CSV.parent.mkdir(
    parents=True,
    exist_ok=True,
)


# ----------------------------------------------
# ⑨-3. 검사 결과 CSV 저장
# ----------------------------------------------
with OUTPUT_CSV.open(
    "w",
    newline="",
    encoding="utf-8-sig",
) as file:

    # 저장할 CSV 컬럼 지정
    writer = csv.DictWriter(
        file,
        fieldnames=[
            "file",
            "actual",
            "defect_type",
            "anomaly_score",
            "threshold",
            "predicted",
            "error_type",
            "correct",
        ],
    )

    # CSV 컬럼명 작성
    writer.writeheader()

    # 전체 검사 결과 저장
    writer.writerows(rows)


# ==================================================
# ⑩ 실행 결과 출력
#    → Test 검사 완료 메시지 출력
#    → 사용한 Threshold 확인
#    → 검사한 이미지 개수 확인
#    → 결과 CSV 저장 경로 확인
# ==================================================
print("=" * 70)
print("DAY07 TEST COMPLETE")
print("=" * 70)


# ----------------------------------------------
# ⑩-1. 검사에 사용한 Threshold 출력
# ----------------------------------------------
print(
    "Threshold:",
    threshold,
)


# ----------------------------------------------
# ⑩-2. 검사한 이미지 개수 출력
# ----------------------------------------------
print(
    "Images   :",
    len(rows),
)


# ----------------------------------------------
# ⑩-3. 결과 CSV 저장 경로 출력
# ----------------------------------------------
print(
    "Saved    :",
    OUTPUT_CSV,
)