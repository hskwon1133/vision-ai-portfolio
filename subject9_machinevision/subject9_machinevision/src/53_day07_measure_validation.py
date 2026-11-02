# ==================================================
# ① 필요한 라이브러리와 공통 함수 불러오기
#    → csv : 계산한 점수를 CSV 파일로 저장
#    → Path : 입력 및 출력 파일 경로 관리
#    → 공통 함수 : 특징 추출 및 거리 계산
# ==================================================
import csv
from pathlib import Path

from day07_anomaly_core import (
    anomaly_score,
    create_backbone,
    extract_features,
    load_reference,
)


# ==================================================
# ② 입력 및 출력 경로 설정
#    → Validation : 정상 검증 이미지 42장
#    → Reference  : 52번에서 저장한 정상 기준
#    → CSV        : 이미지별 이상 점수 저장
# ==================================================

# 정상 검증 이미지 폴더
VAL_DIR = Path(
    "samples/day07/validation/normal"
)

# 52번 코드에서 생성한 정상 기준 파일
REFERENCE_PATH = Path(
    "artifacts/day07/normal_reference.npz"
)

# Anomaly Score 결과 저장 파일
OUTPUT_CSV = Path(
    "reports/day07_validation_scores.csv"
)

# ==================================================
# ③ 필요한 데이터 확인
#    → Validation 폴더 존재 여부 확인
#    → Normal Reference 파일 존재 여부 확인
#    → Validation 이미지 목록 불러오기
# ==================================================

# ----------------------------------------------
# ③-1. Validation 폴더 확인
# ----------------------------------------------
if not VAL_DIR.exists():
    raise FileNotFoundError(
        f"Validation 폴더를 확인하세요: {VAL_DIR}"
    )


# ----------------------------------------------
# ③-2. Normal Reference 파일 확인
# ----------------------------------------------
if not REFERENCE_PATH.exists():
    raise FileNotFoundError(
        f"Normal Reference를 확인하세요: {REFERENCE_PATH}"
    )


# ----------------------------------------------
# ③-3. 정상 검증 이미지 목록 불러오기
# ----------------------------------------------
image_files = sorted(
    VAL_DIR.glob("*.png")
)

# 이미지가 없으면 오류 발생
if not image_files:
    raise RuntimeError(
        "Validation 이미지가 없습니다."
    )

# ==================================================
# ④ 저장된 Normal Reference 불러오기
#    → 공통 함수 load_reference() 사용
#    → 전체 비교용 reference_vector만 사용
#    → reference_map은 이번 코드에서 사용하지 않음
# ==================================================
reference_vector, _ = load_reference(
    REFERENCE_PATH
)


# ==================================================
# ⑤ Feature 추출용 ResNet18 준비
#    → 공통 함수 create_backbone() 사용
#    → 사전학습된 ResNet18의 layer3까지 사용
#    → 새로운 학습 없이 이미지 특징만 추출
# ==================================================
model = create_backbone()


# ==================================================
# ⑥ 정상 Validation 이미지의 Anomaly Score 계산
#    → 각 이미지에서 Feature Vector 추출
#    → 정상 기준과 유클리드 거리 계산
#    → 이미지별 점수를 리스트에 기록
# ==================================================

# CSV에 저장할 결과 리스트
rows = []

# 검사할 정상 검증 이미지 개수 출력
print(
    "Normal Validation:",
    len(image_files),
)


# ----------------------------------------------
# ⑥-1. Validation 이미지 한 장씩 처리
# ----------------------------------------------
for index, image_path in enumerate(
    image_files,
    start=1,
):

    # ------------------------------------------
    # ⑥-2. 이미지 전체 Feature Vector 추출
    #       → 위치별 Feature Map은 사용하지 않음
    # ------------------------------------------
    vector, _ = extract_features(
        model,
        image_path,
    )

    # ------------------------------------------
    # ⑥-3. Normal Reference와 거리 계산
    #       → 거리 = Anomaly Score
    #
    #       점수 낮음 : 정상 기준과 비슷함
    #       점수 높음 : 정상 기준과 많이 다름
    # ------------------------------------------
    score = anomaly_score(
        vector,
        reference_vector,
    )

    # ------------------------------------------
    # ⑥-4. 이미지 이름과 계산 결과 기록
    #       → 모두 정상 검증 이미지이므로
    #         실제 정답(actual)은 Normal
    #       → 점수는 소수점 6자리로 반올림
    # ------------------------------------------
    rows.append(
        {
            "file": image_path.name,
            "actual": "Normal",
            "anomaly_score": round(
                score,
                6,
            ),
        }
    )

    # ------------------------------------------
    # ⑥-5. 진행 상황 출력
    #       → 첫 번째, 10장마다, 마지막 이미지
    # ------------------------------------------
    if (
        index == 1
        or index % 10 == 0
        or index == len(image_files)
    ):
        print(
            f"[{index}/{len(image_files)}]",
            image_path.name,
        )


# ==================================================
# ⑦ 계산한 Anomaly Score를 CSV 파일로 저장
#    → 이미지 이름, 실제 정답, 이상 점수 기록
#    → 이후 Threshold를 결정할 때 사용
# ==================================================

# ----------------------------------------------
# ⑦-1. 결과 저장 폴더 생성
# ----------------------------------------------
OUTPUT_CSV.parent.mkdir(
    parents=True,
    exist_ok=True,
)


# ----------------------------------------------
# ⑦-2. CSV 파일 열기
#       → utf-8-sig : Excel 한글 호환성 고려
# ----------------------------------------------
with OUTPUT_CSV.open(
    "w",
    newline="",
    encoding="utf-8-sig",
) as file:

    # ------------------------------------------
    # ⑦-3. CSV 열 이름 설정
    # ------------------------------------------
    writer = csv.DictWriter(
        file,
        fieldnames=[
            "file",
            "actual",
            "anomaly_score",
        ],
    )

    # ------------------------------------------
    # ⑦-4. CSV 헤더 작성
    # ------------------------------------------
    writer.writeheader()

    # ------------------------------------------
    # ⑦-5. 이미지별 Anomaly Score 저장
    # ------------------------------------------
    writer.writerows(rows)


# ==================================================
# ⑧ 실행 결과 확인
#    → Validation Score 측정 완료 메시지 출력
#    → 처리한 이미지 개수와 CSV 저장 위치 확인
# ==================================================
print("=" * 70)
print("Validation Score 측정 완료")
print("=" * 70)


# ----------------------------------------------
# ⑧-1. 처리한 정상 검증 이미지 개수 출력
# ----------------------------------------------
print(
    "Images:",
    len(rows),
)


# ----------------------------------------------
# ⑧-2. 결과 CSV 파일 경로 출력
# ----------------------------------------------
print(
    "Saved :",
    OUTPUT_CSV,
)