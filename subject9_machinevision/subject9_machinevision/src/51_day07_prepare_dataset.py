
# ==================================================
# ① 필요한 라이브러리 불러오기
#    → random : 정상 이미지 무작위 섞기
#    → shutil : 파일 복사 및 폴더 삭제
#    → Path   : 파일과 폴더 경로 관리
# ==================================================
import random
import shutil
from pathlib import Path


# ==================================================
# ② 원본 및 실습용 데이터 경로 설정
#    → 원본 데이터는 original/bottle에 보존
#    → 실습용 데이터는 samples/day07에 생성
#    → 정상 데이터의 80%를 Train으로 사용
# ==================================================
RAW_ROOT = Path(
    "samples/day07/original/bottle"
)

OUTPUT_ROOT = Path(
    "samples/day07"
)

# 무작위 분리 결과를 재현하기 위한 고정값
SEED = 42

# 정상 데이터의 80%를 학습용으로 사용
TRAIN_RATIO = 0.8

random.seed(SEED)


# ==================================================
# ③ MVTec bottle 원본 데이터 확인
#    → 원본 폴더가 없으면 오류 발생
#    → 데이터가 없는 상태에서 실행하지 않도록 확인
# ==================================================
if not RAW_ROOT.exists():
    raise FileNotFoundError(
        f"MVTec bottle 원본을 확인하세요: {RAW_ROOT}"
    )


# ==================================================
# ④ 이전에 생성한 실습용 폴더 초기화
#    → 이전 실행 결과를 삭제하고 다시 준비
#    → original 폴더는 삭제하지 않음
# ==================================================
generated_dirs = [
    OUTPUT_ROOT / "train",
    OUTPUT_ROOT / "validation",
    OUTPUT_ROOT / "test",
    OUTPUT_ROOT / "ground_truth",
]

for directory in generated_dirs:

    # 해당 폴더가 이미 존재하면 삭제
    if directory.exists():
        shutil.rmtree(
            directory
        )


# ==================================================
# ⑤ 실습에서 사용할 폴더 생성
#    → train      : 정상 기준 생성용
#    → validation : Threshold 결정용
#    → test       : 최종 검사 및 성능 평가용
#    → ground_truth : 결함 위치 정답 Mask
# ==================================================

# 정상 기준 생성에 사용할 폴더
train_dir = (
    OUTPUT_ROOT
    / "train/normal"
)

# 임계값 결정에 사용할 정상 검증 폴더
val_dir = (
    OUTPUT_ROOT
    / "validation/normal"
)

# 최종 검사에 사용할 정상 이미지 폴더
test_normal_dir = (
    OUTPUT_ROOT
    / "test/normal"
)

# 최종 검사에 사용할 이상 이미지 폴더
test_anomaly_dir = (
    OUTPUT_ROOT
    / "test/anomaly"
)

# 결함 위치 정답 Mask 폴더
ground_truth_dir = (
    OUTPUT_ROOT
    / "ground_truth"
)


# 필요한 폴더를 모두 생성
for directory in [
    train_dir,
    val_dir,
    test_normal_dir,
    test_anomaly_dir,
    ground_truth_dir,
]:
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )


# ==================================================
# ⑥ 정상 이미지 Train / Validation 분리
#    → 원본 train/good의 정상 이미지 목록 불러오기
#    → 무작위로 섞은 후 80% / 20%로 분리
#    → 각 폴더에 이미지 복사
# ==================================================

# ----------------------------------------------
# ⑥-1. 원본 정상 이미지 목록 불러오기
# ----------------------------------------------
normal_files = sorted(
    (RAW_ROOT / "train/good").glob("*.png")
)

# 정상 이미지가 없으면 오류 발생
if not normal_files:
    raise FileNotFoundError(
        RAW_ROOT / "train/good"
    )


# ----------------------------------------------
# ⑥-2. 정상 이미지 순서를 무작위로 섞기
# ----------------------------------------------
random.shuffle(
    normal_files
)


# ----------------------------------------------
# ⑥-3. Train / Validation 분리 위치 계산
#       → 80% Train / 나머지 20% Validation
# ----------------------------------------------
split_index = int(
    len(normal_files)
    * TRAIN_RATIO
)

train_files = (
    normal_files[:split_index]
)

val_files = (
    normal_files[split_index:]
)


# ----------------------------------------------
# ⑥-4. Normal Train 폴더에 이미지 복사
# ----------------------------------------------
for path in train_files:

    shutil.copy2(
        path,
        train_dir / path.name,
    )


# ----------------------------------------------
# ⑥-5. Normal Validation 폴더에 이미지 복사
# ----------------------------------------------
for path in val_files:

    shutil.copy2(
        path,
        val_dir / path.name,
    )


# ==================================================
# ⑦ 최종 검사(Test) 이미지 복사
#    → 정상 이미지와 이상 이미지를 구분하여 복사
#    → 원본은 이동하지 않고 그대로 보존
# ==================================================

# ----------------------------------------------
# ⑦-1. Normal Test 이미지 복사
#       → 원본 test/good → test/normal
# ----------------------------------------------
for path in sorted(
    (RAW_ROOT / "test/good").glob("*.png")
):

    shutil.copy2(
        path,
        test_normal_dir / path.name,
    )


# ----------------------------------------------
# ⑦-2. 이상 이미지 종류 설정
#       → 큰 파손 / 작은 파손 / 오염
# ----------------------------------------------
defect_names = [
    "broken_large",
    "broken_small",
    "contamination",
]


# ----------------------------------------------
# ⑦-3. 이상 이미지 종류별 폴더 생성 및 복사
#       → test/anomaly/ 아래에 각각 저장
# ----------------------------------------------
for defect_name in defect_names:

    # 원본 이상 이미지 폴더
    source_dir = (
        RAW_ROOT
        / "test"
        / defect_name
    )

    # 실습용 이상 이미지 폴더
    destination = (
        test_anomaly_dir
        / defect_name
    )

    destination.mkdir(
        parents=True,
        exist_ok=True,
    )

    # 이상 이미지 복사
    for path in sorted(
        source_dir.glob("*.png")
    ):

        shutil.copy2(
            path,
            destination / path.name,
        )


# ==================================================
# ⑧ Ground Truth Mask 복사
#    → 실제 결함 위치가 표시된 정답 이미지 복사
#    → 이후 Heatmap 결과를 비교할 때 사용
# ==================================================
for defect_name in defect_names:

    # 원본 결함 위치 Mask 폴더
    source_dir = (
        RAW_ROOT
        / "ground_truth"
        / defect_name
    )

    # 실습용 Mask 저장 폴더
    destination = (
        ground_truth_dir
        / defect_name
    )

    destination.mkdir(
        parents=True,
        exist_ok=True,
    )

    # 해당 Mask 폴더가 없으면 건너뛰기
    if not source_dir.exists():
        continue

    # Mask 이미지 복사
    for path in sorted(
        source_dir.glob("*.png")
    ):

        shutil.copy2(
            path,
            destination / path.name,
        )


# ==================================================
# ⑨ 데이터 분리 결과 확인
#    → 정상 Train / Validation / Test 개수 출력
#    → 이상 이미지 종류별 개수 출력
#    → 데이터 준비가 완료되었는지 확인
# ==================================================
print("=" * 70)
print("DAY07 DATASET PREPARED")
print("=" * 70)


# ----------------------------------------------
# ⑨-1. Normal Train 이미지 개수 출력
# ----------------------------------------------
print(
    "Normal Train      :",
    len(train_files),
)


# ----------------------------------------------
# ⑨-2. Normal Validation 이미지 개수 출력
# ----------------------------------------------
print(
    "Normal Validation :",
    len(val_files),
)


# ----------------------------------------------
# ⑨-3. Normal Test 이미지 개수 출력
# ----------------------------------------------
print(
    "Normal Test       :",
    len(
        list(
            test_normal_dir.glob("*.png")
        )
    ),
)


# ----------------------------------------------
# ⑨-4. 이상 이미지 종류별 개수 출력
# ----------------------------------------------
for defect_name in defect_names:

    count = len(
        list(
            (
                test_anomaly_dir
                / defect_name
            ).glob("*.png")
        )
    )

    print(
        f"{defect_name:18}:",
        count,
    )