# ==================================================
# [01] 라이브러리 불러오기
#
# Path         : 입력 이미지 및 결과 저장 경로 관리
# perf_counter : MeanShift 실행 시간 측정
# cv2          : OpenCV 이미지 처리
# ==================================================

from pathlib import Path
from time import perf_counter

import cv2


# ==================================================
# [02] 입력 이미지 및 결과 저장 경로 설정
#
# MeanShift를 적용할 이미지 파일 경로 지정
# 결과 이미지를 저장할 폴더 경로 지정
# 저장 폴더가 없으면 자동 생성
# ==================================================

# 입력 이미지 경로
image_path = Path(
    "samples/day04/targets/target_normal.jpg"
)

# 결과 저장 폴더
output_dir = Path(
    "outputs/day04/meanshift"
)

# 저장 폴더 자동 생성
output_dir.mkdir(
    parents=True,
    exist_ok=True,
)


# ==================================================
# [03] 원본 이미지 읽기
#
# 이미지 파일을 메모리로 불러오기
#
# 이미지가 존재하지 않거나 읽지 못하면
# FileNotFoundError 발생
#
# 원본 이미지의 가로·세로 크기 확인
# ==================================================

# 이미지 읽기
image = cv2.imread(
    str(image_path)
)

# 이미지 읽기 실패 확인
if image is None:
    raise FileNotFoundError(
        image_path
    )

# 원본 이미지 크기 확인
original_h, original_w = image.shape[:2]

# 원본 이미지 크기 출력
print(
    "Original size:",
    original_w,
    "x",
    original_h,
)


# ==================================================
# [04] MeanShift 처리용 이미지 크기 조절
#
# MeanShift는 주변 픽셀의 위치와 색상을
# 반복적으로 분석하므로 이미지가 클수록
# 처리 시간이 증가할 수 있음
#
# 가로 크기가 900px보다 큰 경우:
#   1. 축소 비율 계산
#   2. 원본 비율을 유지하여 이미지 축소
#
# 가로 크기가 900px 이하이면 기존 크기 유지
#
# 처리 이미지 크기를 출력하고
# 비교할 기준 이미지를 저장
# ==================================================

# 처리용 이미지의 최대 가로 크기
process_width = 900

# 원본 이미지가 900px보다 큰 경우
if original_w > process_width:

    # 원본 크기 대비 축소 비율 계산
    scale = (
        process_width
        / original_w
    )

    # 원본 비율을 유지한 세로 크기 계산
    process_height = int(
        original_h * scale
    )

    # 이미지 축소
    image = cv2.resize(
        image,
        (
            process_width,
            process_height,
        ),
        interpolation=cv2.INTER_AREA,
    )


# 실제 처리할 이미지 크기 확인
height, width = image.shape[:2]

# 처리 이미지 크기 출력
print(
    "Process size :",
    width,
    "x",
    height,
)


# MeanShift 결과 비교용 기준 이미지 저장
cv2.imwrite(
    str(
        output_dir /
        "resized_input.jpg"
    ),
    image,
)


# ==================================================
# [05] MeanShift Parameter 설정
#
# MeanShift는 주변 픽셀의 위치와 색상 유사성을
# 이용하여 비슷한 색상 영역을 부드럽게 만듦
#
# sp (Spatial Radius)
#   → 주변 위치를 얼마나 넓게 탐색할 것인가?
#   → 값이 커질수록 더 넓은 공간을 탐색
#
# sr (Color Radius)
#   → 어느 정도 색상 차이까지 비슷하게 볼 것인가?
#   → 값이 커질수록 색상 차이를 더 넓게 허용
#
# 네 가지 Parameter 조합을 준비하여
# 동일한 이미지에 적용하고 결과 비교
#
# (sp=10, sr=20) : 작은 탐색 범위
# (sp=20, sr=30) : 중간 탐색 범위
# (sp=30, sr=50) : 더 넓은 탐색 범위
# (sp=50, sr=80) : 가장 넓은 탐색 범위
# ==================================================

settings = [
    (10, 20),
    (20, 30),
    (30, 50),
    (50, 80),
]


# ==================================================
# [06] Parameter별 MeanShift 실행
#
# settings에 저장된 네 가지 조합을
# 하나씩 가져와 동일한 이미지에 적용
#
# pyrMeanShiftFiltering()은
# 픽셀의 공간적 위치와 색상 유사성을
# 기준으로 필터링을 수행함
#
# 처리 과정:
#   1. sp, sr 값을 가져오기
#   2. 시작 시간 기록
#   3. MeanShift 필터링 실행
#   4. 실행 시간 계산
#
# 결과:
# 색상이 비슷한 영역이 부드럽게 표현된 이미지
#
# 주의:
# 이 함수는 객체별 Label이나
# 이진 Mask를 직접 생성하지 않음
# ==================================================

# 네 가지 Parameter 조합 반복
for sp, sr in settings:

    # 실행 시작 시간 기록
    start = perf_counter()

    # MeanShift 필터링 실행
    shifted = cv2.pyrMeanShiftFiltering(
        image,
        sp=sp,
        sr=sr,
    )

    # 실행 시간 계산 (밀리초)
    elapsed_ms = (
        perf_counter()
        - start
    ) * 1000


    # ==================================================
    # [07] MeanShift 결과 이미지 저장
    #
    # 적용한 Parameter 값을 파일명에 포함
    #
    # 예:
    # meanshift_sp10_sr20.jpg
    # meanshift_sp20_sr30.jpg
    # meanshift_sp30_sr50.jpg
    # meanshift_sp50_sr80.jpg
    #
    # 결과 이미지를 JPG 파일로 저장
    # ==================================================

    # 결과 파일 경로 생성
    output_path = (
        output_dir /
        f"meanshift_sp{sp}_sr{sr}.jpg"
    )

    # MeanShift 결과 이미지 저장
    cv2.imwrite(
        str(output_path),
        shifted,
    )


    # ==================================================
    # [08] Parameter별 실행 결과 출력
    #
    # 각 Parameter 조합의 처리 시간 확인
    #
    # 출력 정보:
    #   sp   : 공간 탐색 범위
    #   sr   : 색상 탐색 범위
    #   time : MeanShift 처리 시간(ms)
    #
    # 네 가지 설정의 처리 시간을 비교
    # ==================================================

    print(
        f"sp={sp:02d} "
        f"sr={sr:02d} "
        f"time={elapsed_ms:.2f} ms"
    )


# ==================================================
# [09] 최종 결과 저장 경로 출력
#
# 네 가지 Parameter 조합의 처리가 완료되면
# 결과 이미지가 저장된 폴더 경로 출력
#
# 저장 결과:
#   resized_input.jpg
#   meanshift_sp10_sr20.jpg
#   meanshift_sp20_sr30.jpg
#   meanshift_sp30_sr50.jpg
#   meanshift_sp50_sr80.jpg
# ==================================================

print()

print(
    "MeanShift 결과 저장:",
    output_dir
)