# ==================================================
# [01] 라이브러리 불러오기
#
# Path         : 이미지 파일 및 저장 경로 관리
# perf_counter : GrabCut 실행 시간 측정
# cv2          : OpenCV 이미지 처리
# numpy        : Mask 및 배열 연산
# ==================================================

from pathlib import Path
from time import perf_counter

import cv2
import numpy as np


# ==================================================
# [02] 입력 이미지 및 결과 저장 경로 설정
#
# GrabCut으로 전경을 분리할 이미지 경로 지정
# 결과 이미지를 저장할 폴더 생성
# ==================================================

image_path = Path(
    "samples/day04/source/grabcut_scene.jpg"
)

output_dir = Path(
    "outputs/day04/grabcut"
)

# 저장 폴더가 없으면 자동 생성
output_dir.mkdir(
    parents=True,
    exist_ok=True,
)


# ==================================================
# [03] 원본 이미지 읽기
#
# 이미지 파일을 메모리로 불러오기
# 이미지가 존재하지 않으면 오류 발생
# 원본 이미지의 가로·세로 크기 확인
# ==================================================

image = cv2.imread(
    str(image_path)
)

if image is None:
    raise FileNotFoundError(
        image_path
    )

# 이미지 크기 확인
original_h, original_w = image.shape[:2]

print(
    "Original size:",
    original_w,
    "x",
    original_h,
)


# ==================================================
# [04] GrabCut 처리용 이미지 크기 조절
#
# 원본 이미지가 너무 크면 처리 시간이 증가함
#
# 가로 크기가 900px보다 클 경우:
#   1. 축소 비율 계산
#   2. 원본 비율을 유지하여 이미지 축소
#   3. 축소된 이미지 저장
#
# 가로 크기가 900px 이하이면 원본 크기 유지
# ==================================================

process_width = 900

if original_w > process_width:

    # 원본 가로 크기 대비 축소 비율
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


# 실제 처리할 이미지 크기
height, width = image.shape[:2]

print(
    "Process size :",
    width,
    "x",
    height,
)


# 처리용 이미지 저장
cv2.imwrite(
    str(
        output_dir /
        "resized_input.jpg"
    ),
    image,
)


# ==================================================
# [05] GrabCut 초기 Rectangle 영역 설정
#
# GrabCut은 객체가 포함된 사각형 영역을 기준으로
# 전경과 배경을 추정함
#
# 이번 실습에서는 사람이 마우스로 ROI를 지정하지 않고
# 이미지 크기와 여백 비율로 Rectangle을 자동 생성
#
# wide   : 여백 3%  → 가장 넓은 Rectangle
# medium : 여백 7%  → 중간 크기 Rectangle
# tight  : 여백 12% → 가장 좁은 Rectangle
#
# 세 가지 Rectangle 조건을 비교하여
# 초기 영역이 결과에 미치는 영향을 확인
# ==================================================

margin_ratios = [
    ("wide", 0.03),
    ("medium", 0.07),
    ("tight", 0.12),
]


# 세 가지 Rectangle 조건 반복 실행
for name, margin_ratio in margin_ratios:

    # 이미지 크기를 기준으로 여백 계산
    margin_x = int(
        width * margin_ratio
    )

    margin_y = int(
        height * margin_ratio
    )

    # Rectangle 시작 좌표
    rect_x = margin_x
    rect_y = margin_y

    # Rectangle 가로·세로 크기 계산
    rect_w = (
        width
        - 2 * margin_x
    )

    rect_h = (
        height
        - 2 * margin_y
    )

    # Rectangle 크기가 유효하지 않으면 건너뜀
    if rect_w <= 0 or rect_h <= 0:
        continue

    # GrabCut에 전달할 사각형 정보
    # (x, y, width, height)
    rectangle = (
        rect_x,
        rect_y,
        rect_w,
        rect_h,
    )


    # ==================================================
    # [06] GrabCut용 Mask와 내부 모델 준비
    #
    # GrabCut은 Mask를 이용하여 각 픽셀이
    # 전경인지 배경인지 구분함
    #
    # mask     : 전경/배경 분류 정보를 저장할 배열
    # bg_model : 배경의 색상 분포를 추정하는 내부 모델
    # fg_model : 전경의 색상 분포를 추정하는 내부 모델
    #
    # GC_INIT_WITH_RECT 방식에서는
    # Rectangle을 기준으로 초기 Mask가 설정됨
    # ==================================================

    # 이미지와 동일한 크기의 Mask 생성
    mask = np.zeros(
        (height, width),
        dtype=np.uint8,
    )

    # 배경 모델 초기화
    bg_model = np.zeros(
        (1, 65),
        dtype=np.float64,
    )

    # 전경 모델 초기화
    fg_model = np.zeros(
        (1, 65),
        dtype=np.float64,
    )


    # ==================================================
    # [07] GrabCut 알고리즘 실행
    #
    # Rectangle 바깥쪽:
    #   → 확실한 배경으로 처리
    #
    # Rectangle 안쪽:
    #   → 색상과 주변 픽셀 관계를 분석하여
    #      전경과 배경을 추정
    #
    # 3번의 반복(iteration)을 통해 분할 결과 개선
    # GrabCut 처리 시간 측정
    # ==================================================

    # 실행 시작 시간
    start = perf_counter()

    # GrabCut 실행
    cv2.grabCut(
        image,
        mask,
        rectangle,
        bg_model,
        fg_model,
        3,
        cv2.GC_INIT_WITH_RECT,
    )

    # 실행 시간 계산 (밀리초)
    elapsed_ms = (
        perf_counter()
        - start
    ) * 1000


    # ==================================================
    # [08] GrabCut 결과를 흑백 Mask로 변환
    #
    # GrabCut Mask에는 네 가지 분류값이 존재함
    #
    # GC_BGD    : 확실한 배경
    # GC_FGD    : 확실한 전경
    # GC_PR_BGD : 배경일 가능성이 높은 영역
    # GC_PR_FGD : 전경일 가능성이 높은 영역
    #
    # 확실한 전경 + 전경 가능성이 높은 영역
    #   → 흰색(255)
    #
    # 나머지 배경 영역
    #   → 검은색(0)
    #
    # 최종적으로 이진 Mask 생성
    # ==================================================

    raw_mask = np.where(
        (mask == cv2.GC_FGD)
        |
        (mask == cv2.GC_PR_FGD),
        255,
        0,
    ).astype(
        np.uint8
    )


    # ==================================================
    # [09] 가장 큰 Contour를 이용한 Mask 정리
    #
    # Raw Mask에는 작은 잡음이나 내부 구멍이
    # 존재할 수 있음
    #
    # 1. 외곽 Contour 검출
    # 2. 가장 큰 Contour 선택
    # 3. Contour 내부를 흰색으로 채움
    # 4. Clean Mask 생성
    #
    # 광천김 포장지처럼 하나의 큰 객체를
    # 분리하는 경우에 활용할 수 있음
    #
    # 주의:
    # 가장 큰 객체만 남기므로 작은 객체는 제거됨
    # ==================================================

    # Mask에서 바깥쪽 Contour 검출
    contours, _ = cv2.findContours(
        raw_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    # 정리된 Mask를 저장할 빈 배열 생성
    clean_mask = np.zeros_like(
        raw_mask
    )

    # Contour가 존재하는 경우
    if contours:

        # 면적이 가장 큰 Contour 선택
        largest_contour = max(
            contours,
            key=cv2.contourArea,
        )

        # 가장 큰 Contour 내부를 흰색으로 채움
        cv2.drawContours(
            clean_mask,
            [largest_contour],
            -1,
            255,
            thickness=cv2.FILLED,
        )


    # ==================================================
    # [10] Clean Mask를 이미지에 적용
    #
    # bitwise_and()를 이용하여
    # Clean Mask의 흰색 영역만 유지
    #
    # Mask 흰색(255):
    #   → 원본 픽셀 유지
    #
    # Mask 검은색(0):
    #   → 픽셀 제거 (검은색으로 표시)
    #
    # 결과:
    # 배경이 검은색인 전경 이미지 생성
    # ==================================================

    foreground = cv2.bitwise_and(
        image,
        image,
        mask=clean_mask,
    )


    # ==================================================
    # [11] Rectangle 위치 확인 이미지 생성
    #
    # GrabCut에 사용한 초기 사각형을
    # 이미지 위에 표시
    #
    # 원본 처리 이미지 복사
    # Rectangle을 노란색 선으로 표시
    #
    # 세 가지 조건의 사각형 크기를 비교
    # ==================================================

    # 이미지 복사
    preview = image.copy()

    # GrabCut 초기 Rectangle 표시
    cv2.rectangle(
        preview,
        (rect_x, rect_y),
        (
            rect_x + rect_w,
            rect_y + rect_h,
        ),
        (0, 255, 255),
        3,
    )


    # ==================================================
    # [12] 결과 저장 및 전경 비율 계산
    #
    # 조건별 네 가지 결과 이미지 저장
    #
    # 1. rect.jpg       : Rectangle 확인 이미지
    # 2. raw_mask.png   : GrabCut의 이진 Mask
    # 3. clean_mask.png : 정리된 Mask
    # 4. foreground.jpg : 전경 추출 결과
    #
    # 전체 이미지에서 전경 픽셀이
    # 차지하는 비율 계산
    #
    # 조건별 전경 비율과 실행 시간 출력
    # ==================================================

    # Rectangle 확인 이미지 저장
    cv2.imwrite(
        str(
            output_dir /
            f"{name}_rect.jpg"
        ),
        preview,
    )

    # GrabCut Raw Mask 저장
    cv2.imwrite(
        str(
            output_dir /
            f"{name}_raw_mask.png"
        ),
        raw_mask,
    )

    # Contour로 정리한 Clean Mask 저장
    cv2.imwrite(
        str(
            output_dir /
            f"{name}_clean_mask.png"
        ),
        clean_mask,
    )

    # 전경 추출 이미지 저장
    cv2.imwrite(
        str(
            output_dir /
            f"{name}_foreground.jpg"
        ),
        foreground,
    )

    # 전체 픽셀 중 전경 픽셀의 비율 계산
    foreground_ratio = (
        cv2.countNonZero(
            clean_mask
        )
        / clean_mask.size
    )

    # 조건별 검사 결과 출력
    print(
        f"{name:6s}",
        f"foreground_ratio={foreground_ratio:.4f}",
        f"time={elapsed_ms:.2f} ms",
    )