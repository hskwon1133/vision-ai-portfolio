
# ============================================================
# ① 필요한 라이브러리 불러오기
# - Path  : 이미지 파일 및 저장 폴더 경로 처리
# - cv2   : 이진화, Morphology, Contour 검출 및 측정
# - numpy : 이미지 배열 처리
# ============================================================

from pathlib import Path

import cv2
import numpy as np


# ============================================================
# ② 입력 이미지 및 결과 저장 경로 설정
# - cap_on.jpg : 병뚜껑이 있는 기준 ROI 이미지
# - output_dir : Contour 검사 결과 저장 폴더
# - 저장 폴더가 없으면 자동으로 생성합니다.
# ============================================================

image_path = Path(
    "samples/day03/reference/cap_on.jpg"
)

output_dir = Path(
    "outputs/day03/contour"
)

output_dir.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# ③ 병뚜껑 이미지 불러오기 및 오류 확인
# - imread()로 기준 이미지를 읽습니다.
# - 이미지가 없거나 읽지 못하면 오류를 발생시킵니다.
# ============================================================

image = cv2.imread(
    str(image_path)
)

if image is None:
    raise FileNotFoundError(
        f"이미지를 읽지 못했습니다: {image_path}"
    )


# ============================================================
# ④ ROI 이미지 확인 및 Gray 변환
# - 이미 ROI로 잘린 이미지이므로 다시 Crop하지 않습니다.
# - BGR 컬러 이미지를 Gray 이미지로 변환합니다.
# ============================================================

roi = image

gray = cv2.cvtColor(
    roi,
    cv2.COLOR_BGR2GRAY,
)


# ============================================================
# ⑤ Otsu 이진화
# - Otsu 알고리즘으로 밝기 기준값을 자동 계산합니다.
# - BINARY_INV로 어두운 병뚜껑을 흰색 객체로 변환합니다.
# ============================================================

_, binary = cv2.threshold(
    gray,
    0,
    255,
    cv2.THRESH_BINARY_INV
    + cv2.THRESH_OTSU,
)


# ============================================================
# ⑥ Morphology로 이진 이미지 정리
# - 3×3 타원형 Kernel을 생성합니다.
# - Opening : 작은 흰색 노이즈 제거
# - Closing : 작은 검은색 틈이나 구멍 정리
# ============================================================

kernel = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE,
    (3, 3),
)

opened = cv2.morphologyEx(
    binary,
    cv2.MORPH_OPEN,
    kernel,
    iterations=1,
)

cleaned = cv2.morphologyEx(
    opened,
    cv2.MORPH_CLOSE,
    kernel,
    iterations=1,
)


# ============================================================
# ⑦ Contour 검출 및 검출 실패 확인
# - 정리된 이미지에서 외곽 Contour를 검출합니다.
# - 검출한 전체 Contour 개수를 출력합니다.
# - Contour가 없으면 중간 결과를 저장하고 종료합니다.
# ============================================================

contours, _ = cv2.findContours(
    cleaned.copy(),
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE,
)

print("전체 Contour 수:", len(contours))


if not contours:

    cv2.imwrite(
        str(output_dir / "binary.jpg"),
        binary,
    )

    cv2.imwrite(
        str(output_dir / "cleaned.jpg"),
        cleaned,
    )

    print("Contour를 찾지 못했습니다.")

    raise SystemExit


# ============================================================
# ⑧ 가장 큰 Contour 선택 및 면적·Bounding Box 계산
# - 검출된 Contour 중 면적이 가장 큰 객체를 선택합니다.
# - Area : 선택한 Contour의 실제 면적
# - BBox : 객체를 둘러싸는 사각형의 위치와 크기
# - x, y : 사각형 왼쪽 위 좌표
# - w, h : 사각형 너비와 높이
# ============================================================

largest = max(
    contours,
    key=cv2.contourArea,
)

area = float(
    cv2.contourArea(largest)
)

x, y, w, h = cv2.boundingRect(
    largest
)


# ============================================================
# ⑨ Contour 중심점 계산
# - Moments를 이용하여 객체의 중심 좌표를 계산합니다.
# - m00이 0이 아니면 중심점(cx, cy)을 구합니다.
# - 계산할 수 없는 경우 center는 None으로 유지합니다.
# ============================================================

moments = cv2.moments(
    largest
)

center = None

if moments["m00"] != 0:

    cx = int(
        moments["m10"]
        / moments["m00"]
    )

    cy = int(
        moments["m01"]
        / moments["m00"]
    )

    center = (
        cx,
        cy,
    )


# ============================================================
# ⑩ Area Ratio 계산
# - ROI 전체 면적 = 이미지 너비 × 높이
# - Area Ratio = 가장 큰 Contour 면적 / ROI 전체 면적
# - 객체가 ROI에서 차지하는 면적 비율을 구합니다.
# ============================================================

height, width = roi.shape[:2]

roi_area = float(
    width * height
)

area_ratio = (
    area / roi_area
    if roi_area > 0
    else 0.0
)


# ============================================================
# ⑪ 검사 결과를 원본 ROI 이미지에 표시
# - 초록색 : 가장 큰 Contour 윤곽선
# - 파란색 : Bounding Box
# - 빨간색 : 객체 중심점
# - 노란색 : Area Ratio 측정값
# ============================================================

result = roi.copy()


# 가장 큰 Contour를 초록색으로 표시
cv2.drawContours(
    result,
    [largest],
    -1,
    (0, 255, 0),
    2,
)


# Bounding Box를 파란색 사각형으로 표시
cv2.rectangle(
    result,
    (x, y),
    (x + w, y + h),
    (255, 0, 0),
    2,
)


# Contour 중심점을 빨간색 점으로 표시
if center is not None:

    cv2.circle(
        result,
        center,
        5,
        (0, 0, 255),
        -1,
    )


# Area Ratio를 노란색 글자로 표시
cv2.putText(
    result,
    f"AreaRatio: {area_ratio:.3f}",
    (10, 25),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.65,
    (0, 255, 255),
    2,
)


# ============================================================
# ⑫ 검사 결과 이미지 저장
# - binary.jpg         : Otsu 이진화 결과
# - cleaned.jpg        : Morphology 처리 결과
# - contour_result.jpg : Contour·BBox·중심점·비율 표시
# ============================================================

cv2.imwrite(
    str(output_dir / "binary.jpg"),
    binary,
)

cv2.imwrite(
    str(output_dir / "cleaned.jpg"),
    cleaned,
)

cv2.imwrite(
    str(output_dir / "contour_result.jpg"),
    result,
)


# ============================================================
# ⑬ 측정 결과 출력
# - Largest Area : 가장 큰 Contour의 면적
# - Area Ratio   : ROI 전체 면적 대비 Contour 면적 비율
# - BBox         : 객체를 둘러싼 사각형 좌표와 크기
# - Center       : 객체의 중심 좌표
# ============================================================

print("Largest Area:", area)
print("Area Ratio:", area_ratio)
print("BBox:", (x, y, w, h))
print("Center:", center)
print("Contour 측정 완료")