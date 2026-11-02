from pathlib import Path

import cv2
import numpy as np


# --------------------------------------------------
# 1. 사용할 원본 이미지와 결과 저장 경로를 설정합니다.
#
# source_path
# → 광천김 전체 제품 이미지
#
# reference_dir
# → 잘라낸 기준 로고 ROI를 저장할 폴더
#
# target_dir
# → 위치·회전·크기를 변경한 비교 이미지를 저장할 폴더
# --------------------------------------------------

source_path = Path(
    "samples/day04/source/full_front.jpg"
)

reference_dir = Path(
    "samples/day04/reference"
)

target_dir = Path(
    "samples/day04/targets"
)


# --------------------------------------------------
# 2. 결과 이미지를 저장할 폴더를 생성합니다.
#
# parents=True
# → 상위 폴더가 없으면 함께 생성
#
# exist_ok=True
# → 폴더가 이미 있어도 오류 없이 진행
# --------------------------------------------------

reference_dir.mkdir(
    parents=True,
    exist_ok=True,
)

target_dir.mkdir(
    parents=True,
    exist_ok=True,
)


# --------------------------------------------------
# 3. 광천김 전체 제품 이미지를 읽고 확인합니다.
#
# cv2.imread()
# → 이미지 파일을 읽는 함수
#
# image가 None이면 파일을 읽지 못한 상태입니다.
# --------------------------------------------------

image = cv2.imread(
    str(source_path)
)

if image is None:
    raise FileNotFoundError(
        source_path
    )


# --------------------------------------------------
# 4. 원본 이미지의 가로와 세로 크기를 확인합니다.
#
# image.shape[:2]
# → 이미지 높이(height), 너비(width)
#
# 이 값은 ROI 좌표 확인과
# Target 이미지 생성에 사용합니다.
# --------------------------------------------------

height, width = image.shape[:2]


# --------------------------------------------------
# 5. 기준 로고로 사용할 ROI 좌표를 설정합니다.
#
# 이번에 자를 영역:
# KC 로고 + (주)광천김 + 영문
#
# x1, y1 → ROI의 왼쪽 위 좌표
# x2, y2 → ROI의 오른쪽 아래 좌표
#
# 아래 좌표는 예시입니다.
# 실제 사진의 로고 위치에 맞게 수정해야 합니다.
# --------------------------------------------------

x1 = 510
y1 = 870
x2 = 950
y2 = 1080


# --------------------------------------------------
# 6. ROI 좌표가 이미지 범위를 벗어나지 않도록 조정합니다.
#
# 이미지 바깥쪽 좌표를 지정하면
# 올바른 영역을 잘라내지 못할 수 있습니다.
#
# min() → 이미지 크기를 넘지 않도록 제한
# max() → 좌표가 음수가 되지 않도록 제한
# --------------------------------------------------

x1 = max(0, min(x1, width))
x2 = max(0, min(x2, width))

y1 = max(0, min(y1, height))
y2 = max(0, min(y2, height))


# --------------------------------------------------
# 7. ROI 좌표가 올바른지 검사합니다.
#
# 왼쪽 좌표가 오른쪽보다 크거나 같으면
# 정상적인 ROI를 만들 수 없습니다.
#
# 위쪽 좌표가 아래쪽보다 크거나 같아도
# 오류를 발생시킵니다.
# --------------------------------------------------

if x1 >= x2 or y1 >= y2:
    raise ValueError(
        f"잘못된 Template 좌표: "
        f"{(x1, y1, x2, y2)}"
    )


# --------------------------------------------------
# 8. 전체 이미지에서 로고 ROI를 잘라 Reference를 만듭니다.
#
# image[y1:y2, x1:x2]
# → 지정한 사각형 영역만 잘라냅니다.
#
# reference_label.jpg
# → 이후 Template / ORB / SIFT / Homography에서
#   공통으로 사용할 기준 로고 이미지입니다.
# --------------------------------------------------

reference = image[
    y1:y2,
    x1:x2
]

reference_path = (
    reference_dir
    / "reference_label.jpg"
)

cv2.imwrite(
    str(reference_path),
    reference,
)


# --------------------------------------------------
# 9. 원본 이미지에 ROI 위치를 표시한 Preview를 만듭니다.
#
# image.copy()
# → 원본 이미지 복사
#
# cv2.rectangle()
# → 로고 ROI 위치에 초록색 사각형 표시
#
# reference_preview.jpg
# → 실제로 올바른 로고를 잘랐는지 확인하는 이미지
# --------------------------------------------------

preview = image.copy()

cv2.rectangle(
    preview,
    (x1, y1),
    (x2, y2),
    (0, 255, 0),
    3,
)

cv2.imwrite(
    str(
        reference_dir
        / "reference_preview.jpg"
    ),
    preview,
)


# --------------------------------------------------
# 10. 정상 Target 이미지를 만듭니다.
#
# target_normal.jpg
# → 원본 이미지를 변경하지 않고 그대로 저장
#
# 이후 다른 Target들과 결과를 비교할 때
# 기준 상태로 사용합니다.
# --------------------------------------------------

cv2.imwrite(
    str(
        target_dir
        / "target_normal.jpg"
    ),
    image,
)


# --------------------------------------------------
# 11. 위치를 이동한 Target 이미지를 만듭니다.
#
# 오른쪽으로 35px 이동
# 아래쪽으로 20px 이동
#
# cv2.warpAffine()
# → 이동 변환 행렬을 이미지에 적용
#
# target_shift.jpg
# → 제품 위치가 바뀌었을 때
#   로고를 찾을 수 있는지 비교하는 이미지
# --------------------------------------------------

shift_x = 35
shift_y = 20

move_matrix = np.float32([
    [1, 0, shift_x],
    [0, 1, shift_y],
])

shifted = cv2.warpAffine(
    image,
    move_matrix,
    (width, height),
    borderMode=cv2.BORDER_REPLICATE,
)

cv2.imwrite(
    str(
        target_dir
        / "target_shift.jpg"
    ),
    shifted,
)


# --------------------------------------------------
# 12. 이미지를 15도 회전한 Target을 만듭니다.
#
# 이미지 중심을 기준으로 회전합니다.
#
# cv2.getRotationMatrix2D()
# → 회전 변환 행렬 생성
#
# cv2.warpAffine()
# → 회전 변환 적용
#
# target_rot15.jpg
# → 로고가 회전했을 때
#   각 검사 방법의 결과가 달라지는지 비교
# --------------------------------------------------

center = (
    width / 2,
    height / 2,
)

rotation_matrix = cv2.getRotationMatrix2D(
    center,
    15,
    1.0,
)

rotated = cv2.warpAffine(
    image,
    rotation_matrix,
    (width, height),
    borderMode=cv2.BORDER_REPLICATE,
)

cv2.imwrite(
    str(
        target_dir
        / "target_rot15.jpg"
    ),
    rotated,
)


# --------------------------------------------------
# 13. 이미지를 90% 크기로 축소한 Target을 만듭니다.
#
# scale=0.90
# → 원본 크기의 90%로 축소
#
# 이미지 중심을 기준으로 축소합니다.
#
# target_scale90.jpg
# → 로고 크기가 달라졌을 때
#   검사 결과가 어떻게 변하는지 비교
# --------------------------------------------------

scale_matrix = cv2.getRotationMatrix2D(
    center,
    0,
    0.90,
)

scaled = cv2.warpAffine(
    image,
    scale_matrix,
    (width, height),
    borderMode=cv2.BORDER_REPLICATE,
)

cv2.imwrite(
    str(
        target_dir
        / "target_scale90.jpg"
    ),
    scaled,
)


# --------------------------------------------------
# 14. 생성된 이미지 정보를 출력하고 결과를 확인합니다.
#
# 확인할 내용:
#
# 1. Reference 이미지가 올바르게 잘렸는가?
# 2. 정상 Target은 원본과 같은가?
# 3. 이동 Target은 위치가 달라졌는가?
# 4. 회전 Target은 15도 회전되었는가?
# 5. 축소 Target은 90% 크기가 되었는가?
#
# 이 이미지들은 뒤의 로고 비교 실습에서 사용합니다.
# --------------------------------------------------

print(
    "Source :",
    source_path
)

print(
    "Template:",
    reference_path
)

print(
    "Template 좌표:",
    (x1, y1, x2, y2)
)

print(
    "Template 크기:",
    reference.shape
)

print(
    "Day04 Target 생성 완료"
)