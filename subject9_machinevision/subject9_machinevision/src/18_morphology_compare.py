
# ============================================================
# ① 필요한 라이브러리 불러오기
# - Path  : 이미지 파일 및 저장 폴더 경로 처리
# - cv2   : 이미지 전처리와 Morphology 수행
# - numpy : 이미지 배열 처리
# ============================================================

from pathlib import Path

import cv2
import numpy as np


# ============================================================
# ② 입력 이미지 및 결과 저장 경로 설정
# - cap_on.jpg : 병뚜껑이 있는 기준 이미지
# - output_dir : 전처리 결과 이미지 저장 폴더
# - 폴더가 없으면 자동으로 생성합니다.
# ============================================================

image_path = Path(
    "samples/day03/reference/cap_on.jpg"
)

output_dir = Path(
    "outputs/day03/morphology"
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
# - 입력 이미지는 이미 ROI로 잘린 이미지입니다.
# - 따라서 추가로 Crop하지 않습니다.
# - BGR 컬러 이미지를 Gray 이미지로 변환합니다.
# ============================================================

roi = image

gray = cv2.cvtColor(
    roi,
    cv2.COLOR_BGR2GRAY,
)


# ============================================================
# ⑤ Otsu 이진화 및 Gray·Binary 이미지 저장
# - Otsu가 밝기 기준값을 자동으로 계산합니다.
# - BINARY_INV : 어두운 병뚜껑을 흰색 객체로 변환
# - Gray와 Binary 결과를 각각 JPG로 저장합니다.
# ============================================================

otsu_value, binary = cv2.threshold(
    gray,
    0,
    255,
    cv2.THRESH_BINARY_INV
    + cv2.THRESH_OTSU,
)

cv2.imwrite(
    str(output_dir / "01_gray.jpg"),
    gray,
)

cv2.imwrite(
    str(output_dir / "02_binary.jpg"),
    binary,
)


# ============================================================
# ⑥ 비교할 Kernel 크기 설정
# - 3×3, 5×5, 9×9 Kernel을 비교합니다.
# - Kernel 크기에 따라 노이즈 제거와 객체 형태가 달라집니다.
# ============================================================

kernel_sizes = [
    3,
    5,
    9,
]


# ============================================================
# ⑦ Kernel 크기별 Opening → Closing 반복 실행
# - 반복문으로 3×3, 5×5, 9×9 Kernel을 차례대로 사용합니다.
# - Opening : 작은 흰색 노이즈 제거
# - Closing : 작은 검은색 틈이나 구멍 메우기
# - 각 Kernel의 결과 이미지를 2장씩 저장합니다.
# ============================================================

for size in kernel_sizes:

    # 현재 크기의 타원형 Kernel 생성
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (size, size),
    )

    # Opening 적용
    opened = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        kernel,
        iterations=1,
    )

    # Opening 결과에 Closing 적용
    cleaned = cv2.morphologyEx(
        opened,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=1,
    )

    # Opening 결과 저장
    cv2.imwrite(
        str(
            output_dir
            / f"kernel_{size}_opening.jpg"
        ),
        opened,
    )

    # Closing 결과 저장
    cv2.imwrite(
        str(
            output_dir
            / f"kernel_{size}_closing.jpg"
        ),
        cleaned,
    )


# ============================================================
# ⑧ 실행 결과 출력
# - 사용한 입력 이미지 경로를 확인합니다.
# - Otsu가 계산한 이진화 기준값을 확인합니다.
# - Morphology 비교 이미지 저장 완료 메시지를 출력합니다.
# ============================================================

print("입력:", image_path)
print("Otsu Threshold:", otsu_value)
print("Morphology 비교 결과 저장 완료")