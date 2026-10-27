
# ============================================================
# ① 필요한 라이브러리 불러오기
# - Path  : 이미지 파일과 저장 폴더 경로 처리
# - csv   : 측정된 Feature를 CSV 파일로 저장
# - cv2   : 영상 전처리와 Contour 검출 및 측정
# - numpy : 이미지 배열 처리를 위한 라이브러리
# ============================================================

from pathlib import Path
import csv

import cv2
import numpy as np


# ============================================================
# ② 검사할 기준 이미지 2장의 경로 설정
# - cap_on  : 병뚜껑이 있는 정상 이미지
# - cap_off : 병뚜껑이 없는 불량 이미지
# - 두 이미지에 동일한 검사 함수를 적용합니다.
# ============================================================

input_files = {
    "cap_on": Path(
        "samples/day03/reference/cap_on.jpg"
    ),
    "cap_off": Path(
        "samples/day03/reference/cap_off.jpg"
    ),
}


# ============================================================
# ③ 결과 저장 경로 및 Noise 제거 기준 설정
# - output_dir : 처리된 이미지 저장 폴더
# - report_path: 측정 Feature를 저장할 CSV 경로
# - MIN_CONTOUR_AREA_RATIO:
#   전체 이미지 면적의 0.1% 미만인 Contour 제외
# ============================================================

output_dir = Path(
    "outputs/day03/feature"
)

output_dir.mkdir(
    parents=True,
    exist_ok=True,
)

report_path = Path(
    "reports/day03_reference_features.csv"
)

MIN_CONTOUR_AREA_RATIO = 0.001
#왜? 최소 Contour 면적 비율을 0.001로 설정했는지에 대한 이유는, 이미지에서 작은 노이즈나 불필요한 객체를 제거하기 위해서입니다. 전체 이미지 면적의 0.1% 미만인 Contour는 실제 관심 있는 객체가 아닐 가능성이 높기 때문에, 이러한 작은 Contour를 제외함으로써 분석의 정확성을 높일 수 있습니다.

# ============================================================
# ④ Gray → Otsu 이진화
# - measure_features()는 이미지 한 장을 입력받아 검사합니다.
# - 컬러 이미지를 Gray로 변환합니다.
# - Otsu로 이진화 기준값을 자동 계산합니다.
# - 어두운 병뚜껑 영역을 흰색 객체로 변환합니다.
# ============================================================

def measure_features(image):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    otsu_value, binary = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV
        + cv2.THRESH_OTSU,
    )


    # ========================================================
    # ⑤ Morphology로 이진 이미지 정리
    # - 3×3 타원형 Kernel을 사용합니다.
    # - Opening : 작은 흰색 노이즈 제거
    # - Closing : 작은 검은색 틈이나 구멍 메우기
    # ========================================================

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


    # ========================================================
    # ⑥ Contour 검출 및 작은 객체 제외
    # - 정리된 이미지에서 외곽 Contour를 찾습니다.
    # - 이미지 전체 면적을 계산합니다.
    # - 너무 작은 Contour는 Noise로 보고 제외합니다.
    # ========================================================

    contours, _ = cv2.findContours(
        cleaned.copy(),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    height, width = image.shape[:2]

    image_area = float(
        width * height
    )

    min_area = (
        image_area
        * MIN_CONTOUR_AREA_RATIO
    )

    valid_contours = [
        contour
        for contour in contours
        if cv2.contourArea(contour)
        >= min_area
    ]


    # ========================================================
    # ⑦ 측정 Feature 저장 구조 준비
    # - Otsu 기준값과 유효 Contour 수를 저장합니다.
    # - 면적, 중심점, 경계 접촉 여부를 저장할 공간을 만듭니다.
    # - Contour가 없다면 기본값을 반환합니다.
    # ========================================================

    features = {
        "otsu_value": float(otsu_value),
        "valid_count": len(valid_contours),
        "largest_area": 0.0,
        "largest_area_ratio": 0.0,
        "center_x_norm": "",
        "center_y_norm": "",
        "touches_border": "",
    }

    annotated = image.copy()

    if not valid_contours:

        return (
            features,
            binary,
            cleaned,
            annotated,
        )


    # ========================================================
    # ⑧ 가장 큰 Contour 선택 및 면적·중심점 측정
    # - 가장 면적이 큰 Contour를 선택합니다.
    # - Contour Area를 계산합니다.
    # - Bounding Box의 위치와 크기를 구합니다.
    # - Moments를 이용하여 객체 중심점을 계산합니다.
    # ========================================================

    largest = max(
        valid_contours,
        key=cv2.contourArea,
    )

    area = float(
        cv2.contourArea(largest)
    )

    x, y, w, h = cv2.boundingRect(
        largest
    )

    moments = cv2.moments(
        largest
    )

    cx = None
    cy = None

    if moments["m00"] != 0:

        cx = (
            moments["m10"]
            / moments["m00"]
        )

        cy = (
            moments["m01"]
            / moments["m00"]
        )


    # ========================================================
    # ⑨ 객체의 주요 Feature 계산 및 저장
    # - touches_border:
    #   Bounding Box가 이미지 경계에 닿는지 확인
    # - largest_area_ratio:
    #   가장 큰 객체 면적 ÷ 이미지 전체 면적
    # - center_x_norm / center_y_norm:
    #   중심점 좌표를 0~1 범위로 정규화
    # ========================================================

    touches_border = (
        x <= 0
        or y <= 0
        or x + w >= width
        or y + h >= height
    )

    features[
        "largest_area"
    ] = area

    features[
        "largest_area_ratio"
    ] = (
        area / image_area
        if image_area > 0
        else 0.0
    )

    if cx is not None:
        features[
            "center_x_norm"
        ] = cx / width

    if cy is not None:
        features[
            "center_y_norm"
        ] = cy / height

    features[
        "touches_border"
    ] = touches_border


    # ========================================================
    # ⑩ 검사 결과를 이미지에 표시하고 반환
    # - 초록색 : 가장 큰 Contour
    # - 파란색 : Bounding Box
    # - 빨간색 : Contour 중심점
    # - Feature와 중간 결과 이미지를 함께 반환합니다.
    # ========================================================

    cv2.drawContours(
        annotated,
        [largest],
        -1,
        (0, 255, 0),
        2,
    )

    cv2.rectangle(
        annotated,
        (x, y),
        (x + w, y + h),
        (255, 0, 0),
        2,
    )

    if (
        cx is not None
        and cy is not None
    ):

        cv2.circle(
            annotated,
            (
                int(cx),
                int(cy),
            ),
            5,
            (0, 0, 255),
            -1,
        )

    return (
        features,
        binary,
        cleaned,
        annotated,
    )


# ============================================================
# ⑪ 기준 이미지 2장을 반복해서 검사
# - rows : CSV에 저장할 측정 결과 목록
# - cap_on과 cap_off 이미지를 차례대로 불러옵니다.
# - measure_features() 함수를 호출하여 특징을 측정합니다.
# ============================================================

rows = []


for label, image_path in input_files.items():

    image = cv2.imread(
        str(image_path)
    )

    if image is None:
        raise FileNotFoundError(
            f"이미지를 읽지 못했습니다: "
            f"{image_path}"
        )

    (
        features,
        binary,
        cleaned,
        annotated,
    ) = measure_features(
        image
    )


    # ========================================================
    # ⑫ 이미지별 Feature 정리 및 결과 저장
    # - label과 파일명을 측정 Feature에 추가합니다.
    # - CSV 저장을 위해 rows 목록에 추가합니다.
    # - 각 이미지의 Binary, Cleaned, Contour 결과를 저장합니다.
    # - 측정한 Feature 값을 터미널에 출력합니다.
    # ========================================================

    row = {
        "label": label,
        "file": image_path.name,
        **features,
    }

    rows.append(
        row
    )

    cv2.imwrite(
        str(
            output_dir
            / f"{label}_binary.jpg"
        ),
        binary,
    )

    cv2.imwrite(
        str(
            output_dir
            / f"{label}_cleaned.jpg"
        ),
        cleaned,
    )

    cv2.imwrite(
        str(
            output_dir
            / f"{label}_contour.jpg"
        ),
        annotated,
    )

    print()
    print("Label:", label)

    for key, value in features.items():
        print(
            f"{key}:",
            value,
        )


# ============================================================
# ⑬ 측정 Feature를 CSV 보고서로 저장
# - reports 폴더가 없으면 자동으로 생성합니다.
# - CSV의 열 이름(fieldnames)을 지정합니다.
# - cap_on과 cap_off의 측정값을 각각 한 행으로 저장합니다.
# - utf-8-sig : Excel에서 한글을 읽기 쉽게 저장
# ============================================================

report_path.parent.mkdir(
    parents=True,
    exist_ok=True,
)

with report_path.open(
    "w",
    newline="",
    encoding="utf-8-sig",
) as file:

    fieldnames = [
        "label",
        "file",
        "otsu_value",
        "valid_count",
        "largest_area",
        "largest_area_ratio",
        "center_x_norm",
        "center_y_norm",
        "touches_border",
    ]

    writer = csv.DictWriter(
        file,
        fieldnames=fieldnames,
    )

    writer.writeheader()
    writer.writerows(
        rows
    )


# ============================================================
# ⑭ 최종 CSV 저장 결과 출력
# - 측정 결과를 저장한 CSV 파일 경로를 출력합니다.
# ============================================================

print()
print(
    "측정 결과 CSV:",
    report_path,
)