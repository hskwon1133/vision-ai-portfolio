
# ============================================================
# ① 필요한 라이브러리 불러오기
# - Path  : 기준 이미지 및 결과 저장 경로 처리
# - cv2   : 영상 전처리, Contour 검출 및 판정 결과 표시
# - numpy : 이미지 배열 처리를 위한 라이브러리
# ============================================================

from pathlib import Path

import cv2
import numpy as np


# ============================================================
# ② 기준 이미지와 결과 저장 경로 설정
# - cap_on.jpg  : 병뚜껑이 있는 기준 이미지
# - cap_off.jpg : 병뚜껑이 없는 기준 이미지
# - OUTPUT_DIR  : 검사 결과 이미지 저장 폴더
# - MIN_CONTOUR_AREA_RATIO : 작은 객체 제외 기준(0.1%)
# ============================================================

CAP_ON_PATH = Path(
    "samples/day03/reference/cap_on.jpg"
)

CAP_OFF_PATH = Path(
    "samples/day03/reference/cap_off.jpg"
)

OUTPUT_DIR = Path(
    "outputs/day03/rule"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

MIN_CONTOUR_AREA_RATIO = 0.001


# ============================================================
# ③ 이미지 불러오기 및 오류 확인
# - 입력받은 경로에서 이미지를 읽습니다.
# - 이미지가 없거나 읽을 수 없으면 오류를 발생시킵니다.
# ============================================================

def measure_area_ratio(
    image_path,
):

    image = cv2.imread(
        str(image_path)
    )

    if image is None:
        raise FileNotFoundError(
            f"이미지를 읽지 못했습니다: "
            f"{image_path}"
        )


    # ========================================================
    # ④ Gray → Otsu 이진화
    # - BGR 컬러 이미지를 Gray로 변환합니다.
    # - Otsu로 이진화 밝기 기준값을 자동 계산합니다.
    # - BINARY_INV로 어두운 병뚜껑을 흰색 객체로 변환합니다.
    # ========================================================

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    _, binary = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV
        + cv2.THRESH_OTSU,
    )


    # ========================================================
    # ⑤ Morphology로 이미지 정리
    # - 3×3 타원형 Kernel을 사용합니다.
    # - Opening : 작은 흰색 노이즈 제거
    # - Closing : 작은 검은색 틈이나 구멍 정리
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
    # ⑥ Contour 검출 및 Area Ratio 측정
    # - 정리된 이미지에서 외곽 Contour를 찾습니다.
    # - ROI 전체 면적의 0.1%보다 작은 객체는 제외합니다.
    # - 가장 큰 Contour의 면적을 계산합니다.
    # - Area Ratio = Contour 면적 / ROI 전체 면적
    # - Contour가 없으면 0.0과 None을 반환합니다.
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

    valid = [
        contour
        for contour in contours
        if cv2.contourArea(contour)
        >= min_area
    ]

    if not valid:

        return (
            0.0,
            image,
            None,
        )

    largest = max(
        valid,
        key=cv2.contourArea,
    )

    largest_area = float(
        cv2.contourArea(largest)
    )

    area_ratio = (
        largest_area / image_area
        if image_area > 0
        else 0.0
    )

    return (
        area_ratio,
        image,
        largest,
    )


# ============================================================
# ⑦ 정상·불량 기준 이미지의 Area Ratio 측정
# - 앞에서 만든 measure_area_ratio() 함수를 호출합니다.
# - cap_on과 cap_off의 면적 비율을 각각 계산합니다.
# - 측정값은 다음 단계의 판정 기준값 생성에 사용합니다.
# ============================================================

cap_on_ratio, _, _ = (
    measure_area_ratio(
        CAP_ON_PATH
    )
)

cap_off_ratio, _, _ = (
    measure_area_ratio(
        CAP_OFF_PATH
    )
)


# ============================================================
# ⑧ OK / NG 판정 기준값 생성
# - 두 기준 이미지의 Area Ratio 중간값을 계산합니다.
# - 이 값을 candidate_threshold로 저장합니다.
# - 뚜껑이 있을 때 면적 비율이 더 큰지도 확인합니다.
# ============================================================

candidate_threshold = (
    cap_on_ratio
    + cap_off_ratio
) / 2.0


cap_present_when_greater = (
    cap_on_ratio
    > cap_off_ratio
)


# ============================================================
# ⑨ 병뚜껑 유무 판정 함수 작성
# - 새로 측정한 Area Ratio를 기준값과 비교합니다.
# - 뚜껑이 있을 때 비율이 큰 경우와 작은 경우를 구분합니다.
# - 병뚜껑이 있다고 판정하면 OK를 반환합니다.
# - 병뚜껑이 없다고 판정하면 NG와 사유를 반환합니다.
# ============================================================

def judge_cap(
    area_ratio,
):

    if cap_present_when_greater:

        cap_present = (
            area_ratio
            >= candidate_threshold
        )

    else:

        cap_present = (
            area_ratio
            <= candidate_threshold
        )

    if cap_present:

        return (
            "OK",
            "",
        )

    return (
        "NG",
        "NG_CAP_AREA",
    )


# ============================================================
# ⑩ 기준값 측정 결과 출력
# - cap_on ratio      : 병뚜껑이 있는 이미지의 면적 비율
# - cap_off ratio     : 병뚜껑이 없는 이미지의 면적 비율
# - reference gap     : 두 기준 측정값의 차이
# - candidate threshold : OK / NG 구분을 위한 후보 기준값
# - GREATER / LESS    : 뚜껑이 있을 때의 비율 방향
# ============================================================

print(
    "cap_on ratio :",
    cap_on_ratio,
)

print(
    "cap_off ratio:",
    cap_off_ratio,
)

print(
    "reference gap:",
    abs(
        cap_on_ratio
        - cap_off_ratio
    ),
)

print(
    "candidate threshold:",
    candidate_threshold,
)

print(
    "cap present when:",
    "GREATER"
    if cap_present_when_greater
    else "LESS",
)


# ============================================================
# ⑪ 정상·불량 기준 이미지에 판정 규칙 적용
# - cap_on과 cap_off 이미지를 차례대로 검사합니다.
# - measure_area_ratio()로 면적 비율을 다시 측정합니다.
# - judge_cap()으로 OK / NG를 판정합니다.
# ============================================================

for label, image_path in [
    (
        "cap_on",
        CAP_ON_PATH,
    ),
    (
        "cap_off",
        CAP_OFF_PATH,
    ),
]:

    area_ratio, image, contour = (
        measure_area_ratio(
            image_path
        )
    )

    decision, reason = judge_cap(
        area_ratio
    )


    # ========================================================
    # ⑫ 판정 결과를 이미지에 표시하고 저장
    # - 가장 큰 Contour를 초록색으로 표시합니다.
    # - OK는 초록색, NG는 빨간색 글자로 표시합니다.
    # - Area Ratio도 이미지에 함께 표시합니다.
    # - cap_on_rule.jpg / cap_off_rule.jpg로 저장합니다.
    # ========================================================

    result = image.copy()

    if contour is not None:

        cv2.drawContours(
            result,
            [contour],
            -1,
            (0, 255, 0),
            2,
        )

    text = (
        f"{decision} "
        f"ratio={area_ratio:.3f}"
    )

    cv2.putText(
        result,
        text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (
            0,
            200,
            0,
        )
        if decision == "OK"
        else (
            0,
            0,
            255,
        ),
        2,
    )

    output_path = (
        OUTPUT_DIR
        / f"{label}_rule.jpg"
    )

    cv2.imwrite(
        str(output_path),
        result,
    )


    # ========================================================
    # ⑬ 최종 판정 결과 출력
    # - file       : 검사한 이미지 파일명
    # - area_ratio : 측정한 객체 면적 비율
    # - decision   : OK 또는 NG
    # - reason     : NG 판정 사유
    # ========================================================

    print()
    print("file:", image_path.name)
    print("area_ratio:", area_ratio)
    print("decision:", decision)
    print("reason:", reason)