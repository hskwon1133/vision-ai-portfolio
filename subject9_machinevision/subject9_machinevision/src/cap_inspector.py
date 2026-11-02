##AI파일 : 실시간 병뚜껑 검사 시스템의 핵심 검사 로직(Business Logic)

"""1~3일차 OpenCV 병뚜껑 검사 로직을 재사용 가능한 함수로 구성합니다."""


# ============================================================
# ① 필요한 라이브러리 불러오기
# - dataclass : 검사 기준값을 하나로 묶어서 저장
# - Path      : 이미지 파일 경로 처리
# - cv2       : 이미지 전처리 및 Contour 검출
# - numpy     : 이미지 배열 처리
# ============================================================

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


# ============================================================
# ② 최소 객체 면적 기준 설정
# - ROI 전체 면적의 0.1%보다 작은 Contour는 제외합니다.
# - 너무 작은 노이즈가 검사에 포함되지 않도록 합니다.
# ============================================================

MIN_CONTOUR_AREA_RATIO = 0.001


# ============================================================
# ③ 병뚜껑 검사 기준값을 저장할 구조 만들기
# - cap_on_ratio  : 뚜껑이 있는 기준 이미지의 면적 비율
# - cap_off_ratio : 뚜껑이 없는 기준 이미지의 면적 비율
# - threshold     : 정상과 불량을 구분할 기준값
# - cap_present_when_greater :
#   뚜껑이 있을 때 면적 비율이 더 큰지 저장
#
# frozen=True : 생성 후 기준값을 변경하지 못하게 합니다.
# ============================================================

@dataclass(frozen=True)
class CapRule:
    cap_on_ratio: float
    cap_off_ratio: float
    threshold: float
    cap_present_when_greater: bool


# ============================================================
# ④ 입력 ROI 이미지가 정상인지 확인
# - ROI는 검사할 영역만 잘라 놓은 이미지입니다.
# - 비어 있거나 BGR 컬러 이미지가 아니면 오류 발생
# ============================================================

def measure_area_ratio(roi: np.ndarray) -> tuple[float, np.ndarray | None]:
    """ROI에서 가장 큰 Contour의 면적 비율을 계산합니다."""

    if roi is None or roi.size == 0 or roi.ndim != 3 or roi.shape[2] != 3:
        raise ValueError("ROI는 비어 있지 않은 BGR 이미지여야 합니다.")


    # ========================================================
    # ⑤ Gray → Otsu 이진화
    # - BGR 컬러 이미지를 Gray로 변환합니다.
    # - Otsu를 이용하여 자동으로 이진화 기준값을 정합니다.
    # - BINARY_INV를 적용하여 흑백을 반전합니다.
    # ========================================================

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    _, binary = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )


    # ========================================================
    # ⑥ Opening → Closing으로 이진 이미지 정리
    # - Opening : 작은 노이즈 제거
    # - Closing : 객체 내부의 작은 틈이나 구멍 메우기
    # - 3×3 타원형 Kernel 사용
    # ========================================================

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    opened = cv2.morphologyEx(
        binary, cv2.MORPH_OPEN, kernel, iterations=1
    )

    cleaned = cv2.morphologyEx(
        opened, cv2.MORPH_CLOSE, kernel, iterations=1
    )


    # ========================================================
    # ⑦ Contour 검출 및 작은 객체 제외
    # - 정리된 이진 이미지에서 외곽 윤곽선을 찾습니다.
    # - ROI 전체 면적을 계산합니다.
    # - ROI 면적의 0.1% 미만인 Contour는 제외합니다.
    # ========================================================

    contours, _ = cv2.findContours(
        cleaned.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    height, width = roi.shape[:2]
    roi_area = float(height * width)

    min_area = roi_area * MIN_CONTOUR_AREA_RATIO

    valid = [
        c for c in contours
        if cv2.contourArea(c) >= min_area
    ]


    # ========================================================
    # ⑧ 가장 큰 객체의 Area Ratio 계산
    # - 유효한 Contour가 없으면 0.0과 None을 반환합니다.
    # - 가장 큰 Contour를 선택합니다.
    # - Contour 면적을 ROI 전체 면적으로 나눕니다.
    # ========================================================

    if not valid:
        return 0.0, None

    largest = max(valid, key=cv2.contourArea)

    return float(cv2.contourArea(largest)) / roi_area, largest


# ============================================================
# ⑨ 기준 이미지 2장 불러오기
# - cap_on.jpg  : 병뚜껑이 있는 이미지
# - cap_off.jpg : 병뚜껑이 없는 이미지
# - 파일이 없거나 두 이미지의 크기가 다르면 오류 발생
# ============================================================

def load_rule(reference_dir: Path) -> CapRule:
    """기준 이미지 2장에서 병뚜껑 판정 기준값을 만듭니다."""

    on_path = reference_dir / "cap_on.jpg"
    off_path = reference_dir / "cap_off.jpg"

    cap_on = cv2.imread(str(on_path))
    cap_off = cv2.imread(str(off_path))

    if cap_on is None or cap_off is None:
        raise FileNotFoundError(
            "기준 파일을 확인하세요: "
            f"{on_path} / {off_path}"
        )

    if cap_on.shape[:2] != cap_off.shape[:2]:
        raise ValueError(
            "cap_on.jpg와 cap_off.jpg의 ROI 크기가 서로 다릅니다."
        )


    # ========================================================
    # ⑩ 두 기준 이미지의 Area Ratio 측정
    # - 앞에서 만든 measure_area_ratio() 함수를 재사용합니다.
    # - 두 이미지의 면적 비율을 각각 계산합니다.
    # - 차이가 너무 작으면 구분하기 어려우므로 오류 발생
    # ========================================================

    on_ratio, _ = measure_area_ratio(cap_on)
    off_ratio, _ = measure_area_ratio(cap_off)

    if abs(on_ratio - off_ratio) < 0.01:
        raise ValueError(
            "기준 이미지의 Area Ratio 차이가 너무 작습니다. "
            "촬영·ROI·조명 조건을 다시 확인하세요."
        )


    # ========================================================
    # ⑪ 병뚜껑 판정 기준값 생성 및 저장
    # - 두 기준 이미지의 Area Ratio 중간값을 계산합니다.
    # - 뚜껑이 있을 때 비율이 더 큰지도 확인합니다.
    # - 계산한 값을 CapRule에 저장하여 반환합니다.
    # ========================================================

    return CapRule(
        cap_on_ratio=on_ratio,
        cap_off_ratio=off_ratio,
        threshold=(on_ratio + off_ratio) / 2.0,
        cap_present_when_greater=(on_ratio > off_ratio),
    )


# ============================================================
# ⑫ 새로운 이미지의 Area Ratio를 기준값과 비교
# - 뚜껑이 있을 때 비율이 큰 경우와 작은 경우를 구분합니다.
# - 기준값과 비교하여 OK 또는 NG를 반환합니다.
# ============================================================

def judge_cap(area_ratio: float, rule: CapRule) -> tuple[str, str]:
    """측정된 면적 비율을 기준값과 비교하여 판정합니다."""

    if rule.cap_present_when_greater:
        cap_present = area_ratio >= rule.threshold
    else:
        cap_present = area_ratio <= rule.threshold

    return ("OK", "") if cap_present else ("NG", "NG_CAP_AREA")


# ============================================================
# ⑬ 새로운 ROI의 최종 검사 결과 반환
# - 새로운 ROI 이미지가 들어오면 Area Ratio를 측정합니다.
# - 유효한 Contour가 없으면 CHECK(확인 필요)를 반환합니다.
# - Contour가 있으면 ⑫의 judge_cap()으로 판정합니다.
# - 판정 결과, 원인, 면적 비율을 딕셔너리로 반환합니다.
# ============================================================

def inspect_roi(roi: np.ndarray, rule: CapRule) -> dict:
    """새로운 ROI를 검사하고 최종 판정 결과를 반환합니다."""

    ratio, contour = measure_area_ratio(roi)

    # 유효한 Contour가 없으면 정상/불량을 단정하지 않습니다.
    if contour is None:
        return {
            "decision": "CHECK",
            "reason": "ROI_CONTOUR_NOT_FOUND",
            "ratio": ratio
        }

    # 기준값과 비교하여 OK 또는 NG를 결정합니다.
    decision, reason = judge_cap(ratio, rule)

    return {
        "decision": decision,
        "reason": reason,
        "ratio": ratio
    }