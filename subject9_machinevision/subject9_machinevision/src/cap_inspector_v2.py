#6단계: 검사 함수 개선판 파일 만들기
"""병뚜껑 검사 개선판 (v2).

교재 cap_inspector.py와 같은 함수 이름을 유지해서 FastAPI에서 import만 바꾸면 되게 만들었다.
바뀐 점
  1. 판정 특징: 가장 큰 덩어리 면적 비율 -> 위쪽 절반의 '꽉 찬 줄 비율'
     (병 거리·크기가 바뀌어도 덜 흔들림, 근거: outputs/okng/fill_sweep.csv)
  2. 기준선 근처는 CHECK(확인 필요)로 분류
  3. ROI가 비정상(거의 다 흰색/검은색)이면 CHECK
  4. 비교용으로 교재 방식(면적 비율)도 같이 계산해서 반환
"""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

# ===== 고정 파라미터 (Rule Freeze: 바꿀 때는 근거와 날짜를 기록) =====
KERNEL_SIZE = 3          # Opening/Closing 커널 (교재 결론: 3x3, 각 1회)
FILL = 0.5               # 줄의 흰색이 이 비율을 넘으면 '꽉 찬 줄' (5단계 스윕으로 선택)
REVIEW_BAND = 0.10       # 기준선 ± (ON-OFF 간격의 10%)는 CHECK
MIN_WHITE, MAX_WHITE = 0.01, 0.99   # ROI 유효성: 흰색 비율이 이 범위 밖이면 CHECK


@dataclass(frozen=True)
class CapRule:
    cap_on_ratio: float      # 기준 ON의 꽉 찬 줄 비율
    cap_off_ratio: float     # 기준 OFF의 꽉 찬 줄 비율
    threshold: float         # 판정선 (두 값의 중간)
    review_low: float        # 이 사이는 CHECK
    review_high: float
    cap_present_when_greater: bool


def binarize(roi: np.ndarray) -> np.ndarray:
    """BGR ROI -> Gray -> Otsu(INV) -> Opening -> Closing"""
    if roi is None or roi.size == 0 or roi.ndim != 3 or roi.shape[2] != 3:
        raise ValueError("ROI는 비어 있지 않은 BGR 이미지여야 합니다.")
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, b = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (KERNEL_SIZE, KERNEL_SIZE))
    b = cv2.morphologyEx(b, cv2.MORPH_OPEN, k)
    return cv2.morphologyEx(b, cv2.MORPH_CLOSE, k)


def solid_row_ratio(binary: np.ndarray, fill: float = FILL) -> float:
    """위쪽 절반에서 흰색이 fill을 넘는 가로줄의 비율"""
    top = binary[: binary.shape[0] // 2]
    row_fill = (top > 0).mean(axis=1)
    return float((row_fill > fill).mean())


def area_ratio(binary: np.ndarray) -> float:
    """(비교용) 교재 방식: 가장 큰 Contour 면적 / ROI 면적"""
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0
    return max(cv2.contourArea(c) for c in contours) / binary.size


def load_rule(reference_dir: Path) -> CapRule:
    """기준 이미지 2장(cap_on.jpg, cap_off.jpg)으로 판정선을 만든다."""
    on = cv2.imread(str(reference_dir / "cap_on.jpg"))
    off = cv2.imread(str(reference_dir / "cap_off.jpg"))
    if on is None or off is None:
        raise FileNotFoundError(f"기준 파일을 확인하세요: {reference_dir}")
    on_s = solid_row_ratio(binarize(on))
    off_s = solid_row_ratio(binarize(off))
    gap = abs(on_s - off_s)
    if gap < 0.05:
        raise ValueError("기준 이미지의 차이가 너무 작습니다. 촬영·ROI를 확인하세요.")
    th = (on_s + off_s) / 2
    return CapRule(on_s, off_s, th,
                   th - gap * REVIEW_BAND, th + gap * REVIEW_BAND,
                   on_s > off_s)


def inspect_roi(roi: np.ndarray, rule: CapRule) -> dict:
    """ROI 한 장을 검사해서 OK / NG / CHECK와 근거 숫자를 돌려준다."""
    b = binarize(roi)
    white = float((b > 0).mean())
    score = solid_row_ratio(b)
    result = {"ratio": score, "area_ratio": area_ratio(b), "white_ratio": white}

    if not (MIN_WHITE <= white <= MAX_WHITE):
        return {**result, "decision": "CHECK", "reason": "ROI_INVALID"}
    if rule.review_low <= score <= rule.review_high:
        return {**result, "decision": "CHECK", "reason": "NEAR_THRESHOLD"}

    present = score > rule.threshold if rule.cap_present_when_greater else score < rule.threshold
    return {**result, "decision": "OK" if present else "NG",
            "reason": "" if present else "NG_CAP_SHAPE"}
