
# ============================================================
# ① 필요한 라이브러리 불러오기
# - Path : 이미지 및 결과 저장 경로 처리
# - csv  : 검사 결과를 CSV 파일로 저장
# - cv2  : 이미지 전처리, Contour 검출 및 판정 결과 표시
# ============================================================

from pathlib import Path
import csv

import cv2


# ============================================================
# ② 이미지 경로, 결과 저장 폴더 및 Noise 제거 기준 설정
# - cap_on / cap_off 폴더 지정
# - Reference 이미지 2장 지정
# - 검사 결과 이미지 및 CSV 저장 경로 지정
# - ROI 면적의 0.1%보다 작은 Contour는 제외
# ============================================================

CAP_ON_DIR = Path(
    "samples/day03/cap_on"
)

CAP_OFF_DIR = Path(
    "samples/day03/cap_off"
)

REFERENCE_ON = (
    CAP_ON_DIR / "cap_on_001.jpg"
)

REFERENCE_OFF = (
    CAP_OFF_DIR / "cap_off_001.jpg"
)

OUTPUT_DIR = Path(
    "outputs/day03/batch"
)

REPORT_PATH = Path(
    "reports/day03_inspection_results.csv"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

REPORT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

MIN_CONTOUR_AREA_RATIO = 0.001


# ============================================================
# ③ 검사할 이미지 불러오기
# - measure()는 이미지 한 장의 특징을 측정하는 함수입니다.
# - 이미지를 읽지 못하면 None을 반환합니다.
# ============================================================

def measure(image_path):

    image = cv2.imread(
        str(image_path)
    )

    if image is None:
        return None


    # ========================================================
    # ④ Gray → Otsu 이진화
    # - BGR 이미지를 Gray로 변환합니다.
    # - Otsu로 밝기 기준값을 자동 계산합니다.
    # - BINARY_INV로 어두운 객체를 흰색으로 변환합니다.
    # ========================================================

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
    # ⑥ Contour 검출 및 작은 객체 제외
    # - 정리된 이미지에서 외곽 Contour를 검출합니다.
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
    # ⑦ 가장 큰 Contour의 Feature 측정 및 반환
    # - 가장 큰 Contour를 선택합니다.
    # - Area Ratio = Contour 면적 / ROI 전체 면적
    # - Moments를 이용하여 객체 중심점을 계산합니다.
    # - 중심점은 0~1 비율 좌표로 저장합니다.
    # - 측정 결과를 딕셔너리로 반환합니다.
    # ========================================================

    largest_contour = None
    largest_area_ratio = 0.0

    center_x_norm = ""
    center_y_norm = ""

    if valid_contours:

        largest_contour = max(
            valid_contours,
            key=cv2.contourArea,
        )

        largest_area = float(
            cv2.contourArea(
                largest_contour
            )
        )

        largest_area_ratio = (
            largest_area / image_area
            if image_area > 0
            else 0.0
        )

        moments = cv2.moments(
            largest_contour
        )

        if moments["m00"] != 0:

            cx = (
                moments["m10"]
                / moments["m00"]
            )

            cy = (
                moments["m01"]
                / moments["m00"]
            )

            center_x_norm = (
                cx / width
            )

            center_y_norm = (
                cy / height
            )

    return {
        "image": image,
        "otsu_value": float(
            otsu_value
        ),
        "valid_count": len(
            valid_contours
        ),
        "largest_contour": (
            largest_contour
        ),
        "largest_area_ratio": (
            largest_area_ratio
        ),
        "center_x_norm": (
            center_x_norm
        ),
        "center_y_norm": (
            center_y_norm
        ),
    }


# ============================================================
# ⑧ Reference 이미지 2장으로 판정 기준값 생성
# - cap_on_001.jpg와 cap_off_001.jpg를 측정합니다.
# - 이미지가 없으면 오류를 발생시킵니다.
# - 두 Area Ratio의 중간값을 판정 기준값으로 사용합니다.
# - cap_on일 때 Area Ratio가 더 큰지도 확인합니다.
# ============================================================

reference_on = measure(
    REFERENCE_ON
)

reference_off = measure(
    REFERENCE_OFF
)

if (
    reference_on is None
    or reference_off is None
):

    raise RuntimeError(
        "Reference 이미지를 확인하세요."
    )

on_ratio = (
    reference_on[
        "largest_area_ratio"
    ]
)

off_ratio = (
    reference_off[
        "largest_area_ratio"
    ]
)

candidate_threshold = (
    on_ratio + off_ratio
) / 2.0

cap_present_when_greater = (
    on_ratio > off_ratio
)

print(
    "Reference ON :",
    round(on_ratio, 4),
)

print(
    "Reference OFF:",
    round(off_ratio, 4),
)

print(
    "Threshold    :",
    round(candidate_threshold, 4),
)


# ============================================================
# ⑨ OK / NG 판정 함수 작성
# - 새로운 이미지의 Area Ratio를 입력받습니다.
# - Reference에서 만든 기준값과 비교합니다.
# - 뚜껑이 있다고 판정하면 cap_on, OK를 반환합니다.
# - 뚜껑이 없다고 판정하면 cap_off, NG를 반환합니다.
# ============================================================

def judge(area_ratio):

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
            "cap_on",
            "OK",
            "",
        )

    return (
        "cap_off",
        "NG",
        "NG_CAP_AREA",
    )


# ============================================================
# ⑩ Test 이미지 4장 및 평가 변수 준비
# - Reference로 사용한 001 이미지는 제외합니다.
# - cap_on 2장, cap_off 2장을 검사합니다.
# - expected는 각 이미지의 실제 정답입니다.
# - total   : 실제로 검사한 이미지 수
# - correct : 정답과 일치한 이미지 수
# ============================================================

TEST_IMAGES = [
    (
        "cap_on",
        CAP_ON_DIR
        / "cap_on_002.jpg",
    ),
    (
        "cap_on",
        CAP_ON_DIR
        / "cap_on_003.jpg",
    ),
    (
        "cap_off",
        CAP_OFF_DIR
        / "cap_off_002.jpg",
    ),
    (
        "cap_off",
        CAP_OFF_DIR
        / "cap_off_003.jpg",
    ),
]

rows = []

total = 0
correct = 0


# ============================================================
# ⑪ Test 이미지에 동일한 검사 규칙 적용
# - 새로운 이미지를 한 장씩 불러옵니다.
# - measure()를 이용하여 Area Ratio를 측정합니다.
# - judge()로 cap_on / cap_off 및 OK / NG를 판정합니다.
# - 읽지 못한 이미지는 검사에서 제외합니다.
# ============================================================

for expected, image_path in TEST_IMAGES:

    measured = measure(
        image_path
    )

    if measured is None:

        print(
            "읽기 실패:",
            image_path,
        )

        continue

    predicted, decision, reason = (
        judge(
            measured[
                "largest_area_ratio"
            ]
        )
    )


    # ========================================================
    # ⑫ 예측 결과와 실제 정답 비교
    # - expected  : 실제 정답(cap_on / cap_off)
    # - predicted : 코드가 예측한 결과
    # - is_correct : 두 값의 일치 여부
    # - 전체 검사 수와 정답 수를 누적합니다.
    # ========================================================

    is_correct = (
        predicted == expected
    )

    total += 1

    if is_correct:
        correct += 1


    # ========================================================
    # ⑬ 검사 결과 이미지를 생성하고 저장
    # - 검출된 Contour를 초록색으로 표시합니다.
    # - 예측 결과와 Area Ratio를 이미지에 표시합니다.
    # - 정답과 일치하면 초록색, 불일치하면 빨간색입니다.
    # - 이미지별 결과를 JPG 파일로 저장합니다.
    # ========================================================

    result = measured[
        "image"
    ].copy()

    contour = measured[
        "largest_contour"
    ]

    if contour is not None:

        cv2.drawContours(
            result,
            [contour],
            -1,
            (0, 255, 0),
            2,
        )

    text = (
        f"{predicted} "
        f"{measured['largest_area_ratio']:.3f}"
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
        if is_correct
        else (
            0,
            0,
            255,
        ),
        2,
    )

    output_path = (
        OUTPUT_DIR
        / f"{image_path.stem}_result.jpg"
    )

    cv2.imwrite(
        str(output_path),
        result,
    )


    # ========================================================
    # ⑭ 이미지별 검사 결과를 CSV 저장 목록에 추가
    # - 파일명, 실제 정답, 예측값을 기록합니다.
    # - 정답 여부와 Area Ratio도 저장합니다.
    # - Contour 수, 중심점, 판정 사유를 함께 기록합니다.
    # ========================================================

    rows.append(
        {
            "file": image_path.name,
            "expected": expected,
            "predicted": predicted,
            "correct": is_correct,
            "largest_area_ratio": measured[
                "largest_area_ratio"
            ],
            "valid_count": measured[
                "valid_count"
            ],
            "center_x_norm": measured[
                "center_x_norm"
            ],
            "center_y_norm": measured[
                "center_y_norm"
            ],
            "decision": decision,
            "reason": reason,
        }
    )


# ============================================================
# ⑮ 검사 이미지 수 확인 및 CSV 보고서 저장
# - 검사한 이미지가 하나도 없다면 프로그램을 종료합니다.
# - 각 이미지의 검사 결과를 CSV 표로 저장합니다.
# - utf-8-sig로 저장하여 Excel에서 한글을 읽기 쉽게 합니다.
# ============================================================

if total == 0:

    print(
        "검사할 이미지가 없습니다."
    )

    raise SystemExit

with REPORT_PATH.open(
    "w",
    newline="",
    encoding="utf-8-sig",
) as file:

    fieldnames = [
        "file",
        "expected",
        "predicted",
        "correct",
        "largest_area_ratio",
        "valid_count",
        "center_x_norm",
        "center_y_norm",
        "decision",
        "reason",
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
# ⑯ 최종 Batch Test 성능 결과 출력
# - 확인 이미지 수 : 검사에 사용된 전체 이미지 수
# - 기준과 일치   : 정답과 예측값이 일치한 수
# - 불일치        : 틀리게 판정한 이미지 수
# - Accuracy      : 정답 수 / 전체 검사 수
# - 결과 CSV와 이미지 저장 경로를 출력합니다.
# ============================================================

print()
print("==============================")
print("Batch Test 결과")
print("==============================")

print(
    "확인 이미지 수:",
    total,
)

print(
    "기준과 일치:",
    correct,
)

print(
    "불일치:",
    total - correct,
)

print(
    "Accuracy:",
    round(
        correct / total,
        3,
    ),
)

print(
    "CSV:",
    REPORT_PATH,
)

print(
    "결과 이미지:",
    OUTPUT_DIR,
)