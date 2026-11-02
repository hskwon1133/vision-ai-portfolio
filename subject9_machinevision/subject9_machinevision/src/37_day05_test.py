from pathlib import Path
import csv
import json

import cv2
import numpy as np

# 앞에서 만든 공통 측정 함수
#
# prepare_reference():
# 기준 로고의 특징점을 준비합니다.
#
# measure_target():
# 이미지 한 장의 Template, SIFT, Homography
# 측정값을 계산합니다.

from day05_logo_features import (
    prepare_reference,
    measure_target,
)


# --------------------------------------------------
# 1. Rule JSON, Test 이미지 및 결과 저장 경로 설정
#
# 무엇을 하나요?
# - 동결한 Baseline Rule JSON 경로를 지정합니다.
# - Test 이미지가 있는 폴더를 지정합니다.
# - 검사 결과 이미지와 CSV의 저장 경로를 정합니다.
#
# RULE_PATH:
# Calibration에서 선택한 판정 기준이 저장된 JSON
#
# DATA_DIR:
# 정상 Test 3장과 불량 Test 3장이 있는 폴더
#
# OUTPUT_DIR:
# 판정 결과를 표시한 이미지를 저장할 폴더
#
# OUTPUT_CSV:
# 이미지별 측정값과 OK/NG 판정을 기록할 파일
#
# 왜 필요한가요?
# Test에서는 기준값을 새로 만드는 것이 아니라
# 앞에서 동결한 JSON을 읽어서 사용하기 때문입니다.
# --------------------------------------------------

RULE_PATH = Path(
    "configs/day05_logo_rules.json"
)

DATA_DIR = Path(
    "samples/day05/resized"
)

OUTPUT_DIR = Path(
    "outputs/day05/test"
)

OUTPUT_CSV = Path(
    "reports/day05_test_results.csv"
)


# --------------------------------------------------
# 2. Test 이미지 목록과 실제 정답 준비
#
# 무엇을 하나요?
# - test_ok_*.jpg 패턴의 이미지를 찾습니다.
# - test_ng_*.jpg 패턴의 이미지를 찾습니다.
# - 각 이미지에 실제 정답인 OK 또는 NG를 연결합니다.
#
# 예상 데이터:
# test_ok_01.jpg ~ test_ok_03.jpg
# test_ng_01.jpg ~ test_ng_03.jpg
#
# actual:
# 사람이 미리 알고 있는 실제 정답입니다.
#
# 주의:
# 여기서는 파일명으로 실제 정답을 지정합니다.
# 프로그램이 예측한 결과는 나중에 따로 계산합니다.
# --------------------------------------------------

def find_test_images():

    image_infos = []

    # --------------------------------------------------
    # 2-1. 실제 정상(OK) Test 이미지 찾기
    #
    # - test_ok_로 시작하는 이미지를 찾습니다.
    # - 파일명 순서대로 정렬합니다.
    # - 이미지 경로와 실제 정답 "OK"를 저장합니다.
    # --------------------------------------------------

    for image_path in sorted(
        DATA_DIR.glob("test_ok_*.jpg")
    ):

        image_infos.append(
            (
                image_path,
                "OK"
            )
        )

    # --------------------------------------------------
    # 2-2. 실제 불량(NG) Test 이미지 찾기
    #
    # - test_ng_로 시작하는 이미지를 찾습니다.
    # - 파일명 순서대로 정렬합니다.
    # - 이미지 경로와 실제 정답 "NG"를 저장합니다.
    # --------------------------------------------------

    for image_path in sorted(
        DATA_DIR.glob("test_ng_*.jpg")
    ):

        image_infos.append(
            (
                image_path,
                "NG"
            )
        )

    # 이미지 경로와 실제 정답 목록 반환
    return image_infos


# --------------------------------------------------
# 3. 측정값에 동결한 Rule을 적용하는 판정 함수
#
# 무엇을 하나요?
# - measure_target()에서 계산한 측정값을 받습니다.
# - JSON에 저장된 검사 기준과 비교합니다.
# - 조건을 만족하지 못한 이유를 기록합니다.
# - 최종 OK 또는 NG를 판정합니다.
#
# 이번 Baseline Rule:
#
# Homography 계산 성공
#           AND
# SIFT Good Match >= 129
#           AND
# Inlier Ratio >= 0.8664
#
# 세 조건을 모두 만족하면 OK입니다.
# 하나라도 만족하지 못하면 NG입니다.
#
# 입력:
# measurement → 이미지의 측정값
# rules       → JSON에서 읽은 검사 기준
#
# 반환:
# predicted → OK 또는 NG
# reasons   → 기준을 만족하지 못한 이유
# --------------------------------------------------

def judge_measurement(
    measurement,
    rules,
):

    # 실패 이유를 저장할 빈 리스트
    reasons = []

    # JSON의 required_rules 가져오기
    required = rules[
        "required_rules"
    ]

    # --------------------------------------------------
    # 3-1. Homography 계산 성공 여부 검사
    #
    # 무엇을 하나요?
    # - JSON에서 Homography가 필수 조건인지 확인합니다.
    # - 필수인데 계산에 실패했다면 NG 이유를 기록합니다.
    #
    # HOMOGRAPHY_NOT_FOUND:
    # 두 이미지 사이의 Homography 계산에 실패
    #
    # 주의:
    # Homography 성공은 로고가 정상이라는 의미가 아니라
    # 기하학적 변환 행렬을 계산했다는 의미입니다.
    # --------------------------------------------------

    if (
        required["homography_found"]
        and measurement["homography_found"] != 1
    ):

        reasons.append(
            "HOMOGRAPHY_NOT_FOUND"
        )

    # --------------------------------------------------
    # 3-2. SIFT Good Match 기준 검사
    #
    # 무엇을 하나요?
    # - 측정한 SIFT Good Match 개수를 확인합니다.
    # - JSON의 최소 기준값과 비교합니다.
    #
    # 예:
    # 측정값 = 105
    # 기준값 = 129
    #
    # 105 < 129 이므로 기준 미달입니다.
    #
    # LOW_SIFT_MATCH:
    # 특징점 대응 개수가 부족하다는 의미입니다.
    # --------------------------------------------------

    if (
        measurement["sift_good_matches"]
        < required["sift_good_matches_min"]
    ):

        reasons.append(
            "LOW_SIFT_MATCH"
        )

    # --------------------------------------------------
    # 3-3. Inlier Ratio 기준 검사
    #
    # 무엇을 하나요?
    # - 측정한 Inlier Ratio를 확인합니다.
    # - JSON의 최소 기준값과 비교합니다.
    #
    # 예:
    # 측정값 = 0.8397
    # 기준값 = 0.8664
    #
    # 0.8397 < 0.8664 이므로 기준 미달입니다.
    #
    # LOW_INLIER_RATIO:
    # 일관된 특징점 대응의 비율이 부족하다는 의미입니다.
    # --------------------------------------------------

    if (
        measurement["inlier_ratio"]
        < required["inlier_ratio_min"]
    ):

        reasons.append(
            "LOW_INLIER_RATIO"
        )

    # --------------------------------------------------
    # 3-4. 검사 실패 이유를 확인하여 최종 판정
    #
    # 무엇을 하나요?
    # - reasons 리스트가 비어 있는지 확인합니다.
    #
    # 실패 이유 없음:
    # 세 조건을 모두 통과했으므로 OK
    #
    # 실패 이유 있음:
    # 하나 이상의 조건을 통과하지 못했으므로 NG
    #
    # 반환값:
    # predicted와 reasons
    # --------------------------------------------------

    if not reasons:

        predicted = "OK"

    else:

        predicted = "NG"

    return predicted, reasons


# --------------------------------------------------
# 4. 실제 정답과 프로그램 판정을 비교하는 함수
#
# 무엇을 하나요?
# - 사람이 알고 있는 실제 정답(actual)과
#   프로그램의 판정(predicted)을 비교합니다.
# - 결과를 TP, TN, FP, FN으로 구분합니다.
#
# 이번 제조검사에서는 NG를 Positive로 정의합니다.
#
# TP:
# 실제 NG → 프로그램 NG
# 불량을 올바르게 검출
#
# TN:
# 실제 OK → 프로그램 OK
# 정상을 올바르게 통과
#
# FP:
# 실제 OK → 프로그램 NG
# 정상을 불량으로 잘못 판정
#
# FN:
# 실제 NG → 프로그램 OK
# 불량을 정상으로 잘못 통과
#
# 왜 필요한가요?
# 전체 Accuracy뿐 아니라 어떤 종류의
# 판정 오류가 발생했는지 확인하기 위해서입니다.
# --------------------------------------------------

def classify_error(
    actual,
    predicted,
):

    # 실제 NG를 NG로 올바르게 검출
    if (
        actual == "NG"
        and predicted == "NG"
    ):
        return "TP"

    # 실제 OK를 OK로 올바르게 판정
    if (
        actual == "OK"
        and predicted == "OK"
    ):
        return "TN"

    # 실제 OK를 NG로 잘못 판정
    if (
        actual == "OK"
        and predicted == "NG"
    ):
        return "FP"

    # 실제 NG를 OK로 잘못 통과
    if (
        actual == "NG"
        and predicted == "OK"
    ):
        return "FN"

    return "UNKNOWN"


# --------------------------------------------------
# 5. 판정 결과와 로고 위치를 이미지에 표시하는 함수
#
# 무엇을 하나요?
# - 검사 이미지의 복사본을 만듭니다.
# - Homography로 계산한 로고 위치를 사각형으로 표시합니다.
# - OK/NG에 따라 사각형 색상을 구분합니다.
# - 실제 정답, 프로그램 판정, 측정값, NG 이유를 표시합니다.
#
# 왜 필요한가요?
# CSV의 숫자만 보는 것보다 실제 이미지에서
# 어떤 위치를 찾았고 왜 NG가 되었는지
# 쉽게 확인할 수 있기 때문입니다.
#
# 주의:
# 사각형은 Homography로 계산한 위치이며,
# 실제 로고가 정확히 검출되었음을 보증하지 않습니다.
# --------------------------------------------------

def annotate_result(
    measurement,
    actual,
    predicted,
    reasons,
):

    # 원본 Target을 수정하지 않도록 복사
    image = (
        measurement["target_image"]
        .copy()
    )

    # Homography로 계산한 로고 영역 좌표
    polygon = measurement.get(
        "polygon"
    )

    # --------------------------------------------------
    # 5-1. Homography로 찾은 로고 영역 표시
    #
    # 무엇을 하나요?
    # - polygon 좌표가 있는지 확인합니다.
    # - 네 모서리 좌표를 정수로 변환합니다.
    # - 네 점을 연결할 수 있는 형태로 변환합니다.
    #
    # np.int32():
    # 좌표를 정수로 변환합니다.
    #
    # reshape(-1, 1, 2):
    # OpenCV의 polylines()에서 사용할 배열 형태입니다.
    # --------------------------------------------------

    if polygon is not None:

        points = np.int32(
            polygon
        ).reshape(
            -1,
            1,
            2
        )

        # --------------------------------------------------
        # 5-2. 프로그램 판정에 따라 사각형 색상 선택
        #
        # Predicted = OK:
        # 초록색 (0, 255, 0)
        #
        # Predicted = NG:
        # 빨간색 (0, 0, 255)
        #
        # OpenCV는 BGR 색상 순서를 사용합니다.
        #
        # 중요:
        # 색상은 프로그램 판정을 나타냅니다.
        # 실제 정답과 일치한다는 뜻은 아닙니다.
        # --------------------------------------------------

        if predicted == "OK":

            line_color = (
                0,
                255,
                0
            )

        else:

            line_color = (
                0,
                0,
                255
            )

        # --------------------------------------------------
        # 5-3. 계산된 로고 영역에 사각형 그리기
        #
        # cv2.polylines():
        # 여러 점을 선으로 연결하는 함수
        #
        # True:
        # 마지막 점과 처음 점을 연결하여 도형을 닫습니다.
        #
        # 3:
        # 선의 두께입니다.
        # --------------------------------------------------

        cv2.polylines(
            image,
            [points],
            True,
            line_color,
            3
        )

    # --------------------------------------------------
    # 5-4. 검사 실패 이유를 표시할 문자열 만들기
    #
    # 무엇을 하나요?
    # - 실패 이유가 여러 개이면 +로 연결합니다.
    # - 실패 이유가 없다면 PASS로 표시합니다.
    #
    # 예:
    # LOW_SIFT_MATCH+LOW_INLIER_RATIO
    #
    # 또는:
    # PASS
    #
    # 이 문자열은 결과 이미지에 표시할 때 사용합니다.
    # --------------------------------------------------

    reason_text = (
        "+".join(reasons)
        if reasons
        else "PASS"
    )

    # --------------------------------------------------
    # 5-5. 결과 이미지에 표시할 측정 정보 준비
    #
    # 표시 항목:
    #
    # Actual:
    # 실제 정답인 OK 또는 NG
    #
    # Predicted:
    # 프로그램이 판정한 OK 또는 NG
    #
    # SIFT Good:
    # Good Match 개수
    #
    # Inlier Ratio:
    # 기하학적으로 일관된 대응점 비율
    #
    # Template:
    # Template Matching 유사도
    #
    # Reason:
    # NG 판정 이유 또는 PASS
    # --------------------------------------------------

    lines = [

        f"Actual: {actual}",

        f"Predicted: {predicted}",

        (
            "SIFT Good: "
            f"{measurement['sift_good_matches']}"
        ),

        (
            "Inlier Ratio: "
            f"{measurement['inlier_ratio']:.4f}"
        ),

        (
            "Template: "
            f"{measurement['template_score']:.4f}"
        ),

        f"Reason: {reason_text}",
    ]

    # --------------------------------------------------
    # 5-6. 결과 이미지에 측정값을 문자로 표시
    #
    # 무엇을 하나요?
    # - 준비한 lines를 한 줄씩 가져옵니다.
    # - 이미지의 왼쪽 위부터 순서대로 표시합니다.
    #
    # cv2.putText():
    # 이미지 위에 문자를 작성하는 함수
    #
    # (30, y):
    # 문자를 표시할 위치
    #
    # (255, 255, 255):
    # 흰색
    #
    # y += 35:
    # 다음 줄을 아래로 35픽셀 이동합니다.
    # --------------------------------------------------

    y = 40

    for line in lines:

        cv2.putText(
            image,
            line,
            (30, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        y += 35

    # 결과 문자가 표시된 이미지 반환
    return image


# --------------------------------------------------
# 6. 동결한 Baseline Rule JSON 읽기
#
# 무엇을 하나요?
# - main() 함수를 정의합니다.
# - 이전 실습에서 저장한 JSON이 있는지 확인합니다.
# - JSON을 Python 딕셔너리로 읽습니다.
# - 사용할 Rule 버전과 Threshold를 출력합니다.
#
# 왜 필요한가요?
# Calibration에서 결정한 기준을
# Test에서 변경하지 않고 그대로 사용하기 위해서입니다.
#
# json.load():
# JSON 파일의 데이터를 Python 객체로 읽는 함수입니다.
#
# Rule JSON이 없다면:
# 먼저 36_day05_freeze_rule.py를 실행해야 합니다.
# --------------------------------------------------

def main():

    if not RULE_PATH.exists():

        raise FileNotFoundError(
            "Rule JSON이 없습니다. "
            "먼저 36_day05_freeze_rule.py를 실행하세요."
        )

    with RULE_PATH.open(
        "r",
        encoding="utf-8"
    ) as file:

        rules = json.load(
            file
        )

    # Rule 버전과 Threshold 확인
    print("=" * 70)

    print(
        "Baseline Rule:",
        rules["baseline_version"]
    )

    print(
        "SIFT >=",
        rules[
            "required_rules"
        ][
            "sift_good_matches_min"
        ]
    )

    print(
        "Inlier Ratio >=",
        rules[
            "required_rules"
        ][
            "inlier_ratio_min"
        ]
    )


    # --------------------------------------------------
    # 7. JSON에 기록된 Reference 이미지 준비
    #
    # 무엇을 하나요?
    # - JSON에 저장된 Reference 이미지 경로를 읽습니다.
    # - prepare_reference() 공통 함수를 호출합니다.
    # - 기준 로고의 SIFT 특징점을 한 번 준비합니다.
    #
    # 왜 필요한가요?
    # Calibration과 Test에서 동일한 Reference를 사용하고
    # Target마다 특징점을 다시 계산하지 않기 위해서입니다.
    # --------------------------------------------------

    reference_path = Path(
        rules[
            "reference"
        ][
            "reference_label"
        ]
    )

    reference_data = prepare_reference(
        reference_path
    )


    # --------------------------------------------------
    # 8. Test 이미지 6장 확인 및 결과 저장 폴더 준비
    #
    # 무엇을 하나요?
    # - find_test_images() 함수를 호출합니다.
    # - 실제 OK 3장과 NG 3장의 목록을 가져옵니다.
    # - 전체 이미지 개수를 확인합니다.
    # - 결과 이미지와 CSV를 저장할 폴더를 생성합니다.
    #
    # 예상 데이터:
    # OK 3장 + NG 3장 = 총 6장
    #
    # 주의:
    # 현재 코드는 6장이 아니더라도 경고만 출력하고
    # 프로그램을 계속 진행합니다.
    #
    # rows:
    # 이미지별 검사 결과를 저장할 리스트입니다.
    # --------------------------------------------------

    image_infos = (
        find_test_images()
    )

    print(
        "Test Images:",
        len(image_infos)
    )

    if len(image_infos) != 6:

        print(
            "주의: Test 이미지는 "
            "OK 3장 + NG 3장 = 6장을 예상합니다."
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    OUTPUT_CSV.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    rows = []


    # --------------------------------------------------
    # 9. Test 이미지마다 공통 측정 함수 실행
    #
    # 무엇을 하나요?
    # - Test 이미지를 하나씩 가져옵니다.
    # - measure_target() 공통 함수를 호출합니다.
    # - 이미지의 Template, SIFT, Homography 측정값을 구합니다.
    #
    # measure_target() 내부 처리:
    #
    # ① Target 이미지 읽기
    # ② Template Matching
    # ③ SIFT Keypoint / Descriptor 추출
    # ④ BFMatcher
    # ⑤ Ratio Test
    # ⑥ RANSAC + Homography
    # ⑦ Inlier Count / Inlier Ratio 계산
    #
    # 아직 이 단계에서는 판정하지 않고
    # 측정값만 구합니다.
    # --------------------------------------------------

    for image_path, actual in image_infos:

        measurement = measure_target(
            image_path,
            reference_data
        )


        # --------------------------------------------------
        # 10. 측정값에 동결한 Baseline Rule 적용
        #
        # 무엇을 하나요?
        # - 3번에서 만든 judge_measurement()를 호출합니다.
        # - 측정값과 JSON 기준을 비교합니다.
        #
        # 모든 기준 통과:
        # predicted = OK
        #
        # 하나라도 기준 미달:
        # predicted = NG
        #
        # reasons:
        # 기준을 통과하지 못한 이유를 저장합니다.
        #
        # 왜 필요한가요?
        # Calibration에서 결정한 복합 Rule을
        # 새로운 Test 이미지에 그대로 적용하기 위해서입니다.
        # --------------------------------------------------

        predicted, reasons = (
            judge_measurement(
                measurement,
                rules
            )
        )


        # --------------------------------------------------
        # 11. 실제 정답과 프로그램 판정 비교
        #
        # 무엇을 하나요?
        # - 4번에서 만든 classify_error()를 호출합니다.
        # - 실제 정답과 예측 결과를 비교합니다.
        # - TP, TN, FP, FN 중 하나로 구분합니다.
        #
        # 예:
        # actual = OK
        # predicted = NG
        #
        # 결과:
        # FP
        #
        # 왜 필요한가요?
        # Test 이미지에서 프로그램이
        # 몇 장을 올바르게 판정했는지 평가하기 위해서입니다.
        # --------------------------------------------------

        error_type = classify_error(
            actual,
            predicted
        )


        # --------------------------------------------------
        # 12. 검사 결과 이미지 생성 및 저장
        #
        # 무엇을 하나요?
        # - annotate_result() 함수를 호출합니다.
        # - 로고 위치와 측정값을 이미지에 표시합니다.
        # - Target 이름을 이용하여 결과 파일명을 만듭니다.
        # - 결과 이미지를 JPG로 저장합니다.
        #
        # 예:
        # test_ok_01.jpg
        #        ↓
        # test_ok_01_result.jpg
        #
        # 저장 위치:
        # outputs/day05/test/
        #
        # 왜 필요한가요?
        # 실제 이미지에서 어떤 위치를 검출했고
        # 어떤 이유로 판정했는지 눈으로 확인하기 위해서입니다.
        # --------------------------------------------------

        result_image = annotate_result(
            measurement,
            actual,
            predicted,
            reasons
        )

        output_image = (
            OUTPUT_DIR
            / (
                f"{image_path.stem}"
                "_result.jpg"
            )
        )

        cv2.imwrite(
            str(output_image),
            result_image
        )


        # --------------------------------------------------
        # 13. 이미지별 판정 결과를 CSV용 데이터로 정리
        #
        # 무엇을 하나요?
        # - 현재 Test 이미지의 측정값과 판정 결과를 정리합니다.
        # - 하나의 딕셔너리로 저장합니다.
        # - rows 리스트에 추가합니다.
        # - 터미널에 현재 검사 결과를 출력합니다.
        #
        # 주요 저장 항목:
        #
        # file:
        # 이미지 이름
        #
        # actual:
        # 실제 정답
        #
        # predicted:
        # 프로그램 판정
        #
        # error_type:
        # TP / TN / FP / FN
        #
        # ng_reason:
        # NG 판정 이유 또는 PASS
        #
        # template_score:
        # Template Matching 유사도
        #
        # sift_good_matches:
        # SIFT Good Match 개수
        #
        # homography_found:
        # Homography 계산 성공 여부
        #
        # homography_inliers:
        # RANSAC Inlier 개수
        #
        # inlier_ratio:
        # Inlier 비율
        #
        # result_image:
        # 결과 이미지 저장 경로
        # --------------------------------------------------

        row = {

            "file":
                image_path.name,

            "actual":
                actual,

            "predicted":
                predicted,

            "error_type":
                error_type,

            "ng_reason":
                (
                    ";".join(reasons)
                    if reasons
                    else "PASS"
                ),

            "template_score":
                round(
                    measurement[
                        "template_score"
                    ],
                    6
                ),

            "sift_good_matches":
                measurement[
                    "sift_good_matches"
                ],

            "homography_found":
                measurement[
                    "homography_found"
                ],

            "homography_inliers":
                measurement[
                    "homography_inliers"
                ],

            "inlier_ratio":
                round(
                    measurement[
                        "inlier_ratio"
                    ],
                    6
                ),

            "result_image":
                str(output_image),
        }

        rows.append(
            row
        )

        print(
            row
        )


    # --------------------------------------------------
    # 14. Test 6장의 측정 및 판정 결과를 CSV로 저장
    #
    # 무엇을 하나요?
    # - CSV 컬럼명을 정합니다.
    # - CSV 파일을 생성합니다.
    # - Test 이미지별 측정값과 판정을 저장합니다.
    #
    # 저장 파일:
    # reports/day05_test_results.csv
    #
    # utf-8-sig:
    # Excel에서 한글이 깨질 가능성을 줄입니다.
    #
    # writer.writeheader():
    # 첫 번째 행에 컬럼명을 기록합니다.
    #
    # writer.writerows(rows):
    # 모든 이미지의 검사 결과를 기록합니다.
    #
    # 왜 필요한가요?
    # 이후 성능 평가와 실패 분석에서
    # 동일한 검사 결과를 재사용하기 위해서입니다.
    # --------------------------------------------------

    fieldnames = [
        "file",
        "actual",
        "predicted",
        "error_type",
        "ng_reason",
        "template_score",
        "sift_good_matches",
        "homography_found",
        "homography_inliers",
        "inlier_ratio",
        "result_image",
    ]

    with OUTPUT_CSV.open(
        "w",
        encoding="utf-8-sig",
        newline=""
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(
            rows
        )


    # --------------------------------------------------
    # 15. Test 전체의 TP/TN/FP/FN과 Accuracy 계산
    #
    # 무엇을 하나요?
    # - 각 이미지에서 기록한 error_type을 확인합니다.
    # - TP, TN, FP, FN의 전체 개수를 계산합니다.
    # - Test 전체 Accuracy를 계산합니다.
    #
    # Accuracy:
    # (TP + TN) / 전체 Test 이미지 수
    #
    # 예:
    # 6장 중 3장을 올바르게 판정하면
    # Accuracy = 3 / 6 = 0.5
    #
    # 왜 필요한가요?
    # Calibration에서 선택한 Rule이
    # 새로운 이미지에서도 얼마나 정확하게
    # 동작하는지 평가하기 위해서입니다.
    #
    # 주의:
    # 현재 코드는 이미 기록된 error_type을 집계하며,
    # Rule을 다시 변경하거나 재학습하지 않습니다.
    # --------------------------------------------------

    tp = sum(
        row["error_type"] == "TP"
        for row in rows
    )

    tn = sum(
        row["error_type"] == "TN"
        for row in rows
    )

    fp = sum(
        row["error_type"] == "FP"
        for row in rows
    )

    fn = sum(
        row["error_type"] == "FN"
        for row in rows
    )

    total = len(rows)

    accuracy = (
        (tp + tn) / total
        if total
        else 0.0
    )


    # --------------------------------------------------
    # 16. Test 완료 및 전체 검사 결과 출력
    #
    # 무엇을 하나요?
    # - 검사한 Test 이미지 수를 출력합니다.
    # - Accuracy를 출력합니다.
    # - TP, TN, FP, FN 개수를 출력합니다.
    # - CSV와 결과 이미지의 저장 위치를 출력합니다.
    #
    # 왜 필요한가요?
    # Baseline Rule이 Test 데이터에서
    # 어떻게 동작했는지 한눈에 확인하기 위해서입니다.
    #
    # 특히 확인할 값:
    #
    # FP:
    # 정상 이미지를 NG로 잘못 판정한 개수
    #
    # FN:
    # 불량 이미지를 OK로 잘못 통과시킨 개수
    # --------------------------------------------------

    print()

    print("=" * 70)

    print(
        "Test 완료"
    )

    print("=" * 70)

    print(
        "전체:",
        total
    )

    print(
        "Accuracy:",
        round(
            accuracy,
            4
        )
    )

    print(
        "TP:",
        tp
    )

    print(
        "TN:",
        tn
    )

    print(
        "FP:",
        fp
    )

    print(
        "FN:",
        fn
    )

    print(
        "CSV:",
        OUTPUT_CSV
    )

    print(
        "결과 이미지:",
        OUTPUT_DIR
    )


# --------------------------------------------------
# 17. Python 파일을 직접 실행하면 main() 호출
#
# 무엇을 하나요?
# - 현재 파일이 직접 실행되었는지 확인합니다.
# - 직접 실행했다면 main() 함수를 호출합니다.
#
# 왜 필요한가요?
# 다른 파일에서 이 코드를 import할 때
# Test 작업이 자동으로 실행되지 않도록 하기 위해서입니다.
#
# 실행 명령어:
# python src/37_day05_test.py
# --------------------------------------------------

if __name__ == "__main__":
    main()