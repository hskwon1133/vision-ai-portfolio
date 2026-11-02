from pathlib import Path

# --------------------------------------------------
# 앞에서 작성한 공통 측정 함수 가져오기
#
# prepare_reference():
# - 기준 로고의 SIFT 특징점을 준비하는 함수
#
# measure_target():
# - 검사 이미지 한 장에서
#   Template, SIFT, Homography 측정값을 구하는 함수
#
# 두 함수는 앞에서 작성한
# day05_logo_features.py에 정의되어 있습니다.
# --------------------------------------------------

from day05_logo_features import (
    prepare_reference,
    measure_target,
)


# --------------------------------------------------
# 검사에 사용할 이미지 경로 설정
#
# REFERENCE_PATH:
# - 비교 기준인 광천김 로고 이미지
#
# TARGET_PATH:
# - 공통 측정 함수가 정상적으로 동작하는지
#   확인할 정상 이미지 한 장
#
# 이번에는 Calibration 12장을 모두 검사하지 않고
# ok_01.jpg 한 장만 먼저 측정합니다.
# --------------------------------------------------

REFERENCE_PATH = Path(
    "samples/day04/reference/reference_label.jpg"
)

TARGET_PATH = Path(
    "samples/day05/resized/ok_01.jpg"
)


# --------------------------------------------------
# 1. Reference 이미지의 특징점 준비
#
# 무엇을 하나요?
# - prepare_reference() 함수를 호출합니다.
# - 기준 로고 이미지를 읽습니다.
# - Grayscale로 변환합니다.
# - SIFT Keypoint와 Descriptor를 생성합니다.
# - 측정에 필요한 Reference 정보를 반환받습니다.
#
# 왜 필요한가요?
# 이후 Target 이미지와 비교할
# 기준 로고의 특징점 정보가 필요하기 때문입니다.
#
# reference_data:
# - Reference 원본 이미지
# - Grayscale 이미지
# - SIFT 객체
# - Keypoint
# - Descriptor
# - 이미지 가로·세로 크기
#
# Reference Keypoints:
# 기준 이미지에서 추출한 특징점 개수를 출력합니다.
# --------------------------------------------------

reference_data = prepare_reference(
    REFERENCE_PATH
)


print(
    "Reference Keypoints:",
    len(
        reference_data["keypoints"]
    )
)


# --------------------------------------------------
# 2. 실제 Target 이미지 한 장 측정
#
# 무엇을 하나요?
# - measure_target() 함수를 호출합니다.
# - 검사할 ok_01.jpg와 Reference 정보를 전달합니다.
#
# 함수 내부에서 수행하는 작업:
#
# ① Target 이미지를 읽습니다.
# ② Template Matching으로 유사도를 측정합니다.
# ③ SIFT Keypoint와 Descriptor를 추출합니다.
# ④ BFMatcher로 특징점을 비교합니다.
# ⑤ Ratio Test로 Good Match를 선별합니다.
# ⑥ RANSAC으로 Homography를 계산합니다.
# ⑦ Inlier Count와 Inlier Ratio를 계산합니다.
#
# 왜 필요한가요?
# 앞에서 작성한 공통 측정 함수가
# 실제 이미지에서도 정상적으로 실행되는지
# 확인하기 위해서입니다.
#
# measurement:
# 검사 이미지 한 장의 측정값이 저장된 딕셔너리
#
# 주의:
# 이 단계에서는 아직 OK / NG를 판정하지 않습니다.
# --------------------------------------------------

measurement = measure_target(
    TARGET_PATH,
    reference_data
)


# --------------------------------------------------
# 3. Target 이미지의 전체 측정 결과 확인
#
# 무엇을 하나요?
# - 검사한 Target 이미지 이름을 출력합니다.
# - Template Matching 점수를 출력합니다.
# - SIFT Good Match 개수를 출력합니다.
# - Homography 계산 성공 여부를 출력합니다.
# - Homography Inlier 개수를 출력합니다.
# - Inlier Ratio를 출력합니다.
#
# 왜 필요한가요?
# 이후 Calibration 12장을 검사하기 전에
# 공통 측정 함수가 정상적으로 값을 반환하는지
# 한 장의 이미지로 확인하기 위해서입니다.
#
# 측정값 설명:
#
# Template Score:
# 기준 로고와 가장 유사한 영역의 점수
#
# SIFT Good Match:
# Ratio Test를 통과한 특징점 대응 개수
#
# Homography Found:
# Homography 계산 성공 여부
# 1 = 계산 성공, 0 = 계산 실패
#
# Homography Inlier:
# RANSAC의 변환 관계에 잘 맞는 대응점 개수
#
# Inlier Ratio:
# Inlier Count / Good Match Count
#
# round(값, 4):
# 출력할 숫자를 소수점 넷째 자리까지 반올림합니다.
# --------------------------------------------------

print()
print("=" * 60)


# 검사한 Target 이미지 이름
print(
    "Target:",
    TARGET_PATH.name
)


# Template Matching의 최대 유사도
print(
    "Template Score:",
    round(
        measurement["template_score"],
        4
    )
)


# Ratio Test를 통과한 SIFT Good Match 개수
print(
    "SIFT Good Match:",
    measurement[
        "sift_good_matches"
    ]
)


# Homography 계산 성공 여부
print(
    "Homography Found:",
    measurement[
        "homography_found"
    ]
)


# RANSAC이 선택한 Inlier 개수
print(
    "Homography Inlier:",
    measurement[
        "homography_inliers"
    ]
)


# Good Match 중 Inlier가 차지하는 비율
print(
    "Inlier Ratio:",
    round(
        measurement["inlier_ratio"],
        4
    )
)


print("=" * 60)