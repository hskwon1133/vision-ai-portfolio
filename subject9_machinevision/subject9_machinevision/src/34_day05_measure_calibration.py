from pathlib import Path
import csv

# 앞에서 작성한 공통 측정 함수 가져오기
#
# prepare_reference():
# - 기준 로고의 SIFT 특징점을 준비합니다.
#
# measure_target():
# - 이미지 한 장에서 Template Matching,
#   SIFT, Homography 측정값을 계산합니다.

from day05_logo_features import (
    prepare_reference,
    measure_target,
)


# --------------------------------------------------
# 1. Reference, Calibration 이미지 및 CSV 경로 설정
#
# 무엇을 하나요?
# - 검사 기준인 광천김 Reference 경로를 지정합니다.
# - Calibration 이미지 12장이 있는 폴더를 지정합니다.
# - 측정값을 저장할 CSV 경로를 지정합니다.
#
# REFERENCE_PATH:
# 비교 기준으로 사용할 정상 로고 이미지
#
# DATA_DIR:
# OK 6장과 NG 6장이 저장된 폴더
#
# OUTPUT_CSV:
# 12장의 측정 결과를 저장할 CSV 파일
#
# 왜 필요한가요?
# 프로그램이 어떤 이미지를 읽고
# 결과를 어디에 저장해야 하는지 알아야 하기 때문입니다.
# --------------------------------------------------

REFERENCE_PATH = Path(
    "samples/day04/reference/reference_label.jpg"
)

DATA_DIR = Path(
    "samples/day05/resized"
)

OUTPUT_CSV = Path(
    "reports/day05_calibration_measurements.csv"
)


# --------------------------------------------------
# 2. Calibration 이미지 목록과 실제 상태(OK/NG) 준비
#
# 무엇을 하나요?
# - find_calibration_images() 함수를 정의합니다.
# - ok_로 시작하는 이미지들을 찾습니다.
# - ng_로 시작하는 이미지들을 찾습니다.
# - 각 이미지에 실제 정답인 OK 또는 NG를 연결합니다.
#
# 예:
# ok_01.jpg → OK
# ok_02.jpg → OK
# ng_01.jpg → NG
# ng_02.jpg → NG
#
# 왜 필요한가요?
# 나중에 프로그램이 측정한 값과
# 사람이 알고 있는 실제 정상/불량 상태를
# 비교하여 검사 기준을 만들기 위해서입니다.
#
# 주의:
# 지금은 파일명에 따라 실제 상태를 기록합니다.
# 프로그램이 OK/NG를 예측하는 단계는 아닙니다.
# --------------------------------------------------

def find_calibration_images():

    # 이미지 경로와 실제 정답을 저장할 리스트
    image_infos = []


    # --------------------------------------------------
    # 2-1. 정상(OK) 이미지 목록 만들기
    #
    # - ok_*.jpg 패턴의 이미지를 찾습니다.
    # - sorted()로 파일명 순서대로 정렬합니다.
    # - 이미지 경로와 "OK"를 함께 저장합니다.
    # --------------------------------------------------

    # OK 6장
    for image_path in sorted(
        DATA_DIR.glob("ok_*.jpg")
    ):

        image_infos.append(
            (
                image_path,
                "OK"
            )
        )


    # --------------------------------------------------
    # 2-2. 불량(NG) 이미지 목록 만들기
    #
    # - ng_*.jpg 패턴의 이미지를 찾습니다.
    # - sorted()로 파일명 순서대로 정렬합니다.
    # - 이미지 경로와 "NG"를 함께 저장합니다.
    # --------------------------------------------------

    # NG 6장
    for image_path in sorted(
        DATA_DIR.glob("ng_*.jpg")
    ):

        image_infos.append(
            (
                image_path,
                "NG"
            )
        )


    # 준비한 이미지 경로와 정답 목록 반환
    return image_infos


# --------------------------------------------------
# 3. 결과 폴더 생성 및 Reference 특징점 준비
#
# 무엇을 하나요?
# - main() 함수를 정의합니다.
# - 결과 CSV를 저장할 reports 폴더를 준비합니다.
# - prepare_reference() 함수를 호출합니다.
# - Reference의 SIFT 특징점을 추출합니다.
# - 기준 이미지의 특징점 개수를 출력합니다.
#
# 왜 필요한가요?
# Calibration 이미지 12장 모두 같은 Reference를
# 사용하므로 기준 특징점을 한 번만 계산하여
# 반복해서 재사용하기 위해서입니다.
#
# reference_data에는:
# - Reference 원본 이미지
# - Grayscale 이미지
# - SIFT 객체
# - Keypoint / Descriptor
# - 이미지 크기
# 등의 정보가 저장됩니다.
# --------------------------------------------------

def main():

    OUTPUT_CSV.parent.mkdir(
        parents=True,
        exist_ok=True
    )


    # 기준 로고 준비
    reference_data = prepare_reference(
        REFERENCE_PATH
    )


    # 기준 로고의 특징점 개수 출력
    print(
        "Reference Keypoints:",
        len(
            reference_data["keypoints"]
        )
    )


    # --------------------------------------------------
    # 4. Calibration 이미지 목록과 수량 확인
    #
    # 무엇을 하나요?
    # - 2번에서 만든 함수를 호출합니다.
    # - 이미지 경로와 실제 정답을 가져옵니다.
    # - Calibration 이미지의 전체 개수를 확인합니다.
    #
    # 이번 수업에서 예상하는 데이터:
    # OK 6장 + NG 6장 = 총 12장
    #
    # 왜 필요한가요?
    # 측정하기 전에 필요한 데이터가
    # 준비되어 있는지 확인하기 위해서입니다.
    #
    # 주의:
    # 현재 코드는 12장이 아니어도 경고만 출력하고
    # 프로그램을 계속 진행합니다.
    #
    # rows:
    # 이미지별 측정값을 저장할 빈 리스트입니다.
    # --------------------------------------------------

    image_infos = (
        find_calibration_images()
    )


    print(
        "Calibration Images:",
        len(image_infos)
    )


    if len(image_infos) != 12:

        print(
            "주의: Calibration 이미지는 "
            "OK 6장 + NG 6장 = 12장을 예상합니다."
        )


    rows = []


    # --------------------------------------------------
    # 5. Calibration 이미지 12장에 같은 측정 반복
    #
    # 무엇을 하나요?
    # - Calibration 이미지를 한 장씩 가져옵니다.
    # - measure_target() 공통 측정 함수를 호출합니다.
    # - 각 이미지의 측정값을 계산합니다.
    # - 측정값과 실제 정답을 하나의 딕셔너리로 정리합니다.
    #
    # measure_target() 내부 처리:
    #
    # ① Target 이미지 읽기
    # ② Template Matching
    # ③ SIFT 특징점 추출
    # ④ BFMatcher로 특징점 비교
    # ⑤ Ratio Test
    # ⑥ RANSAC + Homography
    # ⑦ Inlier Count / Inlier Ratio 계산
    #
    # 왜 필요한가요?
    # 모든 정상·불량 이미지에서 같은 측정값을
    # 얻어야 나중에 공정하게 비교할 수 있기 때문입니다.
    #
    # 주의:
    # 현재는 특징값을 측정하는 단계입니다.
    # 아직 Threshold나 OK/NG 판정 Rule은 적용하지 않습니다.
    # --------------------------------------------------

    for image_path, actual in image_infos:

        # 이미지 한 장의 특징값 측정
        measurement = measure_target(
            image_path,
            reference_data
        )


        # --------------------------------------------------
        # 5-1. 이미지 한 장의 측정값을 딕셔너리로 정리
        #
        # file:
        # 검사 이미지 이름
        #
        # actual:
        # 파일명으로 구분한 실제 정답(OK/NG)
        #
        # template_score:
        # 기준 로고와 가장 비슷한 영역의 유사도
        #
        # sift_good_matches:
        # Ratio Test를 통과한 SIFT 특징점 대응 개수
        #
        # homography_found:
        # Homography 계산 성공 여부
        # 1 = 성공 / 0 = 실패
        #
        # homography_inliers:
        # RANSAC의 변환 관계에 잘 맞는 대응점 개수
        #
        # inlier_ratio:
        # Inlier Count / Good Match Count
        #
        # round(값, 6):
        # 측정값을 소수점 여섯째 자리까지 반올림
        # --------------------------------------------------

        row = {

            "file":
                image_path.name,

            "actual":
                actual,

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
        }


        # --------------------------------------------------
        # 5-2. 측정 결과를 리스트에 추가하고 출력
        #
        # 무엇을 하나요?
        # - 이미지 한 장의 측정 결과를 rows에 추가합니다.
        # - 현재 이미지의 결과를 터미널에 출력합니다.
        #
        # 반복문이 끝나면 rows에는
        # 모든 Calibration 이미지의 측정값이 저장됩니다.
        # --------------------------------------------------

        rows.append(
            row
        )


        print(
            row
        )


    # --------------------------------------------------
    # 6. Calibration 전체 측정값을 CSV 파일로 저장
    #
    # 무엇을 하나요?
    # - CSV의 컬럼명을 정의합니다.
    # - CSV 파일을 생성합니다.
    # - 각 이미지의 측정값을 한 행씩 저장합니다.
    #
    # 저장할 파일:
    # reports/day05_calibration_measurements.csv
    #
    # utf-8-sig:
    # Excel 등에서 한글이 깨지는 문제를 줄이기 위한 인코딩
    #
    # writer.writeheader():
    # CSV 첫 번째 행에 컬럼명 작성
    #
    # writer.writerows(rows):
    # 모든 Calibration 측정값 저장
    #
    # 왜 필요한가요?
    # 다음 실습에서 정상과 불량의 측정값을 비교하고
    # Threshold 후보를 찾기 위해서입니다.
    # --------------------------------------------------

    fieldnames = [

        "file",

        "actual",

        "template_score",

        "sift_good_matches",

        "homography_found",

        "homography_inliers",

        "inlier_ratio",
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
    # 7. Calibration 측정 완료 및 저장 결과 확인
    #
    # 무엇을 하나요?
    # - Calibration 측정이 끝났음을 출력합니다.
    # - 측정한 이미지의 전체 개수를 출력합니다.
    # - 결과 CSV 파일의 저장 경로를 출력합니다.
    #
    # 왜 필요한가요?
    # 프로그램이 전체 이미지를 처리하고
    # 결과를 어디에 저장했는지 확인하기 위해서입니다.
    #
    # 이 단계에서 만드는 것은 측정 결과 CSV이며,
    # 최종 검사 기준은 다음 실습에서 결정합니다.
    # --------------------------------------------------

    print()
    print("=" * 70)

    print(
        "Calibration 측정 완료"
    )

    print(
        "전체 이미지:",
        len(rows)
    )

    print(
        "저장:",
        OUTPUT_CSV
    )

    print("=" * 70)


# --------------------------------------------------
# 8. Python 파일을 직접 실행할 때 main() 호출
#
# 무엇을 하나요?
# - 이 파일을 직접 실행한 경우 main()을 호출합니다.
#
# __name__ == "__main__":
# 현재 Python 파일을 직접 실행하고 있는지 확인합니다.
#
# 왜 필요한가요?
# 다른 파일에서 이 코드를 import할 때
# Calibration 측정 작업이 자동 실행되지 않도록
# 구분하기 위해서입니다.
#
# 실행 명령어:
# python src/34_day05_measure_calibration.py
# --------------------------------------------------

if __name__ == "__main__":
    main()