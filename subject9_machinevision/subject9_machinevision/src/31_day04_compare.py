from pathlib import Path
import csv


# --------------------------------------------------
# 1. 기존 검사 결과 CSV와 최종 비교 CSV 경로 설정
#
# 무엇을 하나요?
# - 앞에서 실행한 4가지 검사 방법의 CSV 경로를 지정합니다.
# - 네 가지 검사 결과를 하나로 합칠 CSV 경로를 지정합니다.
#
# 입력 파일:
# day04_template_results.csv
# day04_orb_results.csv
# day04_sift_results.csv
# day04_homography_results.csv
#
# 출력 파일:
# day04_rotation_compare.csv
#
# 왜 필요한가요?
# 각 검사 방법의 결과가 서로 다른 CSV에 저장되어 있으므로
# 하나의 표로 합치기 위해 파일 경로를 준비합니다.
# --------------------------------------------------

report_dir = Path("reports")

paths = {
    "template": report_dir / "day04_template_results.csv",
    "orb": report_dir / "day04_orb_results.csv",
    "sift": report_dir / "day04_sift_results.csv",
    "homography": report_dir / "day04_homography_results.csv",
}

output_path = (
    report_dir /
    "day04_rotation_compare.csv"
)


# --------------------------------------------------
# 2. CSV 파일을 읽어 condition 기준으로 정리하는 함수
#
# 무엇을 하나요?
# - 전달받은 CSV 파일이 존재하는지 확인합니다.
# - csv.DictReader()로 CSV 데이터를 읽습니다.
# - condition을 Key로 사용하는 딕셔너리를 만듭니다.
#
# condition이란?
# 이미지에 적용한 검사 조건을 의미합니다.
#
# normal  : 정상
# shift   : 위치 이동
# rot15   : 15도 회전
# scale90 : 90% 크기 변화
#
# 왜 condition을 Key로 사용하나요?
# 서로 다른 CSV에서 같은 조건의 결과를 찾아
# 하나의 행으로 합치기 위해서입니다.
#
# 예:
# template["normal"]
# orb["normal"]
# sift["normal"]
# homography["normal"]
#
# 위 네 결과는 모두 normal 조건의 측정값입니다.
# --------------------------------------------------

def read_results(path):

    # 입력 CSV 파일이 없으면 오류 발생
    if not path.exists():
        raise FileNotFoundError(path)

    # CSV 파일 열기
    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:

        # CSV의 컬럼명을 Key로 사용하여 읽기
        rows = csv.DictReader(file)

        # condition을 Key로 하는 딕셔너리 반환
        return {
            row["condition"]: row
            for row in rows
        }


# --------------------------------------------------
# 3. 앞에서 만든 네 가지 검사 결과 CSV 읽기
#
# 무엇을 하나요?
# - Template Matching 결과를 읽습니다.
# - ORB Matching 결과를 읽습니다.
# - SIFT Matching 결과를 읽습니다.
# - Homography 결과를 읽습니다.
#
# 각 CSV는 read_results() 함수를 이용하여
# condition을 Key로 하는 딕셔너리로 변환합니다.
#
# 목적:
# 네 가지 검사 방법의 측정값을
# 같은 이미지 조건별로 비교할 준비를 합니다.
# --------------------------------------------------

template = read_results(
    paths["template"]
)

orb = read_results(
    paths["orb"]
)

sift = read_results(
    paths["sift"]
)

homography = read_results(
    paths["homography"]
)


# --------------------------------------------------
# 4. 비교할 네 가지 이미지 조건과 결과 리스트 준비
#
# 무엇을 하나요?
# - 비교할 이미지 조건 4개를 지정합니다.
# - 최종 비교 결과를 저장할 빈 리스트를 만듭니다.
#
# conditions:
# 어떤 순서로 결과를 정리할지 결정합니다.
#
# rows:
# 네 가지 검사 방법의 측정값을 합친 결과를 저장합니다.
#
# 왜 필요한가요?
# 모든 검사 방법에서 같은 조건을 기준으로
# 결과를 정리해야 공정하게 비교할 수 있기 때문입니다.
# --------------------------------------------------

conditions = [
    "normal",
    "shift",
    "rot15",
    "scale90",
]


rows = []


# --------------------------------------------------
# 5. 같은 이미지 조건의 검사 결과를 하나의 행으로 합치기
#
# 무엇을 하나요?
# - 네 가지 이미지 조건을 하나씩 반복합니다.
# - 각 조건에 해당하는 검사 결과를 가져옵니다.
# - 측정값들을 하나의 딕셔너리로 정리합니다.
#
# 가져오는 측정값:
#
# template_score:
# Template Matching에서 계산한 유사도 점수
#
# orb_good_matches:
# ORB Ratio Test를 통과한 특징점 대응 개수
#
# sift_good_matches:
# SIFT Ratio Test를 통과한 특징점 대응 개수
#
# homography_inliers:
# RANSAC에서 선택한 Inlier 개수
#
# inlier_ratio:
# Good Match 중 Inlier가 차지하는 비율
#
# homography_status:
# Homography 계산 성공 여부
#
# 왜 필요한가요?
# 검사 방법마다 다른 CSV에 저장된 결과를
# 동일한 이미지 조건을 기준으로 연결하기 위해서입니다.
# --------------------------------------------------

for condition in conditions:

    row = {
        "condition": condition,

        # Template Matching 점수
        "template_score":
            template[condition][
                "template_score"
            ],

        # ORB Good Match 개수
        "orb_good_matches":
            orb[condition][
                "good_matches"
            ],

        # SIFT Good Match 개수
        "sift_good_matches":
            sift[condition][
                "good_matches"
            ],

        # Homography Inlier 개수
        "homography_inliers":
            homography[condition][
                "inliers"
            ],

        # Homography Inlier Ratio
        "inlier_ratio":
            homography[condition][
                "inlier_ratio"
            ],

        # Homography 계산 성공 여부
        "homography_status":
            homography[condition][
                "status"
            ],
    }


    # --------------------------------------------------
    # 6. 통합한 측정값을 최종 결과 리스트에 추가
    #
    # 무엇을 하나요?
    # - 5번에서 만든 하나의 조건별 결과를 저장합니다.
    # - normal, shift, rot15, scale90을 순서대로 추가합니다.
    #
    # 반복이 끝나면 rows에는 총 4개의 행이 들어갑니다.
    #
    # 이 데이터를 마지막에 CSV 파일로 저장합니다.
    # --------------------------------------------------

    rows.append(row)


# --------------------------------------------------
# 7. 최종 비교 CSV의 컬럼명과 순서 지정
#
# 무엇을 하나요?
# - 최종 CSV에 저장할 컬럼을 정의합니다.
# - 각 검사 방법의 측정값이 표시될 순서를 정합니다.
#
# CSV 컬럼:
#
# condition
# template_score
# orb_good_matches
# sift_good_matches
# homography_inliers
# inlier_ratio
# homography_status
#
# 왜 필요한가요?
# CSV를 열었을 때 검사 조건별 측정값이
# 일관된 순서로 표시되도록 하기 위해서입니다.
# --------------------------------------------------

fieldnames = [
    "condition",
    "template_score",
    "orb_good_matches",
    "sift_good_matches",
    "homography_inliers",
    "inlier_ratio",
    "homography_status",
]


# --------------------------------------------------
# 8. 네 가지 검사 결과를 하나의 비교 CSV로 저장
#
# 무엇을 하나요?
# - day04_rotation_compare.csv 파일을 생성합니다.
# - 7번에서 정한 컬럼명을 첫 번째 행에 작성합니다.
# - 6번에서 정리한 네 가지 조건의 결과를 저장합니다.
#
# 저장되는 행:
# normal
# shift
# rot15
# scale90
#
# 최종 결과:
# reports/day04_rotation_compare.csv
#
# 이 CSV를 이용하여
# 이미지의 위치·회전·크기 변화에 따른
# 검사 방법별 측정값을 비교할 수 있습니다.
# --------------------------------------------------

with output_path.open(
    "w",
    encoding="utf-8",
    newline="",
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=fieldnames,
    )

    # CSV 컬럼 제목 작성
    writer.writeheader()

    # 네 가지 조건의 측정 결과 작성
    writer.writerows(rows)


# --------------------------------------------------
# 9. 저장 경로와 조건별 측정 결과를 화면에 출력
#
# 무엇을 하나요?
# - 최종 CSV 파일의 저장 경로를 출력합니다.
# - 네 가지 이미지 조건의 주요 측정값을 출력합니다.
#
# 출력 항목:
# condition
# Template Score
# ORB Good Match
# SIFT Good Match
# Homography Inlier
# Homography Status
#
# 목적:
# CSV 파일을 직접 열기 전에도
# 터미널에서 검사 방법별 결과를 확인할 수 있습니다.
#
# 주의:
# 이 코드는 측정값을 비교표로 정리하는 프로그램입니다.
# 어떤 검사 방법이 가장 우수한지 자동으로 판단하거나
# OK / NG 판정을 수행하지는 않습니다.
# --------------------------------------------------

print("저장:", output_path)
print()


for row in rows:

    print(
        row["condition"],
        "| Template:",
        row["template_score"],
        "| ORB:",
        row["orb_good_matches"],
        "| SIFT:",
        row["sift_good_matches"],
        "| Inlier:",
        row["homography_inliers"],
        "| Status:",
        row["homography_status"],
    )