#그래프(분포도) 만드는 코드
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# --------------------------------------------------
# 1. Calibration 입력 CSV와 결과 저장 경로 설정
#
# 무엇을 하나요?
# - 앞에서 측정한 Calibration CSV 경로를 지정합니다.
# - Threshold 후보별 결과를 저장할 CSV 경로를 정합니다.
# - OK/NG 측정값 분포를 저장할 그래프 경로를 정합니다.
#
# CSV_PATH:
# Calibration 12장의 실제 측정값이 저장된 파일
#
# CANDIDATE_CSV:
# 여러 Threshold 조합의 성능을 저장할 파일
#
# PLOT_PATH:
# SIFT Good Match와 Inlier Ratio의
# OK/NG 분포를 시각화할 그래프 파일
#
# 왜 필요한가요?
# 측정된 데이터를 읽고,
# 분석 결과를 다시 확인할 수 있도록
# CSV와 이미지 파일로 저장하기 위해서입니다.
# --------------------------------------------------

CSV_PATH = Path(
    "reports/day05_calibration_measurements.csv"
)

CANDIDATE_CSV = Path(
    "reports/day05_rule_candidates.csv"
)

PLOT_PATH = Path(
    "outputs/day05/calibration/sift_inlier_distribution.png"
)


# --------------------------------------------------
# 2. Threshold 후보값을 생성하는 함수
#
# 무엇을 하나요?
# - CSV의 측정값에서 최소값과 최대값을 찾습니다.
# - 최소값과 최대값 사이의 후보값을 생성합니다.
# - 기본적으로 15개의 후보값을 만듭니다.
#
# Threshold란?
# 측정값이 정상 기준을 만족하는지 판단할 경계값입니다.
#
# 예:
# SIFT Good Match >= 129
# Inlier Ratio >= 0.8664
#
# series:
# 후보를 만들 측정값 데이터
#
# count=15:
# 생성할 후보값의 기본 개수
#
# integer=False:
# 소수점 값을 사용할지 결정하는 설정
#
# integer=True:
# SIFT Good Match처럼 정수 기준값 생성
#
# integer=False:
# Inlier Ratio처럼 소수 기준값 생성
#
# 왜 필요한가요?
# Threshold를 감으로 하나만 정하는 대신
# 실제 측정값 범위에서 여러 기준을 시험하기 위해서입니다.
# --------------------------------------------------

def make_candidates(
    series,
    count=15,
    integer=False,
):

    # --------------------------------------------------
    # 2-1. 측정값을 숫자로 변환하고 빈 값 제거
    #
    # pd.to_numeric():
    # 데이터를 숫자 형태로 변환합니다.
    #
    # errors="coerce":
    # 숫자로 변환할 수 없는 값은 NaN으로 처리합니다.
    #
    # dropna():
    # NaN 값을 제거합니다.
    # --------------------------------------------------

    values = pd.to_numeric(
        series,
        errors="coerce",
    ).dropna()


    # 사용할 측정값이 없다면 빈 리스트 반환
    if values.empty:
        return []


    # --------------------------------------------------
    # 2-2. 측정값의 최소값과 최대값 찾기
    #
    # low:
    # 가장 작은 측정값
    #
    # high:
    # 가장 큰 측정값
    #
    # 이 범위에서 Threshold 후보를 생성합니다.
    # --------------------------------------------------

    low = float(
        values.min()
    )

    high = float(
        values.max()
    )


    # --------------------------------------------------
    # 2-3. 모든 측정값이 같은 경우 처리
    #
    # 최소값과 최대값이 같으면
    # 서로 다른 후보를 만들 수 없습니다.
    #
    # 따라서 현재 값 하나만 후보로 반환합니다.
    # --------------------------------------------------

    if low == high:

        if integer:
            return [int(low)]

        return [low]


    # --------------------------------------------------
    # 2-4. 최소값부터 최대값까지 후보 생성
    #
    # np.linspace():
    # 시작값과 끝값 사이를 균등하게 나누어
    # 지정한 개수의 숫자를 생성하는 함수입니다.
    #
    # 예:
    # 최소값 100, 최대값 200, count=5
    #
    # 후보:
    # 100, 125, 150, 175, 200
    #
    # 이번에는 기본적으로 15개를 생성합니다.
    # --------------------------------------------------

    candidates = np.linspace(
        low,
        high,
        count,
    )


    # --------------------------------------------------
    # 2-5. 정수 Threshold 후보 정리
    #
    # integer=True이면:
    # - 각 후보를 반올림하여 정수로 변환합니다.
    # - 중복값을 제거합니다.
    # - 작은 값부터 정렬합니다.
    #
    # SIFT Good Match 개수에 사용합니다.
    # --------------------------------------------------

    if integer:

        return sorted(
            set(
                int(round(value))
                for value in candidates
            )
        )


    # --------------------------------------------------
    # 2-6. 소수 Threshold 후보 정리
    #
    # integer=False이면:
    # - 소수점 넷째 자리까지 반올림합니다.
    # - 중복값을 제거합니다.
    # - 작은 값부터 정렬합니다.
    #
    # Inlier Ratio 기준값에 사용합니다.
    #
    # 주의:
    # 중복 제거로 최종 후보 수가 15개보다 적을 수도 있습니다.
    # --------------------------------------------------

    return sorted(
        set(
            round(
                float(value),
                4,
            )
            for value in candidates
        )
    )


# --------------------------------------------------
# 3. 실제 정답과 프로그램 판정을 비교하는 함수
#
# 무엇을 하나요?
# - Calibration 이미지의 실제 정답과
#   프로그램이 임시 Rule로 판정한 값을 비교합니다.
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
# 정상을 불량으로 잘못 판단
#
# FN:
# 실제 NG → 프로그램 OK
# 불량을 정상으로 잘못 통과
#
# 왜 필요한가요?
# Threshold 후보를 비교할 때
# 단순히 맞힌 개수뿐 아니라
# 어떤 종류의 오류가 발생했는지 확인하기 위해서입니다.
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


    # 예상하지 못한 값이 들어온 경우
    return "UNKNOWN"


# --------------------------------------------------
# 4. Calibration CSV를 읽고 데이터 수량 확인
#
# 무엇을 하나요?
# - main() 함수를 정의합니다.
# - 앞에서 만든 CSV가 존재하는지 확인합니다.
# - Pandas로 CSV를 읽습니다.
# - 전체 이미지 개수와 실제 OK/NG 개수를 확인합니다.
#
# 이번 수업에서 예상하는 데이터:
# OK 6장 + NG 6장 = 총 12장
#
# pd.read_csv():
# CSV 파일을 DataFrame 형태로 읽는 함수
#
# value_counts():
# 각 값이 몇 번 나타나는지 세는 함수
#
# 왜 필요한가요?
# Threshold를 분석하기 전에
# 사용할 Calibration 데이터의 구성을 확인하기 위해서입니다.
# --------------------------------------------------

def main():

    if not CSV_PATH.exists():

        raise FileNotFoundError(
            CSV_PATH
        )


    df = pd.read_csv(
        CSV_PATH
    )


    print("=" * 70)
    print("Calibration 데이터")
    print("=" * 70)

    print(
        "전체 이미지:",
        len(df)
    )


    print(
        df["actual"].value_counts()
    )


    # --------------------------------------------------
    # 5. OK와 NG의 측정값 기본 통계 확인
    #
    # 무엇을 하나요?
    # - 분석할 숫자 컬럼 4개를 선택합니다.
    # - 실제 OK/NG를 구분하여 통계를 계산합니다.
    #
    # 측정 항목:
    # template_score:
    # Template Matching 유사도
    #
    # sift_good_matches:
    # SIFT Good Match 개수
    #
    # homography_inliers:
    # RANSAC Inlier 개수
    #
    # inlier_ratio:
    # Good Match 중 Inlier의 비율
    #
    # min:
    # 가장 작은 값
    #
    # mean:
    # 평균값
    #
    # max:
    # 가장 큰 값
    #
    # 왜 필요한가요?
    # 정상과 불량의 측정값이 어느 정도 다르고
    # 서로 겹치는 범위가 있는지 확인하기 위해서입니다.
    #
    # 주의:
    # 여기서는 네 가지 측정값의 통계를 확인하지만,
    # 이후 Rule 후보 생성에는 SIFT와 Inlier Ratio를 사용합니다.
    # --------------------------------------------------

    numeric_columns = [
        "template_score",
        "sift_good_matches",
        "homography_inliers",
        "inlier_ratio",
    ]


    print()

    print("=" * 70)
    print("OK / NG 측정값 통계")
    print("=" * 70)


    statistics = (
        df.groupby("actual")[
            numeric_columns
        ]
        .agg(
            [
                "min",
                "mean",
                "max",
            ]
        )
    )


    print(
        statistics
    )


    # --------------------------------------------------
    # 6. SIFT와 Inlier Ratio의 Threshold 후보 생성
    #
    # 무엇을 하나요?
    # - 2번에서 만든 make_candidates()를 호출합니다.
    # - SIFT Good Match의 Threshold 후보를 생성합니다.
    # - Inlier Ratio의 Threshold 후보를 생성합니다.
    #
    # SIFT:
    # 특징점 개수이므로 정수 후보를 사용합니다.
    #
    # Inlier Ratio:
    # 비율이므로 소수 후보를 사용합니다.
    #
    # 예:
    # SIFT >= 129
    # Inlier Ratio >= 0.8664
    #
    # 왜 필요한가요?
    # 여러 기준값을 조합하여
    # 실제 OK와 NG를 잘 구분하는 Rule을 찾기 위해서입니다.
    # --------------------------------------------------

    sift_candidates = make_candidates(
        df["sift_good_matches"],
        count=15,
        integer=True,
    )


    inlier_candidates = make_candidates(
        df["inlier_ratio"],
        count=15,
        integer=False,
    )


    print()

    print(
        "SIFT Threshold 후보:"
    )

    print(
        sift_candidates
    )


    print()

    print(
        "Inlier Ratio Threshold 후보:"
    )

    print(
        inlier_candidates
    )


    # --------------------------------------------------
    # 7. SIFT와 Inlier Ratio의 모든 Threshold 조합 시험
    #
    # 무엇을 하나요?
    # - SIFT 후보 하나를 선택합니다.
    # - Inlier Ratio 후보 하나를 선택합니다.
    # - 두 값을 조합하여 임시 Rule을 만듭니다.
    # - 현재 Rule을 Calibration 전체 이미지에 적용합니다.
    #
    # 임시 Rule:
    #
    # Homography 계산 성공
    #        AND
    # SIFT Good Match >= 후보값
    #        AND
    # Inlier Ratio >= 후보값
    #
    # 모두 만족하면 OK
    # 하나라도 만족하지 못하면 NG
    #
    # 왜 필요한가요?
    # 여러 Threshold 조합 중
    # 어떤 조합이 실제 정상과 불량을
    # 가장 적게 잘못 판정하는지 확인하기 위해서입니다.
    #
    # 후보가 각각 15개라면:
    # 15 × 15 = 225개 조합을 시험합니다.
    #
    # 단, 중복 후보가 제거되면 실제 조합 수는 줄어듭니다.
    # --------------------------------------------------

    rows = []


    for sift_min in sift_candidates:

        for inlier_min in inlier_candidates:

            # 현재 후보 Rule의 평가값 초기화
            tp = 0
            tn = 0
            fp = 0
            fn = 0


            # --------------------------------------------------
            # 7-1. Calibration 이미지마다 임시 Rule 적용
            #
            # 무엇을 하나요?
            # - CSV의 각 이미지 측정값을 하나씩 확인합니다.
            # - 현재 Threshold 조합을 적용합니다.
            # - 조건을 모두 만족하면 OK로 판정합니다.
            # - 하나라도 실패하면 NG로 판정합니다.
            #
            # 여기서는 이미 저장한 측정값을 사용합니다.
            # 원본 이미지를 다시 읽거나 SIFT를 재실행하지 않습니다.
            # --------------------------------------------------

            for _, row in df.iterrows():

                predicted = (
                    "OK"

                    if (
                        int(
                            row["homography_found"]
                        )
                        == 1

                        and int(
                            row["sift_good_matches"]
                        )
                        >= sift_min

                        and float(
                            row["inlier_ratio"]
                        )
                        >= inlier_min
                    )

                    else "NG"
                )


                # --------------------------------------------------
                # 7-2. 실제 정답과 임시 판정을 비교
                #
                # 무엇을 하나요?
                # - classify_error() 함수를 호출합니다.
                # - 현재 이미지의 결과를 TP/TN/FP/FN으로 구분합니다.
                # - 해당 오류 유형의 개수를 1 증가시킵니다.
                #
                # 특히 FN은 실제 불량을 정상으로 통과시키는
                # 오류이므로 이번 제조검사에서 중요하게 확인합니다.
                # --------------------------------------------------

                error = classify_error(
                    row["actual"],
                    predicted,
                )


                if error == "TP":
                    tp += 1

                elif error == "TN":
                    tn += 1

                elif error == "FP":
                    fp += 1

                elif error == "FN":
                    fn += 1


            # --------------------------------------------------
            # 8. 현재 Threshold 조합의 Accuracy와 오류 기록
            #
            # 무엇을 하나요?
            # - TP/TN/FP/FN을 합하여 평가 이미지 수를 계산합니다.
            # - 올바르게 판정한 비율인 Accuracy를 계산합니다.
            # - 현재 Threshold 후보의 평가 결과를 저장합니다.
            #
            # Accuracy:
            # (TP + TN) / (TP + TN + FP + FN)
            #
            # total_error:
            # FP + FN
            #
            # 예:
            # 12장 중 11장을 맞혔다면
            # Accuracy = 11 / 12 ≈ 0.9167
            #
            # rows에는 모든 Threshold 조합의 결과가 저장됩니다.
            # --------------------------------------------------

            total = (
                tp
                + tn
                + fp
                + fn
            )


            accuracy = (
                (tp + tn) / total
                if total > 0
                else 0.0
            )


            rows.append(
                {
                    "sift_good_matches_min":
                        sift_min,

                    "inlier_ratio_min":
                        inlier_min,

                    "accuracy":
                        round(
                            accuracy,
                            4,
                        ),

                    "tp":
                        tp,

                    "tn":
                        tn,

                    "fp":
                        fp,

                    "fn":
                        fn,

                    "total_error":
                        fp + fn,
                }
            )


    # --------------------------------------------------
    # 9. Threshold 후보를 오류가 적은 순서로 정렬
    #
    # 무엇을 하나요?
    # - 후보별 평가 결과를 DataFrame으로 변환합니다.
    # - 정해진 우선순위에 따라 후보를 정렬합니다.
    #
    # 정렬 우선순위:
    #
    # ① fn
    #    불량을 정상으로 놓친 개수가 적은 후보
    #
    # ② total_error
    #    전체 오류(FP + FN)가 적은 후보
    #
    # ③ fp
    #    정상을 불량으로 잘못 판정한 개수가 적은 후보
    #
    # ④ sift_good_matches_min
    #    앞 조건까지 같으면 SIFT 기준값이 작은 후보
    #
    # ⑤ inlier_ratio_min
    #    앞 조건까지 같으면 Inlier Ratio 기준값이 작은 후보
    #
    # 왜 필요한가요?
    # 제조검사에서는 불량을 정상으로 통과시키는 FN을
    # 특히 중요하게 보기 때문입니다.
    #
    # 주의:
    # 현재 정렬 방식은 Accuracy 최대값보다
    # FN 최소화를 먼저 우선합니다.
    # --------------------------------------------------

    result_df = pd.DataFrame(
        rows
    )


    result_df = (
        result_df.sort_values(
            [
                "fn",
                "total_error",
                "fp",
                "sift_good_matches_min",
                "inlier_ratio_min",
            ],
            ascending=[
                True,
                True,
                True,
                True,
                True,
            ],
        )
        .reset_index(drop=True)
    )


    # --------------------------------------------------
    # 10. Threshold 후보 평가 결과를 CSV로 저장
    #
    # 무엇을 하나요?
    # - 결과를 저장할 폴더를 생성합니다.
    # - 정렬된 모든 후보 Rule의 성능을 CSV로 저장합니다.
    # - 상위 15개 후보를 터미널에 출력합니다.
    #
    # 저장 파일:
    # reports/day05_rule_candidates.csv
    #
    # CSV에는 다음 값이 포함됩니다.
    #
    # SIFT Threshold
    # Inlier Ratio Threshold
    # Accuracy
    # TP / TN / FP / FN
    # Total Error
    #
    # 왜 필요한가요?
    # 여러 후보 Rule의 평가 결과를 나중에 다시
    # 확인하고 비교하기 위해서입니다.
    # --------------------------------------------------

    CANDIDATE_CSV.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    result_df.to_csv(
        CANDIDATE_CSV,
        index=False,
        encoding="utf-8-sig",
    )


    print()

    print("=" * 70)
    print("후보 Rule 상위 15개")
    print("=" * 70)


    print(
        result_df
        .head(15)
        .to_string(
            index=False
        )
    )


    print()

    print(
        "저장:",
        CANDIDATE_CSV
    )


    # --------------------------------------------------
    # 11. 정렬된 결과에서 첫 번째 Rule 후보 선택
    #
    # 무엇을 하나요?
    # - 정렬된 후보 중 첫 번째 행을 가져옵니다.
    # - 해당 후보의 SIFT Threshold를 확인합니다.
    # - Inlier Ratio Threshold를 확인합니다.
    # - Accuracy, FP, FN을 출력합니다.
    #
    # best:
    # 현재 정렬 기준에서 가장 위에 있는 후보
    #
    # best_sift:
    # 선택된 SIFT Good Match 기준값
    #
    # best_inlier:
    # 선택된 Inlier Ratio 기준값
    #
    # 왜 필요한가요?
    # 다음 실습에서 실제 사용할 Baseline Rule을
    # 결정할 때 검토할 후보를 확인하기 위해서입니다.
    #
    # 주의:
    # 이 후보는 Calibration 데이터에서 선택한 것입니다.
    # 새로운 Test 데이터에서의 성능까지 보장하지는 않습니다.
    # --------------------------------------------------

    best = result_df.iloc[0]


    best_sift = int(
        best["sift_good_matches_min"]
    )

    best_inlier = float(
        best["inlier_ratio_min"]
    )


    print()

    print("=" * 70)
    print("현재 가장 좋은 후보")
    print("=" * 70)

    print(
        "SIFT Good Match >=",
        best_sift
    )

    print(
        "Inlier Ratio >=",
        best_inlier
    )

    print(
        "Accuracy:",
        best["accuracy"]
    )

    print(
        "FP:",
        int(best["fp"])
    )

    print(
        "FN:",
        int(best["fn"])
    )


    # --------------------------------------------------
    # 12. SIFT와 Inlier Ratio의 OK/NG 분포 그래프 생성
    #
    # 무엇을 하나요?
    # - Calibration 이미지의 측정값을 X-Y 그래프로 표시합니다.
    # - 실제 OK와 NG를 서로 다른 모양으로 표시합니다.
    # - 선택된 Threshold를 세로선과 가로선으로 표시합니다.
    #
    # X축:
    # SIFT Good Match
    #
    # Y축:
    # Homography Inlier Ratio
    #
    # 원형(o):
    # 실제 OK 이미지
    #
    # X 모양(x):
    # 실제 NG 이미지
    #
    # 세로 점선:
    # SIFT Threshold
    #
    # 가로 점선:
    # Inlier Ratio Threshold
    #
    # 왜 필요한가요?
    # 정상과 불량의 측정값이
    # Threshold를 기준으로 어떻게 분포하는지
    # 눈으로 확인하기 위해서입니다.
    #
    # 현재 Rule에서는 두 기준을 모두 만족하는
    # 오른쪽 위 영역이 OK 판정 영역입니다.
    # --------------------------------------------------

    PLOT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    # 그래프 크기 설정
    plt.figure(
        figsize=(9, 6)
    )


    # --------------------------------------------------
    # 12-1. 실제 OK와 NG 측정값을 점으로 표시
    #
    # 무엇을 하나요?
    # - 실제 정답을 기준으로 OK와 NG를 분리합니다.
    # - SIFT 값을 X축에 표시합니다.
    # - Inlier Ratio를 Y축에 표시합니다.
    # - 각 점에 이미지 파일명을 표시합니다.
    #
    # plt.scatter():
    # 데이터의 위치를 점으로 표시하는 함수
    #
    # plt.annotate():
    # 점 옆에 설명이나 파일명을 표시하는 함수
    # --------------------------------------------------

    for actual, marker in [
        ("OK", "o"),
        ("NG", "x"),
    ]:

        subset = df[
            df["actual"] == actual
        ]


        plt.scatter(
            subset["sift_good_matches"],
            subset["inlier_ratio"],
            label=actual,
            marker=marker,
        )


        # 각 점에 이미지 파일명 표시
        for _, row in subset.iterrows():

            plt.annotate(
                row["file"],
                (
                    row["sift_good_matches"],
                    row["inlier_ratio"],
                ),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=8,
            )


    # --------------------------------------------------
    # 12-2. 선택된 SIFT Threshold를 세로선으로 표시
    #
    # plt.axvline():
    # X축의 특정 위치에 세로선을 그립니다.
    #
    # 예:
    # SIFT >= 129
    #
    # 129 이상인 점은 선의 오른쪽에 위치합니다.
    # --------------------------------------------------

    plt.axvline(
        best_sift,
        linestyle="--",
        label=(
            f"SIFT >= {best_sift}"
        ),
    )


    # --------------------------------------------------
    # 12-3. 선택된 Inlier Ratio Threshold를 가로선으로 표시
    #
    # plt.axhline():
    # Y축의 특정 위치에 가로선을 그립니다.
    #
    # 예:
    # Inlier Ratio >= 0.8664
    #
    # 0.8664 이상인 점은 선의 위쪽에 위치합니다.
    # --------------------------------------------------

    plt.axhline(
        best_inlier,
        linestyle="--",
        label=(
            f"Inlier >= {best_inlier:.4f}"
        ),
    )


    # --------------------------------------------------
    # 12-4. 그래프 제목과 축 이름 설정
    #
    # xlabel:
    # X축 이름
    #
    # ylabel:
    # Y축 이름
    #
    # title:
    # 그래프 제목
    #
    # legend:
    # OK, NG, Threshold 선의 범례 표시
    #
    # grid:
    # 측정값의 위치를 쉽게 확인할 수 있도록
    # 보조 격자선을 표시합니다.
    # --------------------------------------------------

    plt.xlabel(
        "SIFT Good Matches"
    )

    plt.ylabel(
        "Homography Inlier Ratio"
    )

    plt.title(
        "Day05 Calibration Distribution"
    )


    plt.legend()

    plt.grid(
        alpha=0.2
    )


    # --------------------------------------------------
    # 12-5. 분포 그래프를 PNG 파일로 저장
    #
    # plt.savefig():
    # 현재 그래프를 이미지 파일로 저장합니다.
    #
    # dpi=150:
    # 출력 이미지 해상도 설정
    #
    # bbox_inches="tight":
    # 그래프 주변의 불필요한 여백을 줄입니다.
    #
    # plt.close():
    # 그래프 작업이 끝난 후 리소스를 정리합니다.
    #
    # 저장 파일:
    # sift_inlier_distribution.png
    # --------------------------------------------------

    plt.savefig(
        PLOT_PATH,
        dpi=150,
        bbox_inches="tight",
    )


    plt.close()


    print()

    print(
        "분포 이미지:",
        PLOT_PATH
    )


# --------------------------------------------------
# 13. Python 파일을 직접 실행할 때 main() 호출
#
# 무엇을 하나요?
# - 현재 파일을 직접 실행했는지 확인합니다.
# - 직접 실행했다면 main() 함수를 호출합니다.
#
# 왜 필요한가요?
# 다른 Python 파일에서 이 코드를 import할 때
# Calibration 분석이 자동 실행되지 않도록 하기 위해서입니다.
#
# 실행 명령어:
# python src/35_day05_analyze_calibration.py
# --------------------------------------------------

if __name__ == "__main__":
    main()