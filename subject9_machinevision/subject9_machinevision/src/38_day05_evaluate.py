from pathlib import Path
import shutil
import json

import pandas as pd


# --------------------------------------------------
# 1. Test 결과 CSV와 Baseline Rule JSON 경로 설정
#
# 무엇을 하나요?
# - 앞에서 실행한 Test 결과 CSV의 경로를 정합니다.
# - Test에 사용했던 Baseline Rule JSON 경로를 정합니다.
#
# RESULT_CSV:
# 이미지별 실제 정답, 프로그램 판정,
# TP/TN/FP/FN 등이 기록된 파일입니다.
#
# RULE_PATH:
# Calibration에서 확정하고 동결한
# Baseline Rule 정보가 저장된 JSON 파일입니다.
#
# 왜 필요한가요?
# Test 결과를 평가하고
# 어떤 버전의 Rule을 사용했는지 기록하기 위해서입니다.
# --------------------------------------------------

RESULT_CSV = Path(
    "reports/day05_test_results.csv"
)

RULE_PATH = Path(
    "configs/day05_logo_rules.json"
)


# --------------------------------------------------
# 2. 평가 보고서 및 FP/FN 이미지 저장 경로 설정
#
# 무엇을 하나요?
# - 성능 평가 보고서를 저장할 경로를 정합니다.
# - FP 이미지를 모아 둘 폴더를 지정합니다.
# - FN 이미지를 모아 둘 폴더를 지정합니다.
#
# REPORT_PATH:
# Accuracy, Precision, Recall 등을 기록할
# Markdown 보고서 파일입니다.
#
# FP_DIR:
# 실제 OK인데 NG로 잘못 판정한 이미지 저장 폴더
#
# FN_DIR:
# 실제 NG인데 OK로 잘못 통과한 이미지 저장 폴더
#
# 왜 필요한가요?
# 잘못 판정한 이미지만 따로 모아서
# 어떤 원인으로 오류가 발생했는지 분석하기 위해서입니다.
# --------------------------------------------------

REPORT_PATH = Path(
    "reports/day05_test_metrics.md"
)

FP_DIR = Path(
    "outputs/day05/fp"
)

FN_DIR = Path(
    "outputs/day05/fn"
)


# --------------------------------------------------
# 3. 0으로 나누는 오류를 방지하는 공통 함수
#
# 무엇을 하나요?
# - 두 숫자를 전달받아 나눗셈을 수행합니다.
# - 분모가 0이면 0.0을 반환합니다.
#
# a:
# 분자
#
# b:
# 분모
#
# 예:
# safe_divide(3, 6) → 0.5
# safe_divide(3, 0) → 0.0
#
# 왜 필요한가요?
# Accuracy, Precision, Recall 계산 중
# 분모가 0이 되면 ZeroDivisionError가 발생하기 때문입니다.
#
# 주의:
# 분모가 0인 경우의 0.0은 계산을 위한 기본값입니다.
# 해당 지표가 실제로 0%라는 의미와는 구분해야 합니다.
# --------------------------------------------------

def safe_divide(a, b):

    if b == 0:
        return 0.0

    return a / b


# --------------------------------------------------
# 4. Test 결과 CSV 읽기 및 필수 컬럼 확인
#
# 무엇을 하나요?
# - main() 함수를 정의합니다.
# - Test 결과 CSV가 존재하는지 확인합니다.
# - Pandas로 CSV를 읽습니다.
# - 평가에 필요한 컬럼이 있는지 검사합니다.
#
# 필수 컬럼:
#
# file:
# 검사 이미지 이름
#
# actual:
# 실제 정답인 OK 또는 NG
#
# predicted:
# 프로그램이 판정한 OK 또는 NG
#
# error_type:
# TP / TN / FP / FN
#
# result_image:
# 검사 결과 이미지 저장 경로
#
# 왜 필요한가요?
# 앞 단계에서 만들어진 Test 결과가 있어야
# 전체 성능을 평가할 수 있기 때문입니다.
#
# 필수 컬럼이 없다면 프로그램을 중단합니다.
# --------------------------------------------------

def main():

    # Test 결과 CSV 존재 여부 확인
    if not RESULT_CSV.exists():

        raise FileNotFoundError(
            "Test 결과 CSV가 없습니다. "
            "먼저 37_day05_test.py를 실행하세요."
        )


    # CSV를 Pandas DataFrame으로 읽기
    df = pd.read_csv(
        RESULT_CSV
    )


    # 평가에 필요한 컬럼 목록
    required_columns = {
        "file",
        "actual",
        "predicted",
        "error_type",
        "result_image",
    }


    # CSV에서 누락된 컬럼 찾기
    missing = (
        required_columns
        - set(df.columns)
    )


    # 필수 컬럼이 없으면 오류 발생
    if missing:

        raise RuntimeError(
            f"필요한 컬럼이 없습니다: {missing}"
        )


    # --------------------------------------------------
    # 5. Test에 사용한 Baseline Rule 버전 확인
    #
    # 무엇을 하나요?
    # - Rule 버전의 기본값을 UNKNOWN으로 설정합니다.
    # - JSON 파일이 있으면 내용을 읽습니다.
    # - baseline_version 값을 가져옵니다.
    #
    # 예:
    # day05-logo-v1.0
    #
    # 왜 필요한가요?
    # 어떤 Rule을 사용하여 Test했는지
    # 평가 보고서에 기록하기 위해서입니다.
    #
    # 주의:
    # JSON이 없으면 UNKNOWN으로 기록합니다.
    #
    # 또한 이 코드는 현재 JSON의 버전명을 읽는 것이므로
    # CSV를 생성할 당시 사용한 버전과 동일한지까지
    # 자동으로 검증하지는 않습니다.
    # --------------------------------------------------

    baseline_version = "UNKNOWN"


    if RULE_PATH.exists():

        with RULE_PATH.open(
            "r",
            encoding="utf-8"
        ) as file:

            rules = json.load(
                file
            )


        baseline_version = rules.get(
            "baseline_version",
            "UNKNOWN"
        )


    # --------------------------------------------------
    # 6. Test 결과에서 TP / TN / FP / FN 개수 집계
    #
    # 무엇을 하나요?
    # - CSV의 error_type 컬럼을 확인합니다.
    # - TP, TN, FP, FN이 각각 몇 개인지 계산합니다.
    #
    # 이번 제조검사에서는 NG를 Positive로 정의합니다.
    #
    # TP (True Positive):
    # 실제 NG → 프로그램 NG
    # 불량을 올바르게 검출한 경우
    #
    # TN (True Negative):
    # 실제 OK → 프로그램 OK
    # 정상을 올바르게 통과시킨 경우
    #
    # FP (False Positive):
    # 실제 OK → 프로그램 NG
    # 정상을 불량으로 잘못 판정한 경우
    #
    # FN (False Negative):
    # 실제 NG → 프로그램 OK
    # 불량을 정상으로 잘못 통과시킨 경우
    #
    # value_counts():
    # 각 error_type이 나타난 횟수를 계산합니다.
    #
    # counts.get("TP", 0):
    # TP가 없으면 0을 반환합니다.
    #
    # 왜 필요한가요?
    # 전체 정확도뿐 아니라
    # 어떤 종류의 판정 오류가 발생했는지
    # 확인하기 위해서입니다.
    # --------------------------------------------------

    counts = (
        df["error_type"]
        .value_counts()
        .to_dict()
    )


    # 불량을 올바르게 검출한 개수
    tp = int(
        counts.get("TP", 0)
    )


    # 정상을 올바르게 통과한 개수
    tn = int(
        counts.get("TN", 0)
    )


    # 정상을 NG로 잘못 판정한 개수
    fp = int(
        counts.get("FP", 0)
    )


    # 불량을 OK로 잘못 통과시킨 개수
    fn = int(
        counts.get("FN", 0)
    )


    # TP / TN / FP / FN으로 분류된 전체 이미지 수
    total = (
        tp
        + tn
        + fp
        + fn
    )


    # --------------------------------------------------
    # 7. Accuracy / NG Precision / NG Recall 계산
    #
    # 무엇을 하나요?
    # - 전체 판정 정확도를 계산합니다.
    # - NG라고 판정한 결과의 정밀도를 계산합니다.
    # - 실제 NG를 얼마나 검출했는지 계산합니다.
    #
    # ① Accuracy
    #
    # 전체 이미지 중 올바르게 판정한 비율
    #
    # Accuracy = (TP + TN) / 전체 이미지 수
    #
    # ② NG Precision
    #
    # 프로그램이 NG라고 판정한 이미지 중
    # 실제 NG가 차지하는 비율
    #
    # Precision = TP / (TP + FP)
    #
    # ③ NG Recall
    #
    # 실제 NG 이미지 중
    # 프로그램이 NG로 검출한 비율
    #
    # Recall = TP / (TP + FN)
    #
    # 왜 필요한가요?
    # Accuracy 하나만으로는
    # 정상 과검출과 불량 미검출을
    # 충분히 구분하기 어렵기 때문입니다.
    #
    # 제조검사에서는 특히
    # 실제 불량을 OK로 통과시키는 FN을
    # 중요하게 확인합니다.
    # --------------------------------------------------

    # 전체 정확도
    accuracy = safe_divide(
        tp + tn,
        total
    )


    # NG라고 판정한 것 중 실제 NG의 비율
    # NG를 Positive로 정의합니다.
    precision_ng = safe_divide(
        tp,
        tp + fp
    )


    # 실제 NG 중 올바르게 검출한 비율
    recall_ng = safe_divide(
        tp,
        tp + fn
    )


    # --------------------------------------------------
    # 8. 기존 FP / FN 이미지 폴더 초기화
    #
    # 무엇을 하나요?
    # - 이전 실행에서 저장한 FP/FN 폴더를 확인합니다.
    # - 폴더가 있다면 삭제합니다.
    # - 새로운 빈 FP/FN 폴더를 생성합니다.
    #
    # shutil.rmtree():
    # 폴더와 그 안의 파일을 모두 삭제하는 함수
    #
    # mkdir():
    # 폴더를 만드는 함수
    #
    # 왜 필요한가요?
    # 이전 Test에서 만들어진 실패 이미지가
    # 현재 Test 결과에 섞이지 않도록 하기 위해서입니다.
    #
    # 주의:
    # FP_DIR과 FN_DIR 안의 기존 파일은
    # 모두 삭제되므로 별도 자료를 보관하면 안 됩니다.
    # --------------------------------------------------

    for folder in [
        FP_DIR,
        FN_DIR,
    ]:

        # 기존 폴더가 있다면 삭제
        if folder.exists():

            shutil.rmtree(
                folder
            )


        # 새로운 빈 폴더 생성
        folder.mkdir(
            parents=True,
            exist_ok=True
        )


    # --------------------------------------------------
    # 9. FP / FN으로 잘못 판정한 결과 이미지를 분리
    #
    # 무엇을 하나요?
    # - Test 결과 CSV의 각 행을 확인합니다.
    # - 각 이미지의 error_type을 읽습니다.
    # - 결과 이미지 경로를 확인합니다.
    #
    # FP인 경우:
    # outputs/day05/fp/ 폴더에 복사
    #
    # FN인 경우:
    # outputs/day05/fn/ 폴더에 복사
    #
    # shutil.copy2():
    # 원본 파일을 보존하면서 파일을 복사합니다.
    # 가능한 파일 메타데이터도 함께 보존합니다.
    #
    # 왜 필요한가요?
    # 정상인데 NG로 잘못 판정한 이미지와
    # 불량인데 OK로 잘못 통과한 이미지를
    # 구분해서 분석하기 위해서입니다.
    #
    # 주의:
    # 결과 이미지 파일이 없으면
    # 현재 코드는 해당 행을 건너뜁니다.
    # 따라서 오류 개수와 복사된 이미지 수가
    # 다를 수 있습니다.
    # --------------------------------------------------

    for _, row in df.iterrows():

        # 현재 이미지의 오류 유형
        error_type = row[
            "error_type"
        ]


        # 검사 결과 이미지 경로
        source = Path(
            row["result_image"]
        )


        # 이미지가 없으면 건너뛰기
        if not source.exists():
            continue


        # FP: 실제 OK → 프로그램 NG
        if error_type == "FP":

            shutil.copy2(
                source,
                FP_DIR / source.name
            )


        # FN: 실제 NG → 프로그램 OK
        elif error_type == "FN":

            shutil.copy2(
                source,
                FN_DIR / source.name
            )


    # --------------------------------------------------
    # 10. 전체 Test 성능을 Markdown 보고서로 저장
    #
    # 무엇을 하나요?
    # - 보고서를 저장할 폴더를 준비합니다.
    # - Baseline Rule 버전을 기록합니다.
    # - TP/TN/FP/FN을 Confusion Matrix로 정리합니다.
    # - Accuracy, Precision, Recall을 기록합니다.
    # - FP와 FN의 의미를 설명합니다.
    #
    # Confusion Matrix:
    # 실제 정답과 프로그램 판정을 비교하는 표입니다.
    #
    # 보고서 파일:
    # reports/day05_test_metrics.md
    #
    # 왜 필요한가요?
    # Test 결과를 문서로 남겨
    # 이후 Baseline 성능과 실패 원인을
    # 확인할 수 있도록 하기 위해서입니다.
    #
    # 주의:
    # 이 보고서는 집계된 평가 결과를 기록합니다.
    # 실제 Failure Cause를 자동으로 분석하지는 않습니다.
    # --------------------------------------------------

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )


    # Markdown 형식의 보고서 내용 작성
    report = f"""# Day05 Test Metrics

## Baseline

- Rule Version: {baseline_version}
- Positive Class: NG
- Test Images: {total}

## Confusion Matrix

| Actual / Predicted | NG | OK |
|---|---:|---:|
| NG | TP = {tp} | FN = {fn} |
| OK | FP = {fp} | TN = {tn} |

## Metrics

- Accuracy: {accuracy:.4f}
- NG Precision: {precision_ng:.4f}
- NG Recall: {recall_ng:.4f}
- FP: {fp}
- FN: {fn}

## Interpretation

- FP: 실제 OK인데 NG로 잘못 판정
- FN: 실제 NG인데 OK로 잘못 통과
"""


    # Markdown 파일 저장
    REPORT_PATH.write_text(
        report,
        encoding="utf-8"
    )


    # --------------------------------------------------
    # 11. 전체 평가 결과와 저장 경로를 터미널에 출력
    #
    # 무엇을 하나요?
    # - 사용한 Rule 버전을 출력합니다.
    # - Test 이미지 개수를 출력합니다.
    # - TP/TN/FP/FN을 출력합니다.
    # - Accuracy, Precision, Recall을 출력합니다.
    # - 평가 보고서와 FP/FN 이미지 폴더 경로를 출력합니다.
    #
    # 왜 필요한가요?
    # 평가 결과를 파일로 열기 전에도
    # 터미널에서 전체 성능을 확인하기 위해서입니다.
    # --------------------------------------------------

    print("=" * 70)

    print(
        "Day05 Test Evaluation"
    )

    print("=" * 70)


    # 사용한 Baseline Rule 버전
    print(
        "Rule:",
        baseline_version
    )


    # 전체 평가 이미지 수
    print(
        "Test Images:",
        total
    )

    print()


    # TP / TN / FP / FN 출력
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

    print()


    # Accuracy 출력
    print(
        "Accuracy:",
        round(
            accuracy,
            4
        )
    )


    # NG Precision 출력
    print(
        "NG Precision:",
        round(
            precision_ng,
            4
        )
    )


    # NG Recall 출력
    print(
        "NG Recall:",
        round(
            recall_ng,
            4
        )
    )

    print()


    # 결과물 저장 경로 확인
    print(
        "평가 보고서:",
        REPORT_PATH
    )

    print(
        "FP 이미지:",
        FP_DIR
    )

    print(
        "FN 이미지:",
        FN_DIR
    )


# --------------------------------------------------
# 12. Python 파일을 직접 실행하면 main() 호출
#
# 무엇을 하나요?
# - 현재 파일이 직접 실행되었는지 확인합니다.
# - 직접 실행했다면 main() 함수를 호출합니다.
#
# 왜 필요한가요?
# 다른 파일에서 이 코드를 import할 때
# 평가 작업이 자동 실행되지 않도록 하기 위해서입니다.
#
# 실행 명령어:
# python src/38_day05_evaluate.py
# --------------------------------------------------

if __name__ == "__main__":
    main()