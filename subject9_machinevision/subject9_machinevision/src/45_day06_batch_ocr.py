# --------------------------------------------------
# 1. Batch OCR에 필요한 라이브러리와 공통 함수 불러오기
#
# 무엇을 하나요?
# - CSV 처리와 이미지 처리에 필요한 라이브러리를 불러옵니다.
# - 앞에서 만든 OCR 공통 함수와 문자열 검사 함수를 가져옵니다.
#
# csv:
# Ground Truth CSV를 읽고 검사 결과 CSV를 저장합니다.
#
# cv2:
# 제품 이미지 읽기, 확대 및 결과 이미지 저장에 사용합니다.
#
# day06_ocr_utils.py:
# JSON Rule 읽기, ROI Crop, 전처리, Tesseract OCR 담당
#
# day06_ocr_validation.py:
# 문자열 Cleaning, 형식 검사, Expected SN 비교,
# 최종 OK/NG 판정 담당
#
# 왜 필요한가요?
# 앞에서 작성한 기능들을 재사용하여
# 하나의 Batch 검사 프로그램으로 연결하기 위해서입니다.
# --------------------------------------------------

import csv

import cv2

from day06_ocr_utils import (
    PROJECT_ROOT,
    crop_roi,
    get_roi_for_file,
    load_rules,
    preprocess_for_ocr,
    run_ocr,
)

from day06_ocr_validation import (
    inspect_text,
)


# --------------------------------------------------
# 2. 제품 이미지, 정답 데이터 및 결과 저장 경로 설정
#
# 무엇을 하나요?
# - 검사할 제품 이미지 폴더를 지정합니다.
# - 실제 SN과 Expected SN이 저장된 CSV를 지정합니다.
# - OCR 입력 이미지를 저장할 폴더를 지정합니다.
# - 최종 검사 결과 CSV 경로를 지정합니다.
#
# PRODUCT_DIR:
# 실제 제품 이미지 6장이 있는 폴더
#
# LABELS_PATH:
# 제품별 정답 정보를 기록한 CSV
#
# 주요 컬럼:
# file          → 이미지 파일명
# actual_text   → 실제로 인쇄된 SN
# expected_text → 생산상 인쇄되어야 하는 SN
# actual_status → 실제 제품 정답 OK/NG
#
# OUTPUT_DIR:
# Tesseract에 입력할 확대 이미지를 저장할 폴더
#
# REPORT_PATH:
# 전체 제품의 OCR 및 검사 결과를 기록할 CSV
#
# 왜 필요한가요?
# 어떤 데이터를 검사하고
# 검사 결과를 어디에 저장할지 지정하기 위해서입니다.
# --------------------------------------------------

PRODUCT_DIR = (
    PROJECT_ROOT
    / "samples"
    / "day06"
    / "product"
)

LABELS_PATH = (
    PROJECT_ROOT
    / "configs"
    / "day06_labels.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "day06"
    / "batch"
)

REPORT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "day06_ocr_results.csv"
)


# --------------------------------------------------
# 3. JSON에 저장된 OCR Rule 읽기
#
# 무엇을 하나요?
# - load_rules() 공통 함수를 호출합니다.
# - day06_ocr_rules.json을 읽습니다.
# - JSON에 저장된 OCR 설정을 가져옵니다.
#
# 주요 설정:
#
# baseline_version:
# OCR Rule 버전
#
# roi_by_file:
# 이미지별 SN ROI 좌표
#
# preprocess_mode:
# 사용할 전처리 방법
#
# ocr_psm:
# Tesseract 문자 배치 설정
#
# whitelist:
# OCR에서 허용할 문자 목록
#
# expected_pattern:
# 13자리 SN 형식 검사 정규식
#
# 왜 필요한가요?
# 앞에서 확정한 동일한 OCR Rule을
# 모든 제품 이미지에 적용하기 위해서입니다.
# --------------------------------------------------

rules = load_rules()


# --------------------------------------------------
# 4. OCR Rule이 v1.0으로 확정되었는지 확인
#
# 무엇을 하나요?
# - JSON에 기록된 baseline_version을 확인합니다.
# - day06-ocr-v1.0이 아니면 프로그램을 중단합니다.
#
# 이번 실습에서 사용할 Rule:
#
# Version = day06-ocr-v1.0
# Preprocess = original
# PSM = 7
# Whitelist = 0123456789
# Regex = ^[0-9]{13}$
#
# 왜 필요한가요?
# 실험용 Draft 설정으로 Batch 검사를 하지 않고
# 앞에서 확정한 Baseline을 사용하기 위해서입니다.
#
# 주의:
# 이 코드는 버전 문자열을 확인합니다.
# JSON 내부 설정이 실제로 변경되지 않았는지까지
# 자동으로 검증하는 것은 아닙니다.
# --------------------------------------------------

if rules[
    "baseline_version"
] != "day06-ocr-draft":
##day06-ocr-v1.0
    raise RuntimeError(
        "OCR Rule이 Freeze되지 않았습니다."
    )


# --------------------------------------------------
# 5. 제품별 정답 데이터 CSV 존재 여부 확인
#
# 무엇을 하나요?
# - day06_labels.csv 파일이 있는지 확인합니다.
# - 파일이 없다면 오류를 발생시킵니다.
#
# LABELS_PATH.exists():
# 지정한 파일이나 경로가 존재하는지 확인합니다.
#
# FileNotFoundError:
# 필요한 파일을 찾을 수 없을 때 발생시키는 오류입니다.
#
# 왜 필요한가요?
# OCR 결과와 실제 인쇄값을 비교하고
# 제품의 최종 판정이 올바른지 평가하려면
# 제품별 정답 데이터가 필요하기 때문입니다.
#
# 중요:
# actual_text와 actual_status는
# 이번 교육용 실습에서 정확도를 평가하기 위한 정답입니다.
#
# 실제 OCR 검사에서는 이런 정답을
# 미리 알고 있다고 가정하지 않습니다.
# --------------------------------------------------

if not LABELS_PATH.exists():

    raise FileNotFoundError(
        "Batch 검사에 필요한 "
        "day06_labels.csv가 없습니다.\n"
        f"확인 경로: {LABELS_PATH}"
    )


# --------------------------------------------------
# 6. OCR 입력 이미지를 저장할 결과 폴더 생성
#
# 무엇을 하나요?
# - OCR 입력 이미지 저장 폴더가 있는지 확인합니다.
# - 폴더가 없으면 생성합니다.
#
# OUTPUT_DIR.mkdir():
# 결과 폴더를 생성하는 함수
#
# parents=True:
# 필요한 상위 폴더까지 함께 생성합니다.
#
# exist_ok=True:
# 폴더가 이미 있으면 그대로 사용합니다.
#
# 저장 위치:
# outputs/day06/batch/
#
# 왜 필요한가요?
# OCR 실패가 발생했을 때
# 실제 Tesseract에 전달한 이미지를 확인하기 위해서입니다.
# --------------------------------------------------

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# --------------------------------------------------
# 7. Ground Truth와 Expected SN 정보를 CSV에서 읽기
#
# 무엇을 하나요?
# - day06_labels.csv를 엽니다.
# - CSV 내용을 딕셔너리 목록으로 읽습니다.
# - 검사할 데이터가 있는지 확인합니다.
# - 검사 결과를 저장할 빈 리스트를 만듭니다.
#
# csv.DictReader():
# CSV의 컬럼명을 Key로 하여 각 행을 읽습니다.
#
# encoding="utf-8-sig":
# UTF-8 BOM이 포함된 CSV도 읽을 수 있도록 합니다.
#
# labels:
# 제품별 정답 정보가 저장된 리스트
#
# rows:
# OCR 검사 결과를 저장할 빈 리스트
#
# 예:
#
# file:
# 0679256889404_side.png
#
# actual_text:
# 0679256889404
#
# expected_text:
# 0679256889400
#
# actual_status:
# NG
#
# 왜 필요한가요?
# 각 제품에 맞는 Expected SN을 가져와u 
#
# 주의:
# SN의 맨 앞 0이 사라지지 않도록
# 문자열로 읽어야 합니다.
# --------------------------------------------------

with LABELS_PATH.open(
    "r",
    newline="",
    encoding="utf-8-sig",
) as file:

    labels = list(
        csv.DictReader(file)
    )


# 검사 데이터가 비어 있으면 오류 발생
if not labels:

    raise RuntimeError(
        "day06_labels.csv에 "
        "검사 데이터가 없습니다."
    )


# 이미지별 검사 결과를 저장할 리스트
rows = []


# --------------------------------------------------
# 8. CSV에 기록된 제품 이미지를 하나씩 검사
#
# 무엇을 하나요?
# - labels에서 제품 정보를 한 행씩 가져옵니다.
# - 현재 제품 이미지 파일명을 확인합니다.
# - 실제 이미지가 저장된 전체 경로를 만듭니다.
#
# for item in labels:
# CSV의 제품 정보를 하나씩 반복합니다.
#
# item["file"]:
# 현재 검사할 제품 이미지 파일명
#
# image_path:
# 현재 제품 이미지 전체 경로
#
# 왜 필요한가요?
# 대표 이미지 한 장이 아니라
# 여러 제품에 동일한 OCR 검사 과정을 적용하기 위해서입니다.
#
# 이번 실습의 데이터:
# OK 제품 4장
# NG 제품 2장
# 총 6장
#
# 주의:
# 반복 횟수는 CSV에 기록된 행 개수로 결정됩니다.
# 이 코드에서 6장인지 별도로 확인하지는 않습니다.
# --------------------------------------------------

for item in labels:

    file_name = item[
        "file"
    ]

    image_path = (
        PRODUCT_DIR
        / file_name
    )


    # --------------------------------------------------
    # 9. 현재 제품 이미지 읽기 및 오류 확인
    #
    # 무엇을 하나요?
    # - cv2.imread()로 현재 제품 이미지를 읽습니다.
    # - 이미지가 정상적으로 읽혔는지 확인합니다.
    #
    # image:
    # OpenCV로 읽은 원본 제품 이미지
    #
    # image is None:
    # 이미지 읽기에 실패했다는 의미입니다.
    #
    # 왜 필요한가요?
    # 이미지에서 SN 문자 영역을 추출하려면
    # 먼저 원본 제품 이미지를 읽어야 하기 때문입니다.
    # --------------------------------------------------

    image = cv2.imread(
        str(image_path)
    )

    if image is None:

        raise FileNotFoundError(
            f"제품 이미지를 찾을 수 없습니다: "
            f"{image_path}"
        )


    # --------------------------------------------------
    # 10. 현재 이미지에 저장된 ROI 좌표 가져오기
    #
    # 무엇을 하나요?
    # - get_roi_for_file() 공통 함수를 호출합니다.
    # - 이미지 파일명으로 저장된 ROI 좌표를 찾습니다.
    #
    # JSON의 roi_by_file:
    # 각 이미지 파일명에 대응하는 ROI 좌표 목록
    #
    # 반환 예:
    # [434, 294, 529, 318]
    #
    # 좌표 순서:
    # [x1, y1, x2, y2]
    #
    # 왜 필요한가요?
    # 제품마다 SN 위치가 다르므로
    # 현재 이미지에 맞는 좌표를 사용해야 하기 때문입니다.
    #
    # 주의:
    # 이번 코드는 41번에서 사람이 미리 지정한
    # 좌표를 재사용하는 방식입니다.
    # 문자의 위치를 자동 검출하는 방식은 아닙니다.
    # --------------------------------------------------

    roi_coordinates = (
        get_roi_for_file(
            rules,
            file_name,
        )
    )


    # --------------------------------------------------
    # 11. ROI 좌표를 이용하여 SN 숫자 영역 Crop
    #
    # 무엇을 하나요?
    # - crop_roi() 공통 함수를 호출합니다.
    # - 원본 제품 이미지에서 SN 숫자 부분만 잘라냅니다.
    #
    # 내부 처리:
    # image[y1:y2, x1:x2]
    #
    # roi:
    # 제품에 인쇄된 13자리 숫자 영역
    #
    # 왜 필요한가요?
    # Tesseract가 전체 제품의 여러 문자를 읽지 않고
    # 필요한 SN 숫자만 인식하도록 하기 위해서입니다.
    # --------------------------------------------------

    roi = crop_roi(
        image,
        roi_coordinates,
    )


    # --------------------------------------------------
    # 12. Freeze된 OCR 전처리 방법 적용
    #
    # 무엇을 하나요?
    # - preprocess_for_ocr() 함수를 호출합니다.
    # - JSON에서 선택한 전처리 방법을 적용합니다.
    #
    # 현재 확정된 설정:
    # preprocess_mode = original
    #
    # original:
    # SN ROI에 추가 Gray, CLAHE, Otsu 등을 적용하지 않고
    # 원본 ROI를 사용하는 방법입니다.
    #
    # processed:
    # OCR에 사용할 전처리 결과 이미지
    #
    # 왜 필요한가요?
    # 43번 실험에서 결정한 전처리 방법을
    # 모든 제품 이미지에 동일하게 적용하기 위해서입니다.
    #
    # 참고:
    # 현재 공통 함수는 다섯 가지 전처리를 계산한 뒤
    # 선택된 결과 하나를 반환하는 구조입니다.
    # --------------------------------------------------

    processed = (
        preprocess_for_ocr(
            roi,
            rules,
        )
    )


    # --------------------------------------------------
    # 13. OCR 입력 이미지를 가로·세로 4배 확대
    #
    # 무엇을 하나요?
    # - 전처리한 SN 이미지를 확대합니다.
    # - 가로와 세로 크기를 모두 4배로 만듭니다.
    #
    # cv2.resize():
    # 이미지의 크기를 변경하는 함수
    #
    # fx=4.0:
    # 가로 4배 확대
    #
    # fy=4.0:
    # 세로 4배 확대
    #
    # INTER_CUBIC:
    # 주변 픽셀을 이용하는 보간 방법
    #
    # 왜 필요한가요?
    # 원본 SN ROI가 작기 때문에
    # OCR이 처리할 문자 크기를 키우기 위해서입니다.
    #
    # 중요:
    # 43번 실험에서도 4배 확대했습니다.
    #
    # 같은 확대 조건을 유지해야
    # 실험에서 결정한 조건과 실제 Batch 조건을
    # 일치시킬 수 있습니다.
    #
    # 주의:
    # 확대만으로 원본에 없는 문자 정보가
    # 새로 생기는 것은 아닙니다.
    # --------------------------------------------------

    ocr_input = cv2.resize(
        processed,
        None,
        fx=4.0,
        fy=4.0,
        interpolation=cv2.INTER_CUBIC,
    )


    # --------------------------------------------------
    # 14. Tesseract OCR 실행하여 Raw 문자열 얻기
    #
    # 무엇을 하나요?
    # - run_ocr() 공통 함수를 호출합니다.
    # - 확대된 이미지와 확정된 OCR Rule을 전달합니다.
    # - Tesseract가 읽은 문자열을 가져옵니다.
    #
    # OCR 설정:
    #
    # OEM = 3
    # PSM = 7
    # Whitelist = 0123456789
    #
    # raw_text:
    # Tesseract가 반환한 원본 문자열
    #
    # 예:
    # '0637145699424\n'
    #
    # 왜 필요한가요?
    # 이미지 속 인쇄된 SN을
    # Python에서 비교할 수 있는 문자열로 변환하기 위해서입니다.
    #
    # 주의:
    # 이 단계에서는 문자를 읽기만 합니다.
    # 최종 OK/NG 판정은 다음 단계에서 수행합니다.
    # --------------------------------------------------

    raw_text = run_ocr(
        ocr_input,
        rules,
    )


    # --------------------------------------------------
    # 15. OCR 문자열 검사 및 제품 OK/NG 판정
    #
    # 무엇을 하나요?
    # - inspect_text() 공통 함수를 호출합니다.
    # - Raw OCR 문자열을 정리합니다.
    # - 숫자 13자리 형식을 검사합니다.
    # - Expected SN과 비교합니다.
    # - 최종 제품을 OK 또는 NG로 판정합니다.
    #
    # raw_text:
    # Tesseract가 실제로 읽은 문자열
    #
    # expected_text:
    # 해당 제품에 인쇄되어야 하는 SN
    #
    # pattern:
    # 숫자 13자리 형식을 검사하는 정규식
    #
    # 검사 순서:
    #
    # ① OCR 문자열의 공백과 줄바꿈 제거
    #
    # ② OCR 결과가 비어 있는지 확인
    #
    # ③ 숫자 13자리 형식인지 확인
    #
    # ④ Expected SN과 같은지 확인
    #
    # ⑤ 모든 조건을 만족하면 OK
    #    하나라도 실패하면 NG
    #
    # 예:
    #
    # OCR = 0679256889404
    # Expected = 0679256889400
    #
    # OCR이 실제 인쇄값을 정확히 읽었더라도
    # Expected SN과 다르면 NG입니다.
    #
    # 왜 필요한가요?
    # 단순히 문자를 읽는 OCR 기능을
    # 제조 제품의 OK/NG 검사로 연결하기 위해서입니다.
    # --------------------------------------------------

    result = inspect_text(
        raw_text=raw_text,

        expected_text=item[
            "expected_text"
        ],

        pattern=rules[
            "expected_pattern"
        ],
    )


    # --------------------------------------------------
    # 16. OCR이 실제 인쇄된 SN을 정확하게 읽었는지 평가
    #
    # 무엇을 하나요?
    # - OCR 결과와 실제 인쇄된 SN을 비교합니다.
    # - 완전히 같으면 True, 다르면 False를 기록합니다.
    #
    # result["final_text"]:
    # 문자열 Cleaning을 마친 OCR 결과
    #
    # item["actual_text"]:
    # 제품 이미지에 실제로 인쇄된 SN 정답
    #
    # ocr_exact:
    # OCR이 실제 인쇄 문자를 정확하게 읽었는지 나타냅니다.
    #
    # 예:
    #
    # 실제 인쇄 SN = 0679256889404
    # OCR 결과      = 0679256889404
    #
    # ocr_exact = True
    #
    # 왜 필요한가요?
    # OCR 자체의 문자 인식 정확도를
    # 제품 판정 정확도와 별도로 평가하기 위해서입니다.
    #
    # 중요:
    # actual_text는 정확도 평가용 Ground Truth입니다.
    # OCR을 실행할 때 정답으로 사용하지 않습니다.
    # --------------------------------------------------

    ocr_exact = (
        result["final_text"]
        == item["actual_text"]
    )


    # --------------------------------------------------
    # 17. 프로그램의 최종 제품 판정이 정확한지 평가
    #
    # 무엇을 하나요?
    # - 프로그램이 판정한 OK/NG를 확인합니다.
    # - 실제 제품의 정답 OK/NG와 비교합니다.
    #
    # result["predicted_status"]:
    # OCR 결과와 Expected SN으로 판정한 OK/NG
    #
    # item["actual_status"]:
    # Ground Truth에 기록된 실제 제품 정답
    #
    # inspection_correct:
    # 최종 제품 판정이 실제 정답과 일치하는지 나타냅니다.
    #
    # 예:
    #
    # 실제 상태 = NG
    # 프로그램 판정 = NG
    #
    # inspection_correct = True
    #
    # 왜 필요한가요?
    # OCR이 글자를 정확하게 읽었는지와
    # 제품을 올바르게 OK/NG 판정했는지는
    # 서로 다른 평가이기 때문입니다.
    #
    # 중요:
    # actual_status는 평가용 정답으로만 사용합니다.
    # 제품의 predicted_status 계산에는 사용하지 않습니다.
    # --------------------------------------------------

    inspection_correct = (
        result[
            "predicted_status"
        ]
        == item[
            "actual_status"
        ]
    )


    # --------------------------------------------------
    # 18. 실제 Tesseract에 입력한 확대 이미지 저장
    #
    # 무엇을 하나요?
    # - 현재 이미지 이름을 이용하여 결과 파일명을 만듭니다.
    # - Tesseract에 전달한 OCR 입력 이미지를 저장합니다.
    #
    # image_path.stem:
    # 이미지 파일명에서 확장자를 제외한 부분
    #
    # 예:
    # 0637145699424_side.png
    #
    # 저장 파일:
    # 0637145699424_side_ocr_input.jpg
    #
    # cv2.imwrite():
    # 이미지를 파일로 저장하는 함수
    #
    # 왜 필요한가요?
    # OCR 결과가 잘못되었을 때
    # 실제로 어떤 이미지가 입력되었는지
    # 확인하고 분석하기 위해서입니다.
    # --------------------------------------------------

    output_image_path = (
        OUTPUT_DIR
        / f"{image_path.stem}_ocr_input.jpg"
    )

    cv2.imwrite(
        str(output_image_path),
        ocr_input,
    )


    # --------------------------------------------------
    # 19. 이미지별 OCR 및 제품 검사 결과 기록·출력
    #
    # 무엇을 하나요?
    # - 현재 이미지의 검사 결과를 딕셔너리로 정리합니다.
    # - rows 리스트에 추가합니다.
    # - 터미널에 결과를 출력합니다.
    #
    # 주요 결과:
    
    # actual_text:
    # 실제 인쇄된 SN
    #
    # expected_text:
    # 생산상 기대하는 SN
    #
    # raw_text:
    # Tesseract가 반환한 원본 문자열
    #
    # cleaned_text:
    # 공백·줄바꿈 등을 정리한 OCR 결과
    #
    # format_ok:
    # 숫자 13자리 형식 검사 결과
    #
    # expected_match:
    # OCR 결과와 Expected SN 일치 여부
    #
    # ocr_exact:
    # OCR 결과와 실제 인쇄 SN 일치 여부
    #
    # predicted_status:
    # 프로그램의 OK/NG 판정
    #
    # inspection_correct:
    # 프로그램 판정과 실제 제품 정답 일치 여부
    #
    # reasons:
    # NG 판정 이유
    #
    # 왜 필요한가요?
    # 제품별 검사 결과를 저장하고
    # 이후 전체 정확도를 계산하기 위해서입니다.
    # --------------------------------------------------

    row = {
        "file":
            file_name,

        "actual_text":
            item["actual_text"],

        "expected_text":
            item["expected_text"],

        "actual_status":
            item["actual_status"],

        "raw_text":
            repr(
                result["raw_text"]
            ),

        "cleaned_text":
            result["cleaned_text"],

        "final_text":
            result["final_text"],

        "format_ok":
            result["format_ok"],

        "expected_match":
            result["expected_match"],

        "ocr_exact":
            ocr_exact,

        "predicted_status":
            result["predicted_status"],

        "inspection_correct":
            inspection_correct,

        "reasons":
            "|".join(
                result["reasons"]
            ),
    }

    # 현재 검사 결과를 리스트에 추가
    rows.append(
        row
    )


    # ------------------------------------
    # 터미널에 현재 제품 검사 결과 출력
    # ------------------------------------

    print(
        "=" * 70
    )

    print(
        "File:",
        file_name
    )

    print(
        "Actual:",
        item["actual_text"]
    )

    print(
        "Expected:",
        item["expected_text"]
    )

    print(
        "OCR:",
        result["final_text"]
    )

    print(
        "Format OK:",
        result["format_ok"]
    )

    print(
        "Expected Match:",
        result["expected_match"]
    )

    print(
        "OCR Exact:",
        ocr_exact
    )

    print(
        "Predicted:",
        result["predicted_status"]
    )

    print(
        "Actual Status:",
        item["actual_status"]
    )

    print(
        "Inspection Correct:",
        inspection_correct
    )

    print(
        "Reasons:",
        result["reasons"]
    )


# --------------------------------------------------
# 20. 전체 Batch 검사 결과를 CSV로 저장하고 완료 출력
#
# 무엇을 하나요?
# - CSV를 저장할 reports 폴더를 준비합니다.
# - CSV에 기록할 컬럼명을 정의합니다.
# - 모든 제품의 검사 결과를 CSV로 저장합니다.
# - 저장 경로와 완료 메시지를 출력합니다.
#
# REPORT_PATH:
# reports/day06_ocr_results.csv
#
# fieldnames:
# CSV에 사용할 컬럼명 목록
#
# csv.DictWriter():
# 딕셔너리 데이터를 CSV로 저장하는 기능
#
# writer.writeheader():
# CSV 첫 번째 줄에 컬럼명 작성
#
# writer.writerows(rows):
# 모든 이미지의 검사 결과 저장
#
# encoding="utf-8-sig":
# Excel에서 한글이 깨질 가능성을 줄입니다.
#
# 왜 필요한가요?
# 이미지별 결과를 파일로 보존하고
# 46번 전체 성능 평가에서 재사용하기 위해서입니다.
#
# 주의:
# 같은 경로에 CSV가 있으면 기존 결과를 덮어씁니다.
# --------------------------------------------------

REPORT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)


fieldnames = [
    "file",
    "actual_text",
    "expected_text",
    "actual_status",
    "raw_text",
    "cleaned_text",
    "final_text",
    "format_ok",
    "expected_match",
    "ocr_exact",
    "predicted_status",
    "inspection_correct",
    "reasons",
]


with REPORT_PATH.open(
    "w",
    newline="",
    encoding="utf-8-sig",
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=fieldnames,
    )

    writer.writeheader()

    writer.writerows(
        rows
    )


print()

print(
    "Saved:",
    REPORT_PATH
)

print(
    "Day06 Batch OCR Complete"
)