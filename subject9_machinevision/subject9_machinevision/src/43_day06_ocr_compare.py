# --------------------------------------------------
# 1. 필요한 라이브러리와 OCR 공통 함수 불러오기
#
# 무엇을 하나요?
# - CSV 저장, 파일 경로, 이미지 처리 기능을 준비합니다.
# - 앞에서 작성한 OCR 공통 함수들을 불러옵니다.
#
# csv:
# 전처리별 OCR 비교 결과를 CSV로 저장합니다.
#
# Path:
# 파일 경로와 파일명을 관리합니다.
#
# cv2:
# 이미지 읽기, 확대, 결과 이미지 저장에 사용합니다.
#
# load_rules():
# JSON에 저장된 OCR 설정을 읽습니다.
#
# get_roi_for_file():
# 대표 이미지에 해당하는 ROI 좌표를 가져옵니다.
#
# crop_roi():
# SN 숫자 영역만 잘라냅니다.
#
# make_preprocess_variants():
# 5가지 전처리 후보를 생성합니다.
#
# run_ocr():
# Tesseract OCR을 실행하여 문자열을 반환합니다.
#
# 왜 필요한가요?
# 이전 실습에서 작성한 공통 함수를 재사용하여
# 전처리와 OCR 코드를 다시 작성하지 않기 위해서입니다.
# --------------------------------------------------

import csv
from pathlib import Path

import cv2

from day06_ocr_utils import (
    PROJECT_ROOT,
    crop_roi,
    get_roi_for_file,
    load_rules,
    make_preprocess_variants,
    run_ocr,
)


# --------------------------------------------------
# 2. JSON에서 OCR Rule과 ROI 설정 읽기
#
# 무엇을 하나요?
# - load_rules() 공통 함수를 호출합니다.
# - day06_ocr_rules.json의 설정을 읽습니다.
#
# JSON에는 다음 정보가 저장되어 있습니다.
#
# reference_file:
# 대표 실험 이미지의 파일명
#
# roi_by_file:
# 이미지별 SN 영역 좌표
#
# clahe_clip_limit / clahe_grid:
# CLAHE 전처리 설정
#
# ocr_lang:
# Tesseract가 사용할 언어 데이터
#
# ocr_oem / ocr_psm:
# Tesseract 실행 방식
#
# whitelist:
# 인식하도록 허용한 문자 목록
#
# 왜 필요한가요?
# 모든 전처리 후보에 동일한 OCR 설정을 사용하여
# 공정하게 인식 결과를 비교하기 위해서입니다.
#
# 주의:
# 현재 preprocess_mode가 NOT_SELECTED여도 괜찮습니다.
# 이번 코드는 다섯 가지 후보를 직접 비교하기 때문입니다.
# --------------------------------------------------

rules = load_rules()


# --------------------------------------------------
# 3. 대표 이미지와 결과 저장 경로 설정
#
# 무엇을 하나요?
# - JSON에서 대표 이미지 파일명을 가져옵니다.
# - 해당 이미지의 전체 경로를 만듭니다.
# - OCR 비교 결과 CSV와 이미지 저장 폴더를 지정합니다.
#
# file_name:
# 대표 이미지 파일명
#
# IMAGE_PATH:
# 원본 제품 이미지 경로
#
# REPORT_PATH:
# 전처리별 OCR 결과를 저장할 CSV 경로
#
# OUTPUT_DIR:
# 실제 Tesseract에 입력한 확대 이미지를 저장할 폴더
#
# 왜 필요한가요?
# 대표 이미지 한 장에 동일한 OCR 조건을 적용하고
# 각 결과를 비교·기록하기 위해서입니다.
# --------------------------------------------------

file_name = rules[
    "reference_file"
]

IMAGE_PATH = (
    PROJECT_ROOT
    / "samples"
    / "day06"
    / "product"
    / file_name
)

REPORT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "day06_preprocess_compare.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "day06"
    / "preprocess"
)


# --------------------------------------------------
# 4. 대표 이미지 파일명에서 실제 SN 정답 가져오기
#
# 무엇을 하나요?
# - 이미지 파일명에서 확장자를 제거합니다.
# - "_side" 앞부분의 숫자를 가져옵니다.
# - OCR이 정확하게 읽었는지 비교할 정답으로 사용합니다.
#
# 예:
# 0637145699424_side.png
#
# Path(file_name).stem:
# 0637145699424_side
#
# .split("_side")[0]:
# 0637145699424
#
# expected_sn:
# 이번 OCR 비교 실험에서 사용하는 실제 인쇄 SN 정답
#
# 왜 필요한가요?
# Tesseract가 읽은 숫자가 실제 이미지에
# 인쇄된 숫자와 정확히 같은지 확인하기 위해서입니다.
#
# 주의:
# 여기서 expected_sn은 OCR 정확도 비교용 정답입니다.
# 이후 제품 검사에서 사용하는 생산상 Expected SN과는
# 역할이 다를 수 있습니다.
#
# 또한 SN은 앞자리가 0일 수 있으므로
# 숫자형(int)이 아닌 문자열로 유지합니다.
# --------------------------------------------------

expected_sn = (
    Path(file_name)
    .stem
    .split("_side")[0]
)


# --------------------------------------------------
# 5. 대표 제품 이미지 읽기 및 오류 확인
#
# 무엇을 하나요?
# - OpenCV로 대표 제품 이미지를 읽습니다.
# - 이미지 읽기에 성공했는지 확인합니다.
#
# cv2.imread():
# 이미지 파일을 읽는 함수
#
# image:
# 읽어온 원본 제품 이미지
#
# image is None:
# 이미지 파일을 읽지 못했다는 의미
#
# 왜 필요한가요?
# OCR을 수행하려면 먼저 원본 제품 이미지를 읽고
# 그 안에서 SN 숫자 영역을 잘라야 하기 때문입니다.
#
# 이미지가 없거나 읽기에 실패하면
# FileNotFoundError를 발생시킵니다.
# --------------------------------------------------

image = cv2.imread(
    str(IMAGE_PATH)
)

if image is None:
    raise FileNotFoundError(
        IMAGE_PATH
    )


# --------------------------------------------------
# 6. 대표 이미지의 SN ROI 좌표 가져오기
#
# 무엇을 하나요?
# - get_roi_for_file() 함수를 호출합니다.
# - JSON에 저장된 대표 이미지의 ROI 좌표를 가져옵니다.
#
# 전달하는 정보:
# rules:
# OCR 설정 및 이미지별 ROI 좌표
#
# file_name:
# 현재 대표 이미지의 파일명
#
# 반환 예:
# [434, 294, 529, 318]
#
# 좌표 의미:
# [x1, y1, x2, y2]
#
# 왜 필요한가요?
# 41번에서 선택한 정확한 SN 위치를 사용하여
# 같은 문자 영역을 다시 추출하기 위해서입니다.
#
# 현재 단계에서는 좌표만 가져옵니다.
# --------------------------------------------------

roi_coordinates = (
    get_roi_for_file(
        rules,
        file_name,
    )
)


# --------------------------------------------------
# 7. 저장된 좌표로 SN 숫자 영역 Crop
#
# 무엇을 하나요?
# - crop_roi() 공통 함수를 호출합니다.
# - 원본 이미지에서 13자리 SN 부분만 잘라냅니다.
#
# image:
# 원본 제품 이미지
#
# roi_coordinates:
# JSON에 저장된 ROI 좌표
#
# roi:
# 잘라낸 SN 숫자 이미지
#
# crop_roi() 내부 처리:
# image[y1:y2, x1:x2]
#
# 왜 필요한가요?
# 전체 이미지에는 SN 외에도 여러 문자와 배경이 있으므로
# OCR 대상인 13자리 숫자만 추출하기 위해서입니다.
#
# 아직 OCR을 실행하지 않습니다.
# --------------------------------------------------

roi = crop_roi(
    image,
    roi_coordinates,
)


# --------------------------------------------------
# 8. SN ROI에 다섯 가지 전처리 후보 생성
#
# 무엇을 하나요?
# - make_preprocess_variants() 함수를 호출합니다.
# - 동일한 SN ROI에서 5가지 전처리 이미지를 생성합니다.
#
# ① original:
# 원본 SN ROI
#
# ② gray:
# Grayscale 변환
#
# ③ clahe:
# Gray 이미지의 국소 대비 조정
#
# ④ otsu:
# Gray 이미지에 Otsu 자동 이진화
#
# ⑤ clahe_otsu:
# CLAHE 처리 후 Otsu 이진화
#
# 왜 필요한가요?
# 같은 SN이라도 영상의 전처리 상태에 따라
# Tesseract 인식 결과가 달라지는지 확인하기 위해서입니다.
#
# variants:
# 전처리 방법 이름과 결과 이미지를 보관하는 딕셔너리
#
# 예:
# variants["original"] → 원본 ROI
# variants["gray"]     → Gray 이미지
#
# 참고:
# 42번과 동일한 공통 함수를 호출하므로
# 같은 방식으로 전처리 후보를 생성합니다.
# --------------------------------------------------

variants = (
    make_preprocess_variants(
        roi,
        rules,
    )
)


# --------------------------------------------------
# 9. OCR 결과 이미지 폴더 및 결과 리스트 준비
#
# 무엇을 하나요?
# - OCR 입력 이미지를 저장할 폴더를 생성합니다.
# - 전처리별 OCR 결과를 저장할 빈 리스트를 만듭니다.
#
# OUTPUT_DIR.mkdir():
# 결과 이미지 폴더 생성
#
# parents=True:
# 필요한 상위 폴더까지 생성
#
# exist_ok=True:
# 이미 폴더가 있어도 오류 없이 진행
#
# rows:
# 전처리별 OCR 결과를 저장할 리스트
#
# 왜 필요한가요?
# 다섯 가지 OCR 결과를 순서대로 저장하고
# 마지막에 하나의 CSV로 정리하기 위해서입니다.
# --------------------------------------------------

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


rows = []


# --------------------------------------------------
# 10. 전처리 후보를 하나씩 가져와 OCR용으로 4배 확대
#
# 무엇을 하나요?
# - variants에서 전처리 이미지를 하나씩 가져옵니다.
# - 각 이미지를 가로·세로 모두 4배 확대합니다.
#
# enumerate():
# 반복하면서 순번을 생성하는 함수
#
# index:
# 전처리 이미지 순번
#
# name:
# 전처리 방법 이름
#
# processed:
# 현재 전처리 결과 이미지
#
# cv2.resize():
# 이미지 크기를 변경하는 함수
#
# fx=4.0:
# 가로 크기를 4배 확대
#
# fy=4.0:
# 세로 크기를 4배 확대
#
# INTER_CUBIC:
# 주변 픽셀을 이용하여 보간하는 방법
#
# 왜 필요한가요?
# 현재 SN ROI는 작은 이미지이므로
# Tesseract가 문자 획을 좀 더 큰 크기로
# 처리할 수 있도록 준비하기 위해서입니다.
#
# 중요:
# 다섯 가지 이미지에 모두 같은 4배 확대를 적용합니다.
# --------------------------------------------------

for index, (
    name,
    processed,
) in enumerate(
    variants.items(),
    start=1,
):

    # 작은 문자 ROI를 OCR용으로 4배 확대
    enlarged = cv2.resize(
        processed,
        None,
        fx=4.0,
        fy=4.0,
        interpolation=cv2.INTER_CUBIC,
    )


    # --------------------------------------------------
    # 11. Tesseract OCR 실행 및 Raw 문자열 정리
    #
    # 무엇을 하나요?
    # - 앞에서 만든 run_ocr() 공통 함수를 호출합니다.
    # - 4배 확대된 이미지를 Tesseract에 전달합니다.
    # - Tesseract가 읽은 문자열을 가져옵니다.
    # - 앞뒤 공백과 줄바꿈을 제거합니다.
    #
    # run_ocr():
    # Tesseract OCR을 실행하는 공통 함수
    #
    # 내부 OCR 설정 예:
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
    # raw_text.strip():
    # 문자열의 앞뒤 공백과 줄바꿈을 제거합니다.
    #
    # ocr_text:
    # 앞뒤 공백·줄바꿈을 정리한 OCR 결과
    #
    # 예:
    # 0637145699424
    #
    # 왜 필요한가요?
    # OCR 결과를 실제 SN 정답과
    # 정확하게 비교하기 위해서입니다.
    #
    # 주의:
    # strip()은 문자열 중간의 숫자를 고치지 않습니다.
    # 이 코드에서는 중간 공백도 제거하지 않습니다.
    # --------------------------------------------------

    # Tesseract OCR
    raw_text = run_ocr(
        enlarged,
        rules,
    )

    # 줄바꿈과 앞뒤 공백만 제거
    ocr_text = raw_text.strip()


    # --------------------------------------------------
    # 12. 실제 SN과 OCR 결과의 Exact Match 확인
    #
    # 무엇을 하나요?
    # - Tesseract가 읽은 문자열과 실제 SN을 비교합니다.
    # - 처음부터 끝까지 완전히 같은지 확인합니다.
    #
    # expected_sn:
    # 이미지 파일명에서 가져온 실제 SN 정답
    #
    # ocr_text:
    # Tesseract가 인식한 SN
    #
    # exact_match:
    # 두 문자열이 완전히 일치하는지 나타내는 값
    #
    # True:
    # OCR이 실제 SN을 정확하게 읽음
    #
    # False:
    # 한 글자라도 다르거나 길이가 다름
    #
    # 예:
    # 실제 SN: 0637145699424
    # OCR SN:  0637145699424
    # 결과: True
    #
    # 실제 SN: 0637145699424
    # OCR SN:  0637245697424
    # 결과: False
    #
    # 왜 필요한가요?
    # 사람 눈에 선명한 전처리가 아니라
    # 실제로 정확하게 읽힌 전처리를 찾기 위해서입니다.
    #
    # 주의:
    # 이것은 OCR 문자열 정확도 비교입니다.
    # 아직 제품의 최종 OK/NG 판정은 수행하지 않습니다.
    # --------------------------------------------------

    exact_match = (
        ocr_text
        == expected_sn
    )


    # --------------------------------------------------
    # 13. 실제 Tesseract 입력 이미지를 JPG로 저장
    #
    # 무엇을 하나요?
    # - 현재 전처리 이름과 순번으로 파일명을 만듭니다.
    # - 4배 확대된 OCR 입력 이미지를 저장합니다.
    #
    # f"{index:02d}":
    # 숫자를 두 자리 형식으로 표시합니다.
    #
    # 예:
    # 1 → 01
    # 2 → 02
    #
    # 저장 파일 예:
    #
    # ocr_01_original.jpg
    # ocr_02_gray.jpg
    # ocr_03_clahe.jpg
    # ocr_04_otsu.jpg
    # ocr_05_clahe_otsu.jpg
    #
    # 왜 필요한가요?
    # Tesseract에 전달한 이미지의 상태를
    # 직접 확인하고 OCR 오류 원인을 분석하기 위해서입니다.
    #
    # 주의:
    # 실제 OCR은 enlarged 배열에 수행합니다.
    # 저장된 JPG는 해당 입력을 확인하기 위한 결과물입니다.
    # --------------------------------------------------

    output_path = (
        OUTPUT_DIR
        / f"ocr_{index:02d}_{name}.jpg"
    )

    cv2.imwrite(
        str(output_path),
        enlarged,
    )


    # --------------------------------------------------
    # 14. 전처리별 OCR 결과 기록 및 터미널 출력
    #
    # 무엇을 하나요?
    # - 현재 전처리 방법의 OCR 결과를 정리합니다.
    # - rows 리스트에 추가합니다.
    # - 현재 결과를 터미널에 출력합니다.
    #
    # pipeline:
    # 전처리 방법 이름
    #
    # expected_sn:
    # 실제 SN 정답
    #
    # raw_text:
    # Tesseract의 원본 문자열
    #
    # ocr_text:
    # 앞뒤 공백과 줄바꿈을 제거한 결과
    #
    # exact_match:
    # OCR 결과가 정답과 같은지 확인한 값
    #
    # repr(raw_text):
    # 줄바꿈 문자가 포함된 문자열을
    # 사람이 확인하기 쉬운 형태로 표현합니다.
    #
    # 예:
    # '0637145699424\n'
    #
    # 왜 필요한가요?
    # 다섯 가지 전처리의 결과를
    # 한눈에 비교할 수 있도록 기록하기 위해서입니다.
    #
    # 반복문이 끝나면 rows에는
    # 다섯 가지 전처리의 OCR 결과가 저장됩니다.
    # --------------------------------------------------

    rows.append(
        {
            "pipeline":
                name,

            "expected_sn":
                expected_sn,

            "raw_text":
                repr(raw_text),

            "ocr_text":
                ocr_text,

            "exact_match":
                exact_match,
        }
    )


    print(
        "=" * 60
    )

    print(
        "Pipeline:",
        name
    )

    print(
        "Expected:",
        expected_sn
    )

    print(
        "Raw OCR:",
        repr(raw_text)
    )

    print(
        "OCR Text:",
        ocr_text
    )

    print(
        "Exact Match:",
        exact_match
    )


# --------------------------------------------------
# 15. 다섯 가지 전처리의 OCR 비교 결과를 CSV로 저장
#
# 무엇을 하나요?
# - 결과 CSV를 저장할 폴더를 준비합니다.
# - CSV 파일을 생성합니다.
# - 컬럼명을 작성합니다.
# - rows에 모아 둔 OCR 결과를 저장합니다.
#
# 저장 파일:
# reports/day06_preprocess_compare.csv
#
# csv.DictWriter():
# 딕셔너리 데이터를 CSV로 저장하는 기능
#
# fieldnames:
# CSV에 사용할 컬럼 이름
#
# writer.writeheader():
# CSV 첫 행에 컬럼명을 작성
#
# writer.writerows(rows):
# 다섯 가지 전처리 결과를 한 번에 저장
#
# encoding="utf-8-sig":
# Excel에서 한글이 깨질 가능성을 줄입니다.
#
# newline="":
# CSV 저장 시 불필요한 빈 줄이 생기는 문제를 방지합니다.
#
# 왜 필요한가요?
# 전처리별 OCR 정확도를 표로 비교하고
# 다음 Rule Freeze 단계에서 참고하기 위해서입니다.
# --------------------------------------------------

REPORT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)


with REPORT_PATH.open(
    "w",
    newline="",
    encoding="utf-8-sig",
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=[
            "pipeline",
            "expected_sn",
            "raw_text",
            "ocr_text",
            "exact_match",
        ],
    )

    writer.writeheader()

    writer.writerows(
        rows
    )


# --------------------------------------------------
# 16. OCR 비교 완료 메시지 및 저장 위치 출력
#
# 무엇을 하나요?
# - CSV 저장 경로를 출력합니다.
# - OCR 비교가 완료되었다는 메시지를 출력합니다.
#
# 왜 필요한가요?
# 다섯 가지 전처리의 OCR 비교가 끝났고
# 결과 파일이 저장되었는지 확인하기 위해서입니다.
#
# 다음 실습:
# 44_day06_freeze_ocr_rules.py
#
# 이번 OCR 비교 결과를 바탕으로
# 사용할 전처리와 OCR 설정을 확정합니다.
#
# 주의:
# 이 코드에서는 JSON Rule을 변경하지 않습니다.
# 전처리 방법의 최종 확정은 다음 실습에서 수행합니다.
# --------------------------------------------------

print()

print(
    "Saved:",
    REPORT_PATH
)

print(
    "OCR Compare Complete"
)