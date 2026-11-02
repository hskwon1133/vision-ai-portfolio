# --------------------------------------------------
# 1. OCR 공통 처리에 필요한 라이브러리 불러오기
#
# 무엇을 하나요?
# - JSON 파일을 읽는 기능을 불러옵니다.
# - 프로젝트 파일과 폴더 경로를 관리합니다.
# - 이미지 처리와 문자 인식 기능을 불러옵니다.
#
# json:
# JSON 파일에 저장된 OCR Rule을 읽기 위해 사용합니다.
#
# Path:
# 파일과 폴더 경로를 관리합니다.
#
# cv2:
# OpenCV 라이브러리입니다.
# ROI 전처리, Gray, CLAHE, Otsu 등에 사용합니다.
#
# pytesseract:
# Python에서 Tesseract OCR Engine을 호출합니다.
#
# 왜 필요한가요?
# 이후 여러 OCR 실습에서 공통으로 사용할 기능을
# 하나의 Python 파일에 모아 두기 위해서입니다.
# --------------------------------------------------

import json
from pathlib import Path

import cv2
import pytesseract


# --------------------------------------------------
# 2. 프로젝트 최상위 폴더와 OCR Rule 경로 설정
#
# 무엇을 하나요?
# - 현재 Python 파일의 위치를 확인합니다.
# - 프로젝트 최상위 폴더를 찾습니다.
# - OCR Rule이 저장된 JSON 파일 경로를 지정합니다.
#
# __file__:
# 현재 Python 파일의 경로입니다.
#
# resolve():
# 파일의 절대경로를 구합니다.
#
# parent.parent:
# 현재 파일의 상위 폴더로 두 번 이동합니다.
#
# 예:
#
# subject9_machinevision/
# ├── src/
# │   └── day06_ocr_utils.py
# └── configs/
#     └── day06_ocr_rules.json
#
# DEFAULT_RULES_PATH:
# 41번 실습에서 저장한 OCR 설정 JSON 경로입니다.
#
# 왜 필요한가요?
# 이후 OCR 실습에서 동일한 JSON 설정을
# 반복해서 사용하기 위해서입니다.
# --------------------------------------------------

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

DEFAULT_RULES_PATH = (
    PROJECT_ROOT
    / "configs"
    / "day06_ocr_rules.json"
)


# --------------------------------------------------
# 3. JSON에 저장된 OCR Rule을 읽는 함수
#
# 무엇을 하나요?
# - load_rules() 함수를 정의합니다.
# - JSON 파일에 저장된 OCR 설정을 읽습니다.
# - JSON 데이터를 Python 딕셔너리로 변환합니다.
#
# path=DEFAULT_RULES_PATH:
# 별도의 파일 경로를 전달하지 않으면
# 기본 OCR Rule JSON을 사용합니다.
#
# Path(path).read_text():
# JSON 파일의 내용을 문자열로 읽습니다.
#
# json.loads():
# JSON 문자열을 Python 딕셔너리로 변환합니다.
#
# 반환하는 정보:
#
# baseline_version
# reference_file
# roi_by_file
# preprocess_mode
# clahe_clip_limit
# clahe_grid
# ocr_lang
# ocr_oem
# ocr_psm
# whitelist
# expected_pattern
# allow_context_correction
#
# 왜 필요한가요?
# Python 코드에 OCR 설정을 매번 직접 작성하지 않고
# JSON에 저장된 공통 설정을 가져오기 위해서입니다.
#
# 주의:
# JSON 파일이 없거나 내용이 잘못되면
# 읽는 과정에서 오류가 발생합니다.
# --------------------------------------------------

def load_rules(
    path=DEFAULT_RULES_PATH,
):
    return json.loads(
        Path(path).read_text(
            encoding="utf-8"
        )
    )


# --------------------------------------------------
# 4. 이미지 파일명에 해당하는 ROI 좌표 가져오기
#
# 무엇을 하나요?
# - JSON의 roi_by_file 정보를 가져옵니다.
# - 현재 이미지의 파일명을 확인합니다.
# - 해당 이미지에 저장된 ROI 좌표를 반환합니다.
#
# rules:
# JSON에서 읽은 OCR 설정 딕셔너리입니다.
#
# file_name:
# 현재 OCR할 이미지의 파일명입니다.
#
# roi_by_file:
# 각 이미지의 SN 영역 좌표가 저장된 딕셔너리입니다.
#
# 예:
#
# "0637145699424_side.png":
#     [434, 294, 529, 318]
#
# 파일명을 찾지 못했다면:
# KeyError를 발생시킵니다.
#
# 왜 필요한가요?
# 제품마다 SN의 위치가 서로 다르기 때문에
# 현재 이미지에 맞는 ROI 좌표를 가져와야 합니다.
#
# 이 함수는 이미지 자체를 잘라내지 않습니다.
# ROI 좌표만 가져오는 역할입니다.
# --------------------------------------------------

def get_roi_for_file(
    rules,
    file_name,
):

    # JSON에서 이미지별 ROI 좌표 목록 가져오기
    roi_by_file = rules[
        "roi_by_file"
    ]

    # 현재 이미지의 ROI 좌표가 있는지 확인
    if file_name not in roi_by_file:
        raise KeyError(
            f"ROI 정보가 없습니다: {file_name}"
        )

    # 현재 이미지에 해당하는 ROI 좌표 반환
    return roi_by_file[
        file_name
    ]


# --------------------------------------------------
# 5. ROI 좌표를 이용하여 SN 문자 영역 Crop
#
# 무엇을 하나요?
# - 원본 이미지와 ROI 좌표를 전달받습니다.
# - ROI 좌표를 네 개의 변수로 나눕니다.
# - 이미지에서 해당 부분만 잘라냅니다.
#
# image:
# OpenCV로 읽은 원본 이미지입니다.
#
# roi:
# [x1, y1, x2, y2] 형태의 좌표 리스트입니다.
#
# x1:
# 왼쪽 X좌표
#
# y1:
# 위쪽 Y좌표
#
# x2:
# 오른쪽 X좌표
#
# y2:
# 아래쪽 Y좌표
#
# image[y1:y2, x1:x2]:
# 세로와 가로 범위를 지정하여 이미지를 자릅니다.
#
# 주의:
# NumPy 이미지 배열은 [Y, X] 순서입니다.
#
# 왜 필요한가요?
# 제품 전체를 OCR하지 않고
# 필요한 13자리 SN 숫자 부분만 인식하기 위해서입니다.
#
# 이 함수는 전달받은 좌표의 유효성을
# 별도로 검사하지 않습니다.
# --------------------------------------------------

def crop_roi(
    image,
    roi,
):

    # ROI 좌표를 네 개의 변수로 분리
    x1, y1, x2, y2 = roi

    # 선택한 SN 영역만 잘라서 반환
    return image[
        y1:y2,
        x1:x2,
    ]


# --------------------------------------------------
# 6. Grayscale 이미지에 CLAHE 전처리를 적용하는 함수
#
# 무엇을 하나요?
# - Gray 이미지에 CLAHE를 적용합니다.
# - JSON에 저장된 CLAHE 설정을 사용합니다.
# - 대비가 조정된 이미지를 반환합니다.
#
# CLAHE란?
# Contrast Limited Adaptive Histogram Equalization
#
# 이미지의 작은 영역별로 대비를 조정하여
# 흐린 문자와 배경의 밝기 차이를 개선하는 방법입니다.
#
# gray:
# 흑백으로 변환된 이미지입니다.
#
# rules:
# CLAHE 설정이 저장된 OCR Rule 딕셔너리입니다.
#
# clahe_clip_limit:
# 대비가 지나치게 강조되는 것을 제한하는 값
#
# clahe_grid:
# 이미지의 대비를 계산할 영역 구분 설정
#
# cv2.createCLAHE():
# CLAHE 처리 객체를 생성합니다.
#
# clahe.apply():
# Gray 이미지에 CLAHE를 실제로 적용합니다.
#
# 왜 필요한가요?
# 대비가 낮은 문자에서
# CLAHE가 OCR에 도움이 되는지 비교하기 위해서입니다.
#
# 주의:
# CLAHE가 항상 OCR 성능을 높이는 것은 아닙니다.
# 실제 OCR 결과를 비교하여 판단해야 합니다.
# --------------------------------------------------

def apply_clahe(
    gray,
    rules,
):

    # JSON의 설정값으로 CLAHE 객체 생성
    clahe = cv2.createCLAHE(
        clipLimit=float(
            rules["clahe_clip_limit"]
        ),
        tileGridSize=tuple(
            rules["clahe_grid"]
        ),
    )

    # Gray 이미지에 CLAHE 적용
    return clahe.apply(
        gray
    )


# --------------------------------------------------
# 7. 동일한 SN ROI에서 다섯 가지 전처리 후보 생성
#
# 무엇을 하나요?
# - 하나의 SN ROI를 입력받습니다.
# - 서로 다른 다섯 가지 이미지 상태를 만듭니다.
#
# 생성할 전처리 후보:
#
# ① Original
#    원본 ROI를 그대로 사용
#
# ② Gray
#    색상 정보를 제거한 흑백 이미지
#
# ③ CLAHE
#    Gray 이미지의 국소 대비 개선
#
# ④ Otsu
#    Gray 이미지에 자동 이진화 적용
#
# ⑤ CLAHE + Otsu
#    대비 개선 후 자동 이진화 적용
#
# 왜 필요한가요?
# 같은 SN 이미지라도 전처리 방법에 따라
# Tesseract OCR 결과가 달라질 수 있기 때문입니다.
#
# 이 함수에서는 아직 OCR을 실행하지 않습니다.
# OCR에 입력할 이미지 후보만 생성합니다.
# --------------------------------------------------

def make_preprocess_variants(
    roi,
    rules,
):

    # --------------------------------------------------
    # 7-1. 원본 ROI를 Grayscale로 변환
    #
    # cv2.cvtColor():
    # 이미지 색상 형식을 변환하는 함수
    #
    # cv2.COLOR_BGR2GRAY:
    # BGR 컬러 이미지를 Gray로 변환
    #
    # 왜 필요한가요?
    # CLAHE와 Otsu를 적용할 흑백 이미지를
    # 준비하기 위해서입니다.
    # --------------------------------------------------

    # Gray
    gray = cv2.cvtColor(
        roi,
        cv2.COLOR_BGR2GRAY,
    )


    # --------------------------------------------------
    # 7-2. Gray 이미지에 CLAHE 적용
    #
    # 앞에서 작성한 apply_clahe() 함수를 호출합니다.
    #
    # Gray:
    # 색상 정보가 제거된 이미지
    #
    # CLAHE:
    # Gray 이미지에서 국소 대비를 개선한 이미지
    #
    # 왜 필요한가요?
    # 흐린 숫자를 더 잘 구분할 수 있도록
    # 대비를 개선한 OCR 후보를 만들기 위해서입니다.
    # --------------------------------------------------

    # CLAHE
    clahe = apply_clahe(
        gray,
        rules,
    )


    # --------------------------------------------------
    # 7-3. Gray 이미지에 Otsu 이진화 적용
    #
    # 무엇을 하나요?
    # - Gray 이미지에서 자동으로 Threshold를 계산합니다.
    # - 픽셀을 검정 또는 흰색으로 구분합니다.
    #
    # cv2.threshold():
    # Threshold 기준으로 이미지를 변환하는 함수
    #
    # cv2.THRESH_BINARY:
    # 이진화 적용
    #
    # cv2.THRESH_OTSU:
    # Otsu 방법으로 Threshold 자동 계산
    #
    # _, otsu:
    # 첫 번째 반환값인 Threshold는 사용하지 않고
    # 두 번째 반환값인 이진화 이미지만 저장합니다.
    #
    # 왜 필요한가요?
    # 문자와 배경을 이진화하면
    # OCR 결과가 어떻게 달라지는지 확인하기 위해서입니다.
    # --------------------------------------------------

    # Gray + Otsu
    _, otsu = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY
        + cv2.THRESH_OTSU,
    )


    # --------------------------------------------------
    # 7-4. CLAHE 이미지에 Otsu 이진화 적용
    #
    # 무엇을 하나요?
    # - 앞에서 CLAHE를 적용한 이미지를 사용합니다.
    # - 그 이미지에 Otsu 이진화를 다시 적용합니다.
    #
    # 처리 순서:
    #
    # Gray
    #   ↓
    # CLAHE
    #   ↓
    # Otsu
    #
    # 왜 필요한가요?
    # 대비를 먼저 개선한 후 이진화하는 방법이
    # OCR에 도움이 되는지 비교하기 위해서입니다.
    # --------------------------------------------------

    # CLAHE + Otsu
    _, clahe_otsu = cv2.threshold(
        clahe,
        0,
        255,
        cv2.THRESH_BINARY
        + cv2.THRESH_OTSU,
    )


    # --------------------------------------------------
    # 7-5. 다섯 가지 전처리 후보를 딕셔너리로 반환
    #
    # 무엇을 하나요?
    # - 전처리 방법 이름을 Key로 사용합니다.
    # - 해당 전처리 결과 이미지를 Value로 저장합니다.
    #
    # original:
    # 원본 SN ROI
    #
    # gray:
    # 흑백 이미지
    #
    # clahe:
    # CLAHE 적용 이미지
    #
    # otsu:
    # Gray + Otsu 이미지
    #
    # clahe_otsu:
    # CLAHE + Otsu 이미지
    #
    # 왜 필요한가요?
    # 이후 다른 코드에서 전처리 이름으로
    # 필요한 이미지를 쉽게 가져오기 위해서입니다.
    # --------------------------------------------------

    return {
        "original": roi,
        "gray": gray,
        "clahe": clahe,
        "otsu": otsu,
        "clahe_otsu": clahe_otsu,
    }


# --------------------------------------------------
# 8. JSON에서 선택한 전처리 한 가지 가져오기
#
# 무엇을 하나요?
# - 앞에서 만든 다섯 가지 전처리 후보를 준비합니다.
# - JSON의 preprocess_mode 값을 읽습니다.
# - 선택된 전처리 결과 하나를 반환합니다.
#
# preprocess_mode:
# OCR에 사용할 전처리 방법 이름
#
# 예:
# preprocess_mode = "original"
#
# 선택 가능한 값:
#
# original
# gray
# clahe
# otsu
# clahe_otsu
#
# preprocess_mode = NOT_SELECTED이면?
# 사용할 전처리가 아직 확정되지 않았으므로
# ValueError를 발생시킵니다.
#
# 왜 필요한가요?
# OCR 비교 실험에서 선택한 방법을
# 이후 Batch 검사에서도 동일하게 사용하기 위해서입니다.
#
# 중요:
# 현재 코드는 다섯 가지 후보를 먼저 모두 계산하고
# 그중 선택된 하나를 반환합니다.
#
# 따라서 original을 선택하더라도
# 나머지 전처리 계산은 내부적으로 수행합니다.
# --------------------------------------------------

def preprocess_for_ocr(
    roi,
    rules,
):

    # 다섯 가지 전처리 후보 생성
    variants = (
        make_preprocess_variants(
            roi,
            rules,
        )
    )

    # JSON에서 선택된 전처리 방법 확인
    mode = rules[
        "preprocess_mode"
    ]

    # 선택한 방법이 후보 목록에 있는지 확인
    if mode not in variants:
        raise ValueError(
            "아직 사용할 전처리가 "
            f"선택되지 않았습니다: {mode}"
        )

    # 선택된 전처리 결과 이미지 반환
    return variants[
        mode
    ]


# --------------------------------------------------
# 9. Tesseract OCR 실행 옵션을 문자열로 만드는 함수
#
# 무엇을 하나요?
# - JSON에 저장된 Tesseract 설정을 가져옵니다.
# - OCR Engine Mode를 설정합니다.
# - Page Segmentation Mode를 설정합니다.
# - 인식할 문자 종류를 제한합니다.
# - 최종 Tesseract 옵션 문자열을 반환합니다.
#
# ocr_oem:
# Tesseract OCR Engine Mode
#
# ocr_psm:
# Tesseract Page Segmentation Mode
#
# 이번 실습:
# PSM = 7
# → 입력 이미지가 한 줄짜리 문자라고 지정
#
# whitelist:
# 인식할 문자 종류를 제한합니다.
#
# 이번 실습:
# 0123456789
# → 숫자만 허용
#
# 생성되는 설정 예:
#
# --oem 3 --psm 7 -c tessedit_char_whitelist=0123456789
#
# 왜 필요한가요?
# 여러 OCR 코드에서 같은 Tesseract 설정을
# 반복해서 작성하지 않기 위해서입니다.
#
# 참고:
# 이 함수는 OCR을 실행하는 함수가 아닙니다.
# Tesseract에 전달할 옵션 문자열만 생성합니다.
# --------------------------------------------------

def build_ocr_config(
    rules,
):

    return (
        f"--oem {rules['ocr_oem']} "
        f"--psm {rules['ocr_psm']} "
        "-c "
        f"tessedit_char_whitelist="
        f"{rules['whitelist']}"
    )


# --------------------------------------------------
# 10. Tesseract OCR을 실행하여 문자열 반환
#
# 무엇을 하나요?
# - 전처리가 완료된 이미지를 전달받습니다.
# - 앞에서 작성한 build_ocr_config()를 호출합니다.
# - OCR 실행 옵션을 준비합니다.
# - pytesseract를 이용하여 문자를 인식합니다.
# - 인식한 문자열을 반환합니다.
#
# image:
# Tesseract에 입력할 이미지
#
# rules:
# JSON에서 읽은 OCR 설정
#
# config:
# Tesseract 실행 옵션 문자열
#
# pytesseract.image_to_string():
# 이미지에서 문자를 인식하여 문자열로 반환합니다.
#
# lang=rules["ocr_lang"]:
# 사용할 OCR 언어 데이터를 지정합니다.
#
# config=config:
# 앞에서 준비한 Tesseract 실행 옵션을 적용합니다.
#
# 왜 필요한가요?
# Python에서 실제 Tesseract OCR Engine을 호출하여
# 이미지 속 13자리 SN 숫자를 읽기 위해서입니다.
#
# 반환 예:
#
# '0637145699424\n'
#
# 주의:
# 반환되는 값은 Raw OCR 문자열입니다.
# 마지막에 줄바꿈 문자나 공백이 포함될 수 있습니다.
#
# 문자열 정리, 13자리 Regex 검사,
# Expected SN 비교와 OK/NG 판정은
# 이후 day06_ocr_validation.py에서 수행합니다.
# --------------------------------------------------

def run_ocr(
    image,
    rules,
):

    # Tesseract OCR 실행 옵션 생성
    config = (
        build_ocr_config(
            rules
        )
    )

    # 실제 OCR 실행 후 Raw 문자열 반환
    return pytesseract.image_to_string(
        image,
        lang=rules["ocr_lang"],
        config=config,
    )