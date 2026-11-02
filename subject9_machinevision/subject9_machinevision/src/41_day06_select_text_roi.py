# --------------------------------------------------
# 1. 필요한 라이브러리 불러오기
#
# 무엇을 하나요?
# - JSON 파일을 저장할 기능을 불러옵니다.
# - 파일과 폴더 경로를 관리할 기능을 불러옵니다.
# - 이미지 읽기, ROI 선택, 이미지 저장 기능을 불러옵니다.
#
# json:
# Python 딕셔너리를 JSON 파일로 저장하는 라이브러리
#
# Path:
# 프로젝트의 파일과 폴더 경로를 관리하는 클래스
#
# cv2:
# OpenCV 라이브러리
# 이미지 읽기, 마우스 ROI 선택, 이미지 저장에 사용
#
# 왜 필요한가요?
# 제품 이미지에서 SN 영역을 선택하고
# 선택한 이미지와 좌표를 저장하기 위해서입니다.
# --------------------------------------------------

import json
from pathlib import Path

import cv2


# --------------------------------------------------
# 2. 프로젝트와 입력·출력 파일 경로 설정
#
# 무엇을 하나요?
# - 현재 Python 파일의 위치를 기준으로
#   프로젝트 최상위 폴더를 찾습니다.
# - 제품 이미지가 있는 폴더를 지정합니다.
# - OCR Rule JSON을 저장할 위치를 지정합니다.
# - ROI 결과 이미지를 저장할 폴더를 지정합니다.
#
# PROJECT_ROOT:
# 프로젝트 최상위 폴더
#
# PRODUCT_DIR:
# 본 실습용 제품 이미지 6장이 저장된 폴더
#
# CONFIG_PATH:
# 이미지별 ROI 좌표와 OCR 설정을 저장할 JSON 파일
#
# OUTPUT_DIR:
# 선택한 ROI와 위치 확인 이미지를 저장할 폴더
#
# 왜 필요한가요?
# 프로그램이 어떤 이미지를 읽고
# 결과를 어디에 저장해야 하는지 지정하기 위해서입니다.
# --------------------------------------------------

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

PRODUCT_DIR = (
    PROJECT_ROOT
    / "samples"
    / "day06"
    / "product"
)

CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "day06_ocr_rules.json"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "day06"
    / "roi"
)


# --------------------------------------------------
# 3. 본 실습용 제품 이미지 목록 만들기
#
# 무엇을 하나요?
# - PRODUCT_DIR 폴더의 파일을 확인합니다.
# - 확장자가 jpg, jpeg, png인 이미지만 선택합니다.
# - 선택한 이미지를 파일명 순서로 정렬합니다.
#
# PRODUCT_DIR.iterdir():
# 폴더 안에 있는 파일과 폴더를 확인합니다.
#
# path.suffix.lower():
# 파일 확장자를 소문자로 가져옵니다.
#
# sorted():
# 이미지 목록을 정렬합니다.
#
# 왜 필요한가요?
# 제품 이미지 6장을 순서대로 열고
# 각 이미지의 SN 영역을 선택하기 위해서입니다.
#
# 참고:
# 이 코드는 정해진 파일명 6개가 아니라
# 폴더에 존재하는 지원 형식의 이미지를 모두 찾습니다.
# --------------------------------------------------

IMAGE_PATHS = sorted(
    [
        path
        for path in PRODUCT_DIR.iterdir()
        if path.suffix.lower()
        in {".jpg", ".jpeg", ".png"}
    ]
)


# --------------------------------------------------
# 4. 이미지 개수 확인 및 결과 저장 폴더 생성
#
# 무엇을 하나요?
# - 제품 이미지가 있는지 확인합니다.
# - 이미지가 하나도 없다면 오류를 발생시킵니다.
# - 찾은 이미지 개수를 터미널에 출력합니다.
# - ROI 결과를 저장할 폴더를 생성합니다.
#
# if not IMAGE_PATHS:
# 이미지 목록이 비어 있는지 확인합니다.
#
# FileNotFoundError:
# 필요한 이미지가 없을 때 발생시키는 오류입니다.
#
# mkdir():
# 결과를 저장할 폴더를 생성합니다.
#
# parents=True:
# 필요한 상위 폴더까지 함께 생성합니다.
#
# exist_ok=True:
# 폴더가 이미 있어도 오류를 발생시키지 않습니다.
#
# 왜 필요한가요?
# 이미지가 준비되어 있는지 확인하고
# 결과를 저장할 공간을 준비하기 위해서입니다.
#
# 주의:
# 현재 코드는 이미지가 6장인지 검사하지 않습니다.
# 한 장 이상 있으면 발견한 이미지들을 처리합니다.
# --------------------------------------------------

if not IMAGE_PATHS:
    raise FileNotFoundError(
        f"이미지가 없습니다: {PRODUCT_DIR}"
    )


print(
    "찾은 제품 이미지:",
    len(IMAGE_PATHS),
)


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# --------------------------------------------------
# 5. ROI 좌표 저장용 딕셔너리 생성 및 이미지 읽기
#
# 무엇을 하나요?
# - 이미지별 ROI 좌표를 저장할 빈 딕셔너리를 만듭니다.
# - 제품 이미지를 한 장씩 읽습니다.
# - 이미지 읽기에 실패하면 오류를 발생시킵니다.
#
# roi_by_file:
# 이미지 파일명을 Key로 사용하여
# 해당 이미지의 ROI 좌표를 저장할 딕셔너리
#
# 예:
# {
#   "0637145699424_side.png": [434, 294, 529, 318]
# }
#
# for image_path in IMAGE_PATHS:
# 이미지 목록을 한 장씩 반복하는 코드입니다.
#
# cv2.imread():
# 이미지 파일을 읽는 함수입니다.
#
# 왜 필요한가요?
# 이미지마다 SN의 위치가 다르므로
# 각 이미지의 ROI 좌표를 따로 저장해야 하기 때문입니다.
# --------------------------------------------------

roi_by_file = {}


for image_path in IMAGE_PATHS:

    image = cv2.imread(
        str(image_path)
    )

    if image is None:
        raise FileNotFoundError(
            image_path
        )


    # --------------------------------------------------
    # 6. 제품 이미지를 열고 마우스로 SN ROI 선택
    #
    # 무엇을 하나요?
    # - 현재 검사할 이미지 이름을 출력합니다.
    # - SN 숫자 영역만 선택하라는 안내를 출력합니다.
    # - OpenCV 창을 열어 마우스로 ROI를 선택합니다.
    #
    # cv2.selectROI():
    # 이미지 위에서 마우스로 사각형 영역을 선택하는 함수
    #
    # x:
    # 선택 영역의 왼쪽 위 X좌표
    #
    # y:
    # 선택 영역의 왼쪽 위 Y좌표
    #
    # w:
    # 선택 영역의 너비(Width)
    #
    # h:
    # 선택 영역의 높이(Height)
    #
    # fromCenter=False:
    # 중심점이 아니라 모서리 기준으로 드래그합니다.
    #
    # showCrosshair=True:
    # ROI를 선택할 때 십자선을 표시합니다.
    #
    # 왜 필요한가요?
    # 전체 제품 이미지에서 필요한 13자리 SN 숫자만
    # OCR에 입력하도록 영역을 제한하기 위해서입니다.
    #
    # 선택 방법:
    # ① 마우스로 13자리 숫자 영역을 드래그합니다.
    # ② ENTER 또는 SPACE를 누릅니다.
    #
    # 주의:
    # 가능한 한 "SN:" 문자는 제외하고
    # 13자리 숫자가 모두 포함되도록 선택합니다.
    # --------------------------------------------------

    print()
    print(
        "Image:",
        image_path.name
    )

    print(
        "SN의 13자리 숫자 영역만 선택한 후 "
        "ENTER 또는 SPACE를 누르세요."
    )


    x, y, w, h = cv2.selectROI(
        f"Select SN ROI - {image_path.name}",
        image,
        fromCenter=False,
        showCrosshair=True,
    )

    # ROI 선택 창 닫기
    cv2.destroyAllWindows()


    # --------------------------------------------------
    # 7. ROI 선택 여부 확인 및 좌표 변환
    #
    # 무엇을 하나요?
    # - 마우스로 영역을 정상적으로 선택했는지 확인합니다.
    # - 선택 영역의 너비 또는 높이가 0이면 중단합니다.
    # - x, y, w, h를 x1, y1, x2, y2로 변환합니다.
    #
    # 선택할 때 받은 값:
    #
    # x = 시작 X좌표
    # y = 시작 Y좌표
    # w = 너비
    # h = 높이
    #
    # Crop할 때 사용할 값:
    #
    # x1 = 왼쪽 X좌표
    # y1 = 위쪽 Y좌표
    # x2 = 오른쪽 X좌표
    # y2 = 아래쪽 Y좌표
    #
    # 계산 방법:
    #
    # x1 = x
    # y1 = y
    # x2 = x + w
    # y2 = y + h
    #
    # 왜 필요한가요?
    # NumPy 이미지 배열을 자를 때
    # 시작 위치와 끝 위치가 필요하기 때문입니다.
    # --------------------------------------------------

    if w == 0 or h == 0:
        raise RuntimeError(
            f"ROI가 선택되지 않았습니다: "
            f"{image_path.name}"
        )


    x1 = int(x)
    y1 = int(y)
    x2 = int(x + w)
    y2 = int(y + h)


    # --------------------------------------------------
    # 8. 선택한 SN 영역만 ROI Crop
    #
    # 무엇을 하나요?
    # - 원본 이미지에서 선택한 영역만 잘라냅니다.
    # - 13자리 SN 숫자 부분의 이미지를 만듭니다.
    #
    # image[y1:y2, x1:x2]:
    # 이미지 배열에서 지정한 범위만 추출합니다.
    #
    # 주의:
    # NumPy 이미지 배열의 인덱스 순서는
    # [Y좌표, X좌표]입니다.
    #
    # 즉:
    # [세로 범위, 가로 범위]
    #
    # 왜 필요한가요?
    # Tesseract가 전체 제품 이미지가 아니라
    # 실제로 검사해야 하는 숫자 부분에만
    # 집중하도록 하기 위해서입니다.
    #
    # 현재는 아직 OCR을 실행하지 않습니다.
    # --------------------------------------------------

    roi = image[
        y1:y2,
        x1:x2,
    ]


    # --------------------------------------------------
    # 9. 원본 이미지에 ROI 위치 사각형 표시
    #
    # 무엇을 하나요?
    # - 원본 이미지의 복사본을 만듭니다.
    # - 선택한 SN 영역을 초록색 사각형으로 표시합니다.
    #
    # image.copy():
    # 원본 이미지를 변경하지 않도록 복사합니다.
    #
    # cv2.rectangle():
    # 이미지 위에 사각형을 그리는 함수입니다.
    #
    # (x1, y1):
    # 사각형의 왼쪽 위 좌표
    #
    # (x2, y2):
    # 사각형의 오른쪽 아래 좌표
    #
    # (0, 255, 0):
    # OpenCV의 BGR 기준 초록색
    #
    # 3:
    # 사각형 선의 두께
    #
    # 왜 필요한가요?
    # 학생들이 실제 선택한 ROI 위치가
    # 올바른지 결과 이미지를 보고 확인하기 위해서입니다.
    # --------------------------------------------------

    preview = image.copy()

    cv2.rectangle(
        preview,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        3,
    )


    # --------------------------------------------------
    # 10. ROI Preview와 실제 SN Crop 이미지 저장
    #
    # 무엇을 하나요?
    # - 두 종류의 결과 이미지 경로를 만듭니다.
    # - 선택 위치가 표시된 Preview 이미지를 저장합니다.
    # - 숫자 영역만 잘라낸 SN ROI 이미지를 저장합니다.
    #
    # image_path.stem:
    # 파일의 확장자를 제외한 이름
    #
    # 예:
    # 0637145699424_side.png
    #
    # stem:
    # 0637145699424_side
    #
    # 저장 결과:
    #
    # *_roi_preview.jpg
    # → 원본 이미지에 ROI 사각형을 표시한 이미지
    #
    # *_sn_roi.jpg
    # → 실제 숫자 영역만 잘라낸 이미지
    #
    # cv2.imwrite():
    # 이미지를 파일로 저장하는 함수입니다.
    #
    # 왜 두 장을 저장하나요?
    #
    # Preview:
    # ROI 위치를 확인하기 위해서입니다.
    #
    # SN ROI:
    # 다음 전처리 및 Tesseract OCR 실습에 사용합니다.
    # --------------------------------------------------

    preview_path = (
        OUTPUT_DIR
        / f"{image_path.stem}_roi_preview.jpg"
    )

    roi_path = (
        OUTPUT_DIR
        / f"{image_path.stem}_sn_roi.jpg"
    )


    cv2.imwrite(
        str(preview_path),
        preview,
    )

    cv2.imwrite(
        str(roi_path),
        roi,
    )


    # --------------------------------------------------
    # 11. 이미지별 ROI 좌표를 딕셔너리에 기록
    #
    # 무엇을 하나요?
    # - 현재 이미지 파일명을 Key로 사용합니다.
    # - 선택한 ROI의 네 좌표를 리스트로 저장합니다.
    # - 저장된 좌표를 터미널에 출력합니다.
    #
    # 예:
    #
    # "0637145699424_side.png":
    #     [434, 294, 529, 318]
    #
    # 좌표 순서:
    # [x1, y1, x2, y2]
    #
    # 왜 필요한가요?
    # 이후 Batch OCR에서는 같은 이미지를 다시 열어
    # 마우스로 영역을 선택하지 않고,
    # 저장된 좌표를 읽어 자동 Crop하기 위해서입니다.
    #
    # 이 작업은 모든 제품 이미지에 반복됩니다.
    # --------------------------------------------------

    roi_by_file[
        image_path.name
    ] = [
        x1,
        y1,
        x2,
        y2,
    ]


    print(
        "ROI:",
        roi_by_file[
            image_path.name
        ],
    )


# --------------------------------------------------
# 12. Day06 OCR Rule 초안(Draft) 생성
#
# 무엇을 하나요?
# - 앞에서 선택한 이미지별 ROI 좌표를 저장합니다.
# - 앞으로 OCR에서 사용할 기본 설정을 준비합니다.
# - 아직 전처리를 확정하지 않은 Draft 상태로 만듭니다.
#
# baseline_version:
# 현재 OCR Rule의 버전
#
# day06-ocr-draft:
# 아직 최종 확정되지 않은 초안이라는 뜻
#
# reference_file:
# 대표 OCR 실험에 사용할 첫 번째 이미지
#
# roi_by_file:
# 이미지별 SN ROI 좌표
#
# preprocess_mode:
# 아직 전처리 방법을 선택하지 않았으므로
# NOT_SELECTED로 설정
#
# clahe_clip_limit:
# CLAHE에서 대비 강조 정도를 제한하는 값
#
# clahe_grid:
# CLAHE에서 이미지를 나누는 격자 크기
#
# ocr_lang:
# OCR에 사용할 언어 데이터
#
# ocr_oem:
# Tesseract OCR Engine Mode
#
# ocr_psm:
# Tesseract Page Segmentation Mode
# 7은 한 줄짜리 문자로 인식하도록 지정
#
# whitelist:
# 허용할 문자 종류
# 이번 SN 실습에서는 숫자 0~9만 허용
#
# expected_pattern:
# 읽은 문자열이 정확히 숫자 13자리인지 검사하는 Regex
#
# allow_context_correction:
# OCR이 인식한 숫자를 자동으로 다른 숫자로 변경할지 여부
# 이번 실습에서는 False
#
# 왜 필요한가요?
# 다음 전처리 비교와 OCR 실험에서
# 공통으로 사용할 설정을 준비하기 위해서입니다.
#
# 주의:
# 이 단계에서는 아직 최종 OCR Rule이 아닙니다.
# 이후 실험에서 전처리 방법을 결정합니다.
# --------------------------------------------------

rules = {
    "baseline_version":
        "day06-ocr-draft",

    "reference_file":
        IMAGE_PATHS[0].name,

    "roi_by_file":
        roi_by_file,

    "preprocess_mode":
        "NOT_SELECTED",

    "clahe_clip_limit":
        2.0,

    "clahe_grid":
        [8, 8],

    "ocr_lang":
        "eng",

    "ocr_oem":
        3,

    "ocr_psm":
        7,

    "whitelist":
        "0123456789",

    "expected_pattern":
        r"^[0-9]{13}$",

    "allow_context_correction":
        False,
}


# --------------------------------------------------
# 13. OCR Rule 초안을 JSON 파일로 저장
#
# 무엇을 하나요?
# - configs 폴더가 있는지 확인합니다.
# - 폴더가 없으면 생성합니다.
# - rules 딕셔너리를 JSON 문자열로 변환합니다.
# - JSON 파일로 저장합니다.
#
# json.dumps():
# Python 딕셔너리를 JSON 문자열로 변환합니다.
#
# ensure_ascii=False:
# 한글을 읽을 수 있는 형태로 저장합니다.
#
# indent=2:
# 들여쓰기를 적용하여 내용을 읽기 쉽게 만듭니다.
#
# write_text():
# JSON 문자열을 파일로 저장합니다.
#
# 저장 파일:
# configs/day06_ocr_rules.json
#
# 왜 필요한가요?
# 이후 실습에서 같은 ROI 좌표와 OCR 설정을
# 반복하여 사용할 수 있도록 하기 위해서입니다.
#
# 주의:
# 같은 파일이 이미 있으면 기존 내용을 덮어씁니다.
# --------------------------------------------------

CONFIG_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

CONFIG_PATH.write_text(
    json.dumps(
        rules,
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)


# --------------------------------------------------
# 14. ROI 선택 및 OCR Rule 저장 완료 메시지 출력
#
# 무엇을 하나요?
# - JSON 파일 저장 경로를 출력합니다.
# - ROI 준비 작업이 완료되었다는 메시지를 출력합니다.
#
# 왜 필요한가요?
# 다음 실습에서 사용할 입력 데이터와
# OCR 설정 파일이 준비되었음을 확인하기 위해서입니다.
#
# 다음 실습:
# 42_day06_preprocess_compare.py
#
# 대표 이미지의 SN ROI에
# Original / Gray / CLAHE / Otsu / CLAHE+Otsu
# 다섯 가지 전처리를 적용합니다.
# --------------------------------------------------

print()
print(
    "Saved:",
    CONFIG_PATH
)

print(
    "Day06 SN ROI Setup Complete"
)