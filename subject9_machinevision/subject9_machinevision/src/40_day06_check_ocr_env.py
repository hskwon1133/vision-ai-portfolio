from pathlib import Path
import platform
import shutil
import subprocess

import cv2
import pytesseract


# --------------------------------------------------
# 1. OCR 환경 확인에 필요한 라이브러리 준비
#
# 무엇을 하나요?
# - Python 표준 라이브러리와 OCR 관련 라이브러리를
#   불러옵니다.
#
# Path:
# 파일과 폴더 경로를 관리합니다.
#
# **platform:
# 현재 운영체제 정보를 확인합니다.
#
# **shutil:
# Tesseract 실행 파일의 위치를 찾습니다.
#
# **subprocess:
# Python에서 터미널 명령어를 실행합니다.
#
# cv2:
# 현재 설치된 OpenCV 버전을 확인합니다.
#
# **pytesseract:
# Python에서 Tesseract OCR Engine을 연결하고
# 버전 정보를 확인합니다.
#
# 왜 필요한가요?
# OCR 실습을 시작하기 전에 필요한 프로그램들이
# 설치되어 있는지 확인하기 위해서입니다.
#
# 참고:
# 위 import 문에서 필요한 라이브러리를 불러옵니다.
# --------------------------------------------------


# --------------------------------------------------
# 2. 프로젝트 최상위 폴더와 환경 보고서 경로 설정
#
# 무엇을 하나요?
# - 현재 실행 중인 Python 파일의 위치를 찾습니다.
# - src 폴더의 상위 폴더를 프로젝트 Root로 지정합니다.
# - 환경 확인 결과를 저장할 TXT 파일 경로를 만듭니다.
#
# __file__:
# 현재 실행 중인 Python 파일의 경로입니다.
#
# .resolve():
# 파일의 절대경로를 구합니다.
#
# .parent:
# 현재 경로의 상위 폴더로 이동합니다.
#
# .parent.parent:
# src 폴더에서 프로젝트 최상위 폴더로 이동합니다.
#
# 예:
# subject9_machinevision/
# ├── src/
# │   └── 40_day06_check_ocr_env.py
# └── reports/
#     └── day06_ocr_environment.txt
#
# 왜 필요한가요?
# 프로그램을 어느 위치에서 실행하더라도
# 프로젝트의 reports 폴더에 결과를 저장하기 위해서입니다.
# --------------------------------------------------

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

REPORT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "day06_ocr_environment.txt"
)


# --------------------------------------------------
# 3. 현재 운영체제와 OpenCV 버전 확인
#
# 무엇을 하나요?
# - 현재 Python이 실행되는 운영체제 정보를 확인합니다.
# - 설치된 OpenCV 버전을 확인합니다.
# - 확인한 내용을 lines 리스트에 저장합니다.
#
# platform.platform():
# Linux, Windows 등 현재 운영체제의 정보를 반환합니다.
#
# cv2.__version__:
# 현재 설치된 OpenCV 버전을 반환합니다.
#
# lines:
# 환경 확인 결과를 한 줄씩 저장할 리스트입니다.
#
# 왜 필요한가요?
# 학생마다 실행환경이나 라이브러리 버전이 다를 수 있으므로
# 현재 환경 정보를 먼저 기록하기 위해서입니다.
# --------------------------------------------------

lines = [
    f"OS: {platform.platform()}",
    f"OpenCV: {cv2.__version__}",
]


# --------------------------------------------------
# 4. Tesseract OCR Engine 실행 파일 찾기
#
# 무엇을 하나요?
# - 현재 운영체제에서 tesseract 명령어를 찾습니다.
# - 설치된 Tesseract 실행 파일의 경로를 확인합니다.
#
# shutil.which("tesseract"):
# 시스템의 PATH 환경변수에 등록된 폴더에서
# tesseract 실행 파일을 검색합니다.
#
# Tesseract를 찾으면:
# 실행 파일의 경로를 반환합니다.
#
# Tesseract를 찾지 못하면:
# None을 반환합니다.
#
# 예:
# /usr/bin/tesseract
#
# 왜 필요한가요?
# pytesseract는 실제 OCR Engine이 아니라
# Python에서 Tesseract를 호출하는 연결 라이브러리입니다.
#
# 따라서 Tesseract 실행 프로그램이
# 별도로 설치되어 있어야 합니다.
# --------------------------------------------------

tesseract_path = shutil.which(
    "tesseract"
)


# --------------------------------------------------
# 5. Tesseract를 찾지 못한 경우 오류 정보 기록
#
# 무엇을 하나요?
# - Tesseract 실행 파일을 찾았는지 확인합니다.
# - 파일이 없다면 NOT FOUND 메시지를 기록합니다.
#
# tesseract_path is None:
# Tesseract 실행 파일을 찾지 못했다는 의미입니다.
#
# lines.append():
# 환경 확인 결과 리스트에 새로운 내용을 추가합니다.
#
# 왜 필요한가요?
# OCR이 실행되지 않는 원인이
# Tesseract 설치 또는 PATH 설정 문제인지
# 구분하기 위해서입니다.
#
# 주의:
# 이 경우에도 환경 보고서는 저장합니다.
# --------------------------------------------------

if tesseract_path is None:

    lines.append(
        "Tesseract: NOT FOUND"
    )

    lines.append(
        "Tesseract Path: NOT FOUND"
    )


else:

    # --------------------------------------------------
    # 6. Tesseract 설치 경로와 버전 확인
    #
    # 무엇을 하나요?
    # - Tesseract 실행 파일을 찾았다면 경로를 기록합니다.
    # - pytesseract로 실제 Tesseract 버전을 확인합니다.
    #
    # pytesseract.get_tesseract_version():
    # 설치된 Tesseract OCR Engine의 버전을 확인합니다.
    #
    # try:
    # 버전 확인을 시도합니다.
    #
    # except:
    # 버전 확인 중 오류가 발생하면 오류를 기록합니다.
    #
    # 왜 필요한가요?
    # Tesseract 실행 파일이 존재하는지 확인하는 것과
    # 실제로 정상 실행되는지 확인하는 것은 다르기 때문입니다.
    # --------------------------------------------------

    lines.append(
        f"Tesseract Path: {tesseract_path}"
    )

    try:

        version = (
            pytesseract
            .get_tesseract_version()
        )

        lines.append(
            f"Tesseract: {version}"
        )

    except Exception as error:

        lines.append(
            f"Tesseract Version Error: {error}"
        )


    # --------------------------------------------------
    # 7. 설치된 OCR 언어 데이터 확인
    #
    # 무엇을 하나요?
    # - Tesseract가 사용할 수 있는 언어 목록을 확인합니다.
    # - tesseract --list-langs 명령어를 실행합니다.
    #
    # subprocess.run():
    # Python 코드에서 외부 명령어를 실행하는 함수입니다.
    #
    # capture_output=True:
    # 명령어의 출력 결과를 Python에서 받아옵니다.
    #
    # text=True:
    # 출력 결과를 문자열로 처리합니다.
    #
    # check=True:
    # 명령어가 실패하면 예외를 발생시킵니다.
    #
    # result.stdout:
    # 명령어의 표준 출력 결과입니다.
    #
    # strip():
    # 출력 결과의 앞뒤 공백과 줄바꿈을 제거합니다.
    #
    # 예:
    # eng → 영어 OCR
    # kor → 한국어 OCR
    # osd → 문자 방향 및 스크립트 감지
    #
    # 왜 필요한가요?
    # Tesseract Engine이 설치되어 있어도
    # 사용할 OCR 언어 데이터가 없다면
    # 문자를 정상적으로 인식하지 못할 수 있기 때문입니다.
    #
    # 이번 6일차 실습:
    # 숫자 13자리 SN을 검사하며 eng를 사용합니다.
    # --------------------------------------------------

    try:

        result = subprocess.run(
            [
                "tesseract",
                "--list-langs",
            ],
            capture_output=True,
            text=True,
            check=True,
        )

        lines.append(
            "Languages:"
        )

        lines.append(
            result.stdout.strip()
        )

    except Exception as error:

        lines.append(
            f"Language Check Error: {error}"
        )


# --------------------------------------------------
# 8. OCR 환경 확인 결과를 TXT 파일로 저장
#
# 무엇을 하나요?
# - reports 폴더가 있는지 확인합니다.
# - 폴더가 없으면 생성합니다.
# - 지금까지 확인한 환경 정보를 TXT 파일로 저장합니다.
#
# REPORT_PATH.parent.mkdir():
# 결과 파일의 상위 폴더를 생성합니다.
#
# parents=True:
# 필요한 상위 폴더까지 함께 생성합니다.
#
# exist_ok=True:
# 폴더가 이미 있으면 그대로 사용합니다.
#
# "\n".join(lines):
# lines에 저장한 결과를 줄바꿈으로 연결합니다.
#
# write_text():
# 문자열을 TXT 파일에 저장합니다.
#
# encoding="utf-8":
# 한글을 포함한 문자열도 저장할 수 있도록 설정합니다.
#
# 왜 필요한가요?
# 현재 OCR 개발환경을 파일로 기록하여
# 나중에 환경을 재현하거나 오류를 분석할 때
# 참고하기 위해서입니다.
# --------------------------------------------------

REPORT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

REPORT_PATH.write_text(
    "\n".join(lines),
    encoding="utf-8",
)


# --------------------------------------------------
# 9. 확인한 환경 정보와 저장 경로 출력
#
# 무엇을 하나요?
# - lines에 저장된 환경 확인 결과를 출력합니다.
# - TXT 보고서가 저장된 위치를 출력합니다.
#
# 왜 필요한가요?
# 파일을 직접 열지 않아도
# 터미널에서 현재 OCR 환경을 확인하기 위해서입니다.
#
# 출력 내용:
# OS
# OpenCV 버전
# Tesseract 설치 경로
# Tesseract 버전
# 설치된 OCR 언어
# 보고서 저장 경로
# --------------------------------------------------

print(
    "\n".join(lines)
)

print()

print(
    "Saved:",
    REPORT_PATH
)


# --------------------------------------------------
# 10. Tesseract 발견 여부에 따라 최종 메시지 출력
#
# 무엇을 하나요?
# - 4번에서 찾은 Tesseract 실행 파일 경로를 확인합니다.
# - 찾지 못했다면 Failed 메시지를 출력합니다.
# - 찾았다면 OK 메시지를 출력합니다.
#
# OCR Environment Check Failed:
# Tesseract 실행 파일을 찾지 못한 경우
#
# OCR Environment OK:
# Tesseract 실행 파일을 찾은 경우
#
# 주의:
# 현재 코드는 실행 파일의 존재 여부를 기준으로
# 최종 메시지를 결정합니다.
#
# 따라서 OCR Environment OK가 출력되어도
# Tesseract 버전이나 언어 확인에서 오류가 있었다면
# 실제 OCR 실습 전에 추가 확인이 필요합니다.
# --------------------------------------------------

if tesseract_path is None:

    print(
        "OCR Environment Check Failed"
    )

else:

    print(
        "OCR Environment OK"
    )