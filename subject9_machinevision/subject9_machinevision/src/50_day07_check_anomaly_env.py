
# ==================================================
# ① 필요한 라이브러리 불러오기
#    → 폴더 확인과 AI 실행환경 점검에 사용
# ==================================================
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision


# ==================================================
# ② MVTec bottle 원본 데이터 경로 설정
#    → 오늘 검사할 병 이미지가 있는 위치
# ==================================================
ROOT = Path(
    "samples/day07/original/bottle"
)


# ==================================================
# ③ 반드시 있어야 하는 Dataset 폴더 목록
#    → 정상 / 이상 / 실제 결함 위치 폴더 확인용
# ==================================================
required_dirs = [
    # 정상 학습 이미지
    ROOT / "train/good",

    # 정상 및 이상 Test 이미지
    ROOT / "test/good",
    ROOT / "test/broken_large",
    ROOT / "test/broken_small",
    ROOT / "test/contamination",

    # 실제 결함 위치를 표시한 Mask
    ROOT / "ground_truth/broken_large",
    ROOT / "ground_truth/broken_small",
    ROOT / "ground_truth/contamination",
]


# ==================================================
# ④ 프로그램 시작 제목 출력
#    → 7일차 환경 점검이 시작되었음을 표시
# ==================================================
print("=" * 70)
print("DAY07 ANOMALY ENV CHECK")
print("=" * 70)


# ==================================================
# ⑤ AI 라이브러리 버전과 GPU 사용 여부 확인
#    → PyTorch / OpenCV 등이 정상 설치되었는지 확인
# ==================================================
print("Torch      :", torch.__version__)
print("Torchvision:", torchvision.__version__)
print("OpenCV     :", cv2.__version__)
print("NumPy      :", np.__version__)
print("CUDA       :", torch.cuda.is_available())

# CUDA를 사용할 수 있다면 GPU 이름 출력
if torch.cuda.is_available():
    print(
        "GPU        :",
        torch.cuda.get_device_name(0),
    )


# ==================================================
# ⑥ Dataset 폴드가 실제로 존재하는지 확인
#    → required_dirs의 모든 경로를 하나씩 검사
# ==================================================
print()
print("Dataset:", ROOT)

# 처음에는 모든 폴더가 정상이라고 가정
all_ok = True

for directory in required_dirs:

    # 현재 폴더가 존재하면 True, 없으면 False
    exists = directory.exists()

    # 하나라도 없으면 전체 점검 실패로 표시
    if not exists:
        all_ok = False

    # 폴더별 확인 결과 출력
    print(
        f"[{'OK' if exists else 'MISSING'}]",
        directory,
    )


# ==================================================
# ⑦ 필요한 폴더가 없으면 오류 발생
#    → Dataset이 불완전한 상태에서 실습 진행 방지
# ==================================================
if not all_ok:
    raise FileNotFoundError(
        "MVTec bottle 폴더 구조를 다시 확인하세요."
    )


# ==================================================
# ⑧ 정상 / 이상 이미지 수 확인
#    → 각 폴더의 PNG 이미지 개수를 출력
# ==================================================
print()

# 정상 학습 이미지 수
print(
    "Train Good        :",
    len(
        list(
            (ROOT / "train/good").glob("*.png")
        )
    ),
)

# 정상 Test 이미지 수
print(
    "Test Good         :",
    len(
        list(
            (ROOT / "test/good").glob("*.png")
        )
    ),
)

# 크게 파손된 병 이미지 수
print(
    "Broken Large      :",
    len(
        list(
            (ROOT / "test/broken_large").glob("*.png")
        )
    ),
)

# 작게 파손된 병 이미지 수
print(
    "Broken Small      :",
    len(
        list(
            (ROOT / "test/broken_small").glob("*.png")
        )
    ),
)

# 오염된 병 이미지 수
print(
    "Contamination     :",
    len(
        list(
            (ROOT / "test/contamination").glob("*.png")
        )
    ),
)


# ==================================================
# ⑨ 환경 점검 완료 메시지 출력
#    → 필요한 폴더 확인과 이미지 수 점검 완료
# ==================================================
print()
print("Day07 Environment OK")