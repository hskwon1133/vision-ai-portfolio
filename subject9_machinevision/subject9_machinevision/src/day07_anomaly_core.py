
# ==================================================
# ① 필요한 라이브러리 불러오기
#    → 이미지 처리, 딥러닝 모델, 수치 계산에 사용
# ==================================================
import cv2
import numpy as np
import torch
import torch.nn.functional as F

from PIL import Image
from torchvision.models import (
    resnet18,
    ResNet18_Weights,
)


# ==================================================
# ② 실행 장치 설정
#    → GPU(CUDA)가 있으면 GPU, 없으면 CPU 사용
# ==================================================
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ==================================================
# ③ 사전학습 Weight와 이미지 전처리 설정
#    → ImageNet으로 학습된 ResNet18 가중치 사용
#    → 모델 입력에 맞는 이미지 전처리 방법 준비
# ==================================================
WEIGHTS = ResNet18_Weights.DEFAULT

PREPROCESS = WEIGHTS.transforms()


# ==================================================
# ④ Feature 추출용 ResNet18 모델 생성
#    → 마지막 분류 기능은 제외
#    → layer3까지 사용하여 이미지 특징 추출
# ==================================================
def create_backbone():

    # 사전학습된 ResNet18 불러오기
    model = resnet18(
        weights=WEIGHTS
    )

    # ResNet18의 layer3까지 사용
    # 마지막 분류 결과는 사용하지 않음
    feature_extractor = torch.nn.Sequential(
        *list(model.children())[:-3]
    )

    # 학습 모드가 아닌 평가 모드로 설정
    feature_extractor.eval()

    # GPU 또는 CPU로 모델 이동
    feature_extractor.to(DEVICE)

    return feature_extractor


# ==================================================
# ⑤ 이미지를 ResNet18 입력 형식으로 변환
#    → 이미지 읽기 → RGB 변환 → 전처리
#    → Tensor를 GPU 또는 CPU로 이동
# ==================================================
def load_image_tensor(image_path):

    # 이미지 파일을 읽고 RGB로 변환
    image = Image.open(
        image_path
    ).convert("RGB")

    # ResNet18 입력 규격에 맞게 전처리
    # unsqueeze(0): 배치 차원 추가
    tensor = PREPROCESS(
        image
    ).unsqueeze(0)

    return tensor.to(DEVICE)


# ==================================================
# ⑥ ResNet18으로 Feature 추출
#    → Feature Vector: 이미지 전체 특징
#    → Feature Map: 이미지의 위치별 특징
# ==================================================
@torch.no_grad()
def extract_features(
    model,
    image_path,
):

    # ⑤에서 만든 함수로 이미지 전처리
    tensor = load_image_tensor(
        image_path
    )

    # ResNet18으로 Feature Map 추출
    feature_map = model(
        tensor
    )

    # ----------------------------------------------
    # ⑥-1. 이미지 전체 Feature Vector 생성
    #       → Feature Map을 Pooling하여 요약
    #       → 이미지 전체 Anomaly Score에 사용
    # ----------------------------------------------
    pooled = F.adaptive_avg_pool2d(
        feature_map,
        (1, 1),
    ).flatten(1)

    # Feature Vector 정규화
    pooled = F.normalize(
        pooled,
        dim=1,
    )

    # ----------------------------------------------
    # ⑥-2. 위치별 Feature Map 정규화
    #       → 위치별 차이와 Heatmap 계산에 사용
    # ----------------------------------------------
    normalized_map = F.normalize(
        feature_map,
        dim=1,
    )

    # ----------------------------------------------
    # ⑥-3. Feature를 NumPy 배열로 변환
    #       → 이후 평균 및 거리 계산에 사용
    # ----------------------------------------------
    vector = (
        pooled[0]
        .cpu()
        .numpy()
    )

    spatial_map = (
        normalized_map[0]
        .cpu()
        .numpy()
    )

    return vector, spatial_map


# ==================================================
# ⑦ 이미지 전체 Anomaly Score 계산
#    → 검사 이미지와 정상 기준의 Feature 비교
#    → 유클리드 거리로 차이를 숫자로 계산
# ==================================================
def anomaly_score(
    feature_vector,
    reference_vector,
):

    # 검사 Feature - 정상 기준 Feature
    difference = (
        feature_vector
        - reference_vector
    )

    # 유클리드 거리 계산
    score = np.linalg.norm(
        difference
    )

    # 거리값을 Anomaly Score로 반환
    return float(score)


# ==================================================
# ⑧ 위치별 Anomaly 차이 계산
#    → 검사 Feature Map과 정상 기준 Map 비교
#    → 위치마다 정상과 얼마나 다른지 계산
# ==================================================
def anomaly_map(
    spatial_feature,
    reference_map,
):

    # 검사 이미지와 정상 기준의 위치별 차이
    difference = (
        spatial_feature
        - reference_map
    )

    # 각 위치에서 Feature 차이의 크기 계산
    score_map = np.sqrt(
        np.sum(
            difference ** 2,
            axis=0,
        )
    )

    return score_map


# ==================================================
# ⑨ Heatmap과 Overlay 이미지 생성
#    → 정상과 차이가 큰 위치를 색으로 표시
#    → 원본 이미지 위에 Heatmap을 겹쳐 표시
# ==================================================
def make_heatmap_overlay(
    image_path,
    score_map,
):

    # 원본 이미지 읽기
    image = cv2.imread(
        str(image_path)
    )

    if image is None:
        raise FileNotFoundError(
            image_path
        )

    # 원본 이미지 크기 확인
    height, width = image.shape[:2]

    # ----------------------------------------------
    # ⑨-1. Score Map을 원본 이미지 크기로 확대
    # ----------------------------------------------
    resized_map = cv2.resize(
        score_map,
        (width, height),
        interpolation=cv2.INTER_CUBIC,
    )

    # ----------------------------------------------
    # ⑨-2. Score 값을 0~255 범위로 변환
    # ----------------------------------------------
    normalized = cv2.normalize(
        resized_map,
        None,
        0,
        255,
        cv2.NORM_MINMAX,
    )

    heatmap_uint8 = (
        normalized.astype(
            np.uint8
        )
    )

    # ----------------------------------------------
    # ⑨-3. 차이값을 색상으로 표현
    #       → Heatmap 생성
    # ----------------------------------------------
    heatmap_color = cv2.applyColorMap(
        heatmap_uint8,
        cv2.COLORMAP_JET,
    )

    # ----------------------------------------------
    # ⑨-4. 원본 이미지와 Heatmap 합성
    #       → Overlay 생성
    # ----------------------------------------------
    overlay = cv2.addWeighted(
        image,
        0.65,
        heatmap_color,
        0.35,
        0,
    )

    return (
        heatmap_uint8,
        heatmap_color,
        overlay,
    )


# ==================================================
# ⑩ 저장된 Normal Reference 불러오기
#    → 이전 실습에서 만든 정상 기준을 읽음
#    → reference_vector / reference_map 반환
# ==================================================
def load_reference(
    reference_path,
):

    # .npz 파일에 저장된 정상 특징 읽기
    data = np.load(
        reference_path
    )

    # 이미지 전체 기준 + 위치별 기준 반환
    return (
        data["reference_vector"],
        data["reference_map"],
    )