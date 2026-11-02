
# 지역 대비 개선(CLAHE) 실습
from pathlib import Path

import cv2


# --------------------------------------------------
# 1. 사용할 이미지와 결과 저장 경로를 정한다.
# --------------------------------------------------

reference_path = Path(
    "samples/day02/cap_on_reference.jpg"
)

low_contrast_path = Path(
    "samples/day02/low_contrast.jpg"
)

output_dir = Path(
    "outputs/day02/clahe"
)


# --------------------------------------------------
# 2. CLAHE 결과를 저장할 폴더를 준비한다.
# --------------------------------------------------

output_dir.mkdir(
    parents=True,
    exist_ok=True,
)


# --------------------------------------------------
# 3. 기준 ROI 이미지를 읽는다.
# --------------------------------------------------

reference = cv2.imread(
    str(reference_path)
)


# --------------------------------------------------
# 4. 기준 이미지를 정상적으로 읽었는지 확인한다.
#
#    읽지 못했다면 FileNotFoundError를 발생시킨다.
# --------------------------------------------------

if reference is None:
    raise FileNotFoundError(
        f"기준 이미지를 읽지 못했습니다: "
        f"{reference_path}"
    )


# --------------------------------------------------
# 5. 기준 이미지의 Contrast를 낮춰
#    실습용 저대비 이미지를 만든다.
#
#    alpha=0.35 → 픽셀값의 차이를 줄인다.
#    beta=80    → 픽셀값에 80을 더한다.
#
#    병뚜껑과 배경의 밝기 차이가 줄어든다.
# --------------------------------------------------

low_contrast = cv2.convertScaleAbs(
    reference,
    alpha=0.35,
    beta=80,
)


# --------------------------------------------------
# 6. 저대비 이미지를 low_contrast.jpg로 저장한다.
# --------------------------------------------------

cv2.imwrite(
    str(low_contrast_path),
    low_contrast,
)

print(
    "문제 이미지 생성:",
    low_contrast_path,
)


# --------------------------------------------------
# 7. 기준 이미지와 저대비 이미지를 Gray로 변환한다.
#
#    컬러 BGR → 흑백 Gray
#
#    CLAHE는 이번 실습에서
#    1채널 Gray 이미지에 적용한다.
# --------------------------------------------------

reference_gray = cv2.cvtColor(
    reference,
    cv2.COLOR_BGR2GRAY,
)

low_contrast_gray = cv2.cvtColor(
    low_contrast,
    cv2.COLOR_BGR2GRAY,
)


# --------------------------------------------------
# 8. 두 Gray 이미지를 각각 저장한다.
#
#    reference_gray.jpg
#    → 기준 이미지의 Gray 결과
#
#    low_contrast_gray.jpg
#    → Contrast를 낮춘 Gray 결과
#
#    두 이미지를 비교하여 저대비 상태를 확인한다.
# --------------------------------------------------

cv2.imwrite(
    str(
        output_dir
        / "reference_gray.jpg"
    ),
    reference_gray,
)

cv2.imwrite(
    str(
        output_dir
        / "low_contrast_gray.jpg"
    ),
    low_contrast_gray,
)


# --------------------------------------------------
# 9. CLAHE에서 사용할 파라미터를 준비한다.
#
#    clipLimit → 대비 향상 강도 조절
#    tileGridSize → 이미지를 나눌 격자 수
#
#    clipLimit = 1.0, 2.0, 3.0, 4.0
#    tileGridSize = (8, 8) 고정
# --------------------------------------------------

settings = [
    (1.0, (8, 8)),
    (2.0, (8, 8)),
    (3.0, (8, 8)),
    (4.0, (8, 8)),
]


# --------------------------------------------------
# 10. CLAHE 파라미터를 하나씩 가져와 반복한다.
#
#     첫 번째 반복:
#     clip_limit = 1.0
#     tile_size = (8, 8)
#
#     이후 2.0 → 3.0 → 4.0 순서로 반복한다.
# --------------------------------------------------

for clip_limit, tile_size in settings:

    # --------------------------------------------------
    # 11. 현재 파라미터로 CLAHE 객체를 생성한다.
    #
    #     clipLimit → 대비 향상 강도 설정
    #     tileGridSize → 지역 격자 수 설정
    # --------------------------------------------------

    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=tile_size,
    )


    # --------------------------------------------------
    # 12. 저대비 Gray 이미지에 CLAHE를 적용한다.
    #
    #     지역별 밝기 분포를 조절하여
    #     병뚜껑의 경계가 잘 구분되는지 확인한다.
    # --------------------------------------------------

    result = clahe.apply(
        low_contrast_gray
    )


    # --------------------------------------------------
    # 13. 적용한 clipLimit 값을 이용해
    #     결과 이미지의 파일명을 만든다.
    #
    #     예: clahe_clip_1.0.jpg
    # --------------------------------------------------

    output_path = (
        output_dir
        / f"clahe_clip_{clip_limit}.jpg"
    )


    # --------------------------------------------------
    # 14. CLAHE 결과 이미지를 저장한다.
    #
    #     저장 후 다음 clipLimit 값으로
    #     11~14번 과정을 반복한다.
    # --------------------------------------------------

    cv2.imwrite(
        str(output_path),
        result,
    )

    print(
        "저장:",
        output_path,
    )


# --------------------------------------------------
# 15. 모든 결과 이미지를 직접 비교한다.
#
#     reference_gray.jpg
#     low_contrast_gray.jpg
#     clahe_clip_1.0.jpg
#     clahe_clip_2.0.jpg
#     clahe_clip_3.0.jpg
#     clahe_clip_4.0.jpg
#
#     병뚜껑의 세로 홈과 외곽 경계가 잘 보이는가?
#     Noise가 지나치게 강조되지는 않았는가?
#
#     학생이 결과를 직접 관찰하고 분석한다.
# --------------------------------------------------