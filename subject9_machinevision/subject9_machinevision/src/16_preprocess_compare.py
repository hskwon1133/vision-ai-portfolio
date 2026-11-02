
#전처리 전/후 종합 비교
from pathlib import Path

import cv2


# --------------------------------------------------
# 1. 사용할 이미지와 결과 저장 경로를 정한다.
#
#    reference_path
#    → 1일차에서 만든 병뚜껑 기준 ROI 이미지
#
#    low_contrast_path
#    → 대비를 낮춘 실습용 이미지
#
#    output_dir
#    → 전처리 비교 결과를 저장할 폴더
# --------------------------------------------------

reference_path = Path(
    "samples/day02/cap_on_reference.jpg"
)

low_contrast_path = Path(
    "samples/day02/low_contrast.jpg"
)

output_dir = Path(
    "outputs/day02/compare"
)


# --------------------------------------------------
# 2. 결과 이미지를 저장할 폴더를 생성한다.
#
#    폴더가 없다면 새로 만든다.
#    이미 존재한다면 그대로 사용한다.
# --------------------------------------------------

output_dir.mkdir(
    parents=True,
    exist_ok=True,
)


# --------------------------------------------------
# 3. 저대비 이미지 low_contrast.jpg를 읽는다.
#
#    앞의 CLAHE 실습에서 만든 이미지를 재사용한다.
# --------------------------------------------------

image = cv2.imread(
    str(low_contrast_path)
)


# --------------------------------------------------
# 4. 저대비 이미지를 정상적으로 읽었는지 확인한다.
#
#    image가 None이면 5~6번을 실행한다.
#    정상적으로 읽었다면 5~6번을 건너뛴다.
# --------------------------------------------------

if image is None:

    print(
        "low_contrast.jpg가 없어 "
        "기준 이미지에서 생성합니다."
    )


    # --------------------------------------------------
    # 5. 병뚜껑 기준 ROI 이미지를 읽고 확인한다.
    #
    #    저대비 이미지를 다시 만들기 위한 원본이다.
    #    읽지 못했다면 오류를 발생시킨다.
    # --------------------------------------------------

    reference = cv2.imread(
        str(reference_path)
    )

    if reference is None:
        raise FileNotFoundError(
            f"기준 이미지를 읽지 못했습니다: "
            f"{reference_path}"
        )


    # --------------------------------------------------
    # 6. 기준 이미지의 Contrast를 낮추고 저장한다.
    #
    #    alpha=0.35 → 픽셀값의 차이를 줄인다.
    #    beta=80    → 픽셀값에 80을 더한다.
    #
    #    결과를 low_contrast.jpg로 저장한다.
    # --------------------------------------------------

    image = cv2.convertScaleAbs(
        reference,
        alpha=0.35,
        beta=80,
    )

    cv2.imwrite(
        str(low_contrast_path),
        image,
    )

    print(
        "문제 이미지 생성:",
        low_contrast_path,
    )


# --------------------------------------------------
# 7. 저대비 이미지를 Gray로 변환한다.
#
#    BGR → Gray
#
#    병뚜껑과 배경의 밝기 차이를 이용하여
#    다음 이진화 실습을 진행한다.
# --------------------------------------------------

gray = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2GRAY,
)


# --------------------------------------------------
# 8. 전처리 없이 Otsu 이진화를 적용한다.
#
#    Gray 이미지에 별도의 개선 처리를 하지 않고
#    Otsu를 바로 적용한다.
#
#    THRESH_BINARY_INV:
#    → 어두운 병뚜껑을 흰색 객체로 표현
#
#    original_value
#    → Otsu가 자동으로 선택한 임계값
#
#    original_otsu
#    → 이진화 결과 이미지
# --------------------------------------------------

original_value, original_otsu = cv2.threshold(
    gray,
    0,
    255,
    cv2.THRESH_BINARY_INV
    + cv2.THRESH_OTSU,
)


# --------------------------------------------------
# 9. 같은 Gray 이미지에 Gaussian Blur를 적용한다.
#
#    Kernel = 5×5
#
#    주변 픽셀의 가중 평균을 이용해
#    Noise를 부드럽게 줄인다.
#
#    결과를 gaussian 변수에 저장한다.
# --------------------------------------------------

gaussian = cv2.GaussianBlur(
    gray,
    (5, 5),
    0,
)


# --------------------------------------------------
# 10. Gaussian Blur 결과에 Otsu를 적용한다.
#
#     Gaussian으로 처리한 이미지를 이진화한다.
#
#     gaussian_value
#     → 자동 계산된 임계값
#
#     gaussian_otsu
#     → 이진화 결과 이미지
# --------------------------------------------------

gaussian_value, gaussian_otsu = cv2.threshold(
    gaussian,
    0,
    255,
    cv2.THRESH_BINARY_INV
    + cv2.THRESH_OTSU,
)


# --------------------------------------------------
# 11. CLAHE 객체를 생성한다.
#
#     clipLimit=2.0
#     → 지역 대비 향상 강도 설정
#
#     tileGridSize=(8, 8)
#     → 이미지를 가로 8개, 세로 8개 구역으로 나눔
#
#     CLAHE는 지역별 Contrast를 개선하는 방법이다.
# --------------------------------------------------

clahe = cv2.createCLAHE(
    clipLimit=2.0,
    tileGridSize=(8, 8),
)


# --------------------------------------------------
# 12. 같은 Gray 이미지에 CLAHE를 적용한다.
#
#     병뚜껑과 주변 배경의 지역적인
#     밝기 차이를 개선한다.
#
#     결과를 clahe_image에 저장한다.
# --------------------------------------------------

clahe_image = clahe.apply(
    gray
)


# --------------------------------------------------
# 13. CLAHE 결과에 Otsu 이진화를 적용한다.
#
#     CLAHE로 개선한 이미지에서
#     병뚜껑과 배경을 분리한다.
#
#     clahe_value
#     → 자동 계산된 임계값
#
#     clahe_otsu
#     → 이진화 결과 이미지
# --------------------------------------------------

clahe_value, clahe_otsu = cv2.threshold(
    clahe_image,
    0,
    255,
    cv2.THRESH_BINARY_INV
    + cv2.THRESH_OTSU,
)


# --------------------------------------------------
# 14. 세 가지 Otsu 임계값을 출력한다.
#
#     Original + Otsu
#     Gaussian + Otsu
#     CLAHE + Otsu
#
#     입력 이미지가 다르므로
#     Otsu가 선택한 임계값도 달라질 수 있다.
# --------------------------------------------------

print(
    "Original Otsu:",
    original_value,
)

print(
    "Gaussian Otsu:",
    gaussian_value,
)

print(
    "CLAHE Otsu:",
    clahe_value,
)


# --------------------------------------------------
# 15. 비교 화면에 사용할 이미지 크기를 정한다.
#
#     가로 = 400px
#     세로 = 300px
#
#     네 장의 이미지를 같은 크기로 표시한다.
# --------------------------------------------------

view_size = (
    400,
    300,
)


# --------------------------------------------------
# 16. 비교 화면을 만드는 make_view() 함수를 정의한다.
#
#     source    → 표시할 이미지
#     title     → 이미지 위에 표시할 제목
#     is_binary → 이진화 이미지 여부
#
#     함수 내부 처리:
#
#     ① 이미지에 맞는 크기 변경 방법을 선택한다.
#     ② 이미지를 400×300으로 변경한다.
#     ③ Gray 이미지를 화면용 BGR로 변환한다.
#     ④ 이미지 위쪽에 검정색 제목 영역을 만든다.
#     ⑤ 제목을 흰색 글자로 표시한다.
#     ⑥ 완성된 화면 이미지를 반환한다.
#
#     함수는 여기서 정의만 하며,
#     실제 실행은 17~18번에서 호출할 때 이루어진다.
# --------------------------------------------------

def make_view(
    source,
    title,
    is_binary=False,
):

    # 이진화 이미지는 픽셀값이 섞이지 않도록
    # 가장 가까운 픽셀을 사용하는 방법으로 변경
    if is_binary:
        interpolation = cv2.INTER_NEAREST
    else:
        interpolation = cv2.INTER_AREA

    # 비교 화면 크기로 변경
    resized = cv2.resize(
        source,
        view_size,
        interpolation=interpolation,
    )

    # Gray 이미지라면 화면용 BGR로 변환
    if len(resized.shape) == 2:
        canvas = cv2.cvtColor(
            resized,
            cv2.COLOR_GRAY2BGR,
        )
    else:
        canvas = resized.copy()

    # 이미지 상단에 제목 영역 생성
    cv2.rectangle(
        canvas,
        (0, 0),
        (view_size[0], 40),
        (0, 0, 0),
        -1,
    )

    # 제목 표시
    cv2.putText(
        canvas,
        title,
        (10, 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )

    return canvas


# --------------------------------------------------
# 17. 비교 화면의 위쪽 두 이미지를 만든다.
#
#     왼쪽:
#     Low Contrast Gray
#     → 저대비 Gray 이미지
#
#     오른쪽:
#     Original + Otsu
#     → 별도 전처리 없이 Otsu 적용
#
#     cv2.hconcat()
#     → 이미지를 가로 방향으로 연결한다.
# --------------------------------------------------

top = cv2.hconcat(
    [
        make_view(
            gray,
            "Low Contrast Gray",
        ),
        make_view(
            original_otsu,
            "Original + Otsu",
            is_binary=True,
        ),
    ]
)


# --------------------------------------------------
# 18. 비교 화면의 아래쪽 두 이미지를 만든다.
#
#     왼쪽:
#     Gaussian + Otsu
#
#     오른쪽:
#     CLAHE + Otsu
#
#     두 이미지를 가로로 연결한다.
# --------------------------------------------------

bottom = cv2.hconcat(
    [
        make_view(
            gaussian_otsu,
            "Gaussian + Otsu",
            is_binary=True,
        ),
        make_view(
            clahe_otsu,
            "CLAHE + Otsu",
            is_binary=True,
        ),
    ]
)


# --------------------------------------------------
# 19. 위쪽과 아래쪽 이미지를 세로로 연결한다.
#
#     cv2.vconcat()
#     → 이미지를 세로 방향으로 연결한다.
#
#     최종 2×2 비교 화면을 만든다.
#
#     ┌───────────────────┬───────────────────┐
#     │ Low Contrast Gray │ Original + Otsu   │
#     ├───────────────────┼───────────────────┤
#     │ Gaussian + Otsu   │ CLAHE + Otsu      │
#     └───────────────────┴───────────────────┘
# --------------------------------------------------

compare = cv2.vconcat(
    [
        top,
        bottom,
    ]
)


# --------------------------------------------------
# 20. 최종 비교 이미지를 저장하고 경로를 출력한다.
#
#     저장 파일:
#     outputs/day02/compare/preprocess_compare.png
#
#     저장한 이미지를 직접 열어
#     병뚜껑과 배경의 분리 결과를 비교한다.
# --------------------------------------------------

output_path = (
    output_dir
    / "preprocess_compare.png"
)

cv2.imwrite(
    str(output_path),
    compare,
)

print(
    "저장:",
    output_path,
)

