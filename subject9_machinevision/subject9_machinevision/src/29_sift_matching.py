from pathlib import Path
from time import perf_counter
import csv

import cv2


# --------------------------------------------------
# 1. 입력 이미지와 결과 저장 경로 설정
#
# 무엇을 하나요?
# - 비교 기준이 되는 Reference 이미지 위치를 지정합니다.
# - 검사할 Target 이미지 폴더를 지정합니다.
# - SIFT 비교 이미지와 CSV 결과를 저장할 위치를 준비합니다.
# - 저장 폴더가 없으면 자동으로 생성합니다.
# --------------------------------------------------

reference_path = Path(
    "samples/day04/reference/reference_label.jpg"
)

target_dir = Path(
    "samples/day04/targets"
)

output_dir = Path(
    "outputs/day04/sift"
)

report_path = Path(
    "reports/day04_sift_results.csv"
)


output_dir.mkdir(
    parents=True,
    exist_ok=True,
)

report_path.parent.mkdir(
    parents=True,
    exist_ok=True,
)


# --------------------------------------------------
# 2. 비교할 Target 이미지 4장 준비
#
# 무엇을 하나요?
# - 정상, 이동, 회전, 크기 변화 이미지를 준비합니다.
# - 각 이미지에 동일한 SIFT 알고리즘을 적용합니다.
#
# 비교 조건
# normal  : 정상 이미지
# shift   : 위치가 이동한 이미지
# rot15   : 15도 회전한 이미지
# scale90 : 크기가 90%로 변한 이미지
#
# 목적:
# 이미지의 위치, 회전, 크기가 변해도
# SIFT가 같은 특징을 찾아낼 수 있는지 확인합니다.
# --------------------------------------------------

target_settings = [
    (
        "normal",
        target_dir / "target_normal.jpg",
    ),
    (
        "shift",
        target_dir / "target_shift.jpg",
    ),
    (
        "rot15",
        target_dir / "target_rot15.jpg",
    ),
    (
        "scale90",
        target_dir / "target_scale90.jpg",
    ),
]


# --------------------------------------------------
# 3. 현재 OpenCV에서 SIFT 사용 가능 여부 확인
#
# 무엇을 하나요?
# - 설치된 OpenCV에 SIFT_create 기능이 있는지 확인합니다.
#
# 왜 필요한가요?
# OpenCV 설치 버전이나 환경에 따라
# SIFT 기능을 사용할 수 없는 경우가 있기 때문입니다.
#
# SIFT_create가 없으면 프로그램을 중단합니다.
# --------------------------------------------------

if not hasattr(
    cv2,
    "SIFT_create",
):
    raise RuntimeError(
        "현재 OpenCV에서 SIFT_create를 사용할 수 없습니다."
    )


# --------------------------------------------------
# 4. 기준 이미지(Reference)를 흑백으로 읽기
#
# 무엇을 하나요?
# - 광천김 로고의 기준 이미지를 불러옵니다.
# - SIFT 특징점을 검출하기 위해 Grayscale로 읽습니다.(모양을 찾기 때문에)
# - 이미지가 없거나 읽을 수 없으면 오류를 발생시킵니다.
#
# Reference는 이후 모든 Target과 비교할 기준입니다.
# --------------------------------------------------

reference = cv2.imread(
    str(reference_path),
    cv2.IMREAD_GRAYSCALE,
)

if reference is None:
    raise FileNotFoundError(
        reference_path
    )


# --------------------------------------------------
# 5. SIFT 특징점 검출기 생성
#
# 무엇을 하나요?
# - SIFT 알고리즘을 사용할 준비를 합니다.
#
# SIFT란?
# Scale-Invariant Feature Transform
#
# 이미지에서 특징적인 위치를 찾고,
# 크기나 회전 변화에도 비교할 수 있는
# 특징 정보를 만드는 알고리즘입니다.
#
# ORB와 달리 여기서는 SIFT의 기본 설정을 사용합니다.
# --------------------------------------------------

sift = cv2.SIFT_create()


# --------------------------------------------------
# 6. Reference의 Keypoint와 Descriptor 추출
#
# 무엇을 하나요?
# - Reference 이미지에서 특징점(Keypoint)을 찾습니다.
# - 각 특징점을 비교할 수 있는 Descriptor를 만듭니다.
#
# Keypoint:
# 이미지에서 특징적인 위치와 크기, 방향 등의 정보
#
# Descriptor:
# 특징점 주변의 영상 특징을 표현한 숫자 데이터
# SIFT는 일반적으로 128차원 Descriptor를 사용합니다.
#
# Descriptor가 없으면 비교가 불가능하므로 중단합니다.
# --------------------------------------------------

reference_keypoints, reference_descriptors = (
    sift.detectAndCompute(
        reference,
        None,
    )
)


if reference_descriptors is None:
    raise RuntimeError(
        "Reference에서 SIFT Descriptor를 만들 수 없습니다."
    )


print(
    "Reference Keypoints:",
    len(reference_keypoints),
)


# --------------------------------------------------
# 7. 특징점 비교를 위한 Matcher 생성
#
# 무엇을 하나요?
# - Reference와 Target의 Descriptor를 비교할
#   BFMatcher를 준비합니다.
# - SIFT는 L2 Distance를 사용하여 특징을 비교합니다.
#
# L2 Distance:
# 두 Descriptor 사이의 유클리드 거리입니다.
# 거리가 작을수록 특징이 비슷하다는 의미입니다.
#
# crossCheck=False:
# 이후 knnMatch와 Ratio Test를 사용하기 위한 설정입니다.
#
# rows:
# 각 Target의 측정 결과를 저장할 리스트입니다.
# --------------------------------------------------

matcher = cv2.BFMatcher(
    cv2.NORM_L2,
    crossCheck=False,
)


rows = []


# --------------------------------------------------
# 8. 네 개의 Target 이미지를 순서대로 읽기
#
# 무엇을 하나요?
# - 준비한 Target 이미지 4장을 하나씩 검사합니다.
# - 컬러 이미지를 읽은 후 Grayscale로 변환합니다.
# - 처리시간을 측정하기 위해 시작 시간을 기록합니다.
#
# for 반복문 내부에서는
# 9번부터 15번까지의 과정을 Target마다 실행합니다.
# --------------------------------------------------

for condition, target_path in target_settings:

    target_color = cv2.imread(
        str(target_path)
    )

    if target_color is None:
        raise FileNotFoundError(
            target_path
        )


    target_gray = cv2.cvtColor(
        target_color,
        cv2.COLOR_BGR2GRAY,
    )


    start = perf_counter()


    # --------------------------------------------------
    # 9. Target의 Keypoint와 Descriptor 추출
    #
    # 무엇을 하나요?
    # - Target 이미지에서도 SIFT 특징점을 찾습니다.
    # - Reference와 비교할 Descriptor를 생성합니다.
    #
    # Reference와 Target에 동일한 SIFT를 적용해야
    # 특징점의 Descriptor를 비교할 수 있습니다.
    #
    # good_matches:
    # Ratio Test를 통과한 대응점을 저장할 리스트입니다.
    # --------------------------------------------------

    target_keypoints, target_descriptors = (
        sift.detectAndCompute(
            target_gray,
            None,
        )
    )


    good_matches = []


    # --------------------------------------------------
    # 10. Reference와 Target의 Descriptor 비교
    #
    # 무엇을 하나요?
    # - BFMatcher의 knnMatch()를 사용합니다.
    # - Reference의 각 Descriptor에 대해
    #   가장 가까운 Target 후보 2개를 찾습니다.
    #
    # k=2:
    # 가장 가까운 후보와 두 번째 후보를 찾습니다.
    #
    # 두 후보 사이의 거리 차이를 이용해
    # 다음 Ratio Test에서 좋은 Match를 선별합니다.
    #
    # Target Descriptor가 없다면 Matching은 생략합니다.
    # --------------------------------------------------

    if target_descriptors is not None:

        match_candidates = matcher.knnMatch(
            reference_descriptors,
            target_descriptors,
            k=2,
        )


        # --------------------------------------------------
        # 11. Ratio Test로 Good Match 선별
        #
        # 무엇을 하나요?
        # - 가장 가까운 후보와 두 번째 후보의 거리를 비교합니다.
        # - 첫 번째 후보가 충분히 가까울 때만
        #   신뢰할 만한 Match 후보로 인정합니다.
        #
        # 조건:
        # 첫 번째 거리 < 두 번째 거리 × 0.75
        #
        # 조건을 만족하면 Good Match에 추가합니다.
        #
        # 목적:
        # 다른 특징점과도 비슷하게 보이는
        # 모호한 Match를 제거하는 것입니다.
        # --------------------------------------------------

        for candidate in match_candidates:

            if len(candidate) < 2:
                continue


            m, n = candidate


            if m.distance < 0.75 * n.distance:

                good_matches.append(
                    m
                )


    # 특징 추출과 Matching 처리시간을 밀리초로 계산합니다.
    # 이미지 읽기·정렬·시각화·저장 시간은 포함하지 않습니다.

    elapsed_ms = (
        perf_counter() - start
    ) * 1000


    # --------------------------------------------------
    # 12. Good Match 정렬 및 표시 개수 제한
    #
    # 무엇을 하나요?
    # - Good Match를 L2 Distance가 작은 순서로 정렬합니다.
    # - 가까운 Match가 앞에 오도록 합니다.
    # - 화면이 복잡해지지 않도록 상위 30개만 표시합니다.
    #
    # 주의:
    # 실제 Good Match 전체 개수는 그대로 유지합니다.
    # 화면에 표시하는 개수만 최대 30개로 제한합니다.
    # --------------------------------------------------

    good_matches = sorted(
        good_matches,
        key=lambda match: match.distance,
    )


    shown_matches = good_matches[:30]


    # --------------------------------------------------
    # 13. Reference와 Target의 특징점 연결 이미지 생성
    #
    # 무엇을 하나요?
    # - Reference와 Target 이미지를 나란히 배치합니다.
    # - 대응되는 특징점 사이를 선으로 연결합니다.
    # - SIFT Matching 결과를 시각적으로 보여줍니다.
    #
    # drawMatches():
    # 두 이미지 사이의 대응되는 Keypoint를
    # 선으로 연결하여 보여주는 OpenCV 함수입니다.
    #
    # 이를 통해 이동·회전·크기 변화에서도
    # 특징점이 어떻게 연결되는지 확인할 수 있습니다.
    # --------------------------------------------------

    preview = cv2.drawMatches(
        reference,
        reference_keypoints,
        target_color,
        target_keypoints,
        shown_matches,
        None,
        flags=(
            cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        ),
    )


    # --------------------------------------------------
    # 14. SIFT 특징점 비교 결과 이미지 저장
    #
    # 무엇을 하나요?
    # - 연결선이 표시된 결과 이미지를 JPG로 저장합니다.
    # - Target마다 별도의 결과 파일을 생성합니다.
    #
    # 저장 파일 예시:
    # target_normal_sift.jpg
    # target_shift_sift.jpg
    # target_rot15_sift.jpg
    # target_scale90_sift.jpg
    # --------------------------------------------------

    output_path = (
        output_dir
        / f"{target_path.stem}_sift.jpg"
    )


    cv2.imwrite(
        str(output_path),
        preview,
    )


    # --------------------------------------------------
    # 15. Target별 SIFT 측정값 기록
    #
    # 무엇을 하나요?
    # - 검사 조건과 이미지 이름을 기록합니다.
    # - Reference와 Target의 특징점 개수를 기록합니다.
    # - Ratio Test를 통과한 Good Match 개수를 기록합니다.
    # - 특징 추출 및 Matching 처리시간을 기록합니다.
    #
    # rows 리스트에 저장한 값은
    # 마지막 16번 단계에서 CSV 파일로 저장합니다.
    # --------------------------------------------------

    rows.append({
        "condition": condition,
        "filename": target_path.name,
        "reference_keypoints": len(
            reference_keypoints
        ),
        "target_keypoints": len(
            target_keypoints
        ),
        "good_matches": len(
            good_matches
        ),
        "time_ms": f"{elapsed_ms:.3f}",
    })


    print(
        f"{condition:7s}",
        f"reference_kp={len(reference_keypoints)}",
        f"target_kp={len(target_keypoints)}",
        f"good={len(good_matches)}",
        f"time={elapsed_ms:.2f} ms",
    )


# --------------------------------------------------
# 16. 전체 검사 결과를 CSV 파일로 저장
#
# 무엇을 하나요?
# - 4개 Target의 측정값을 하나의 CSV로 저장합니다.
# - 특징점 개수와 Good Match 수를 비교할 수 있습니다.
# - 정상, 이동, 회전, 크기 변화에 따른
#   SIFT Matching 결과를 분석할 수 있습니다.
#
# 최종 산출물:
# reports/day04_sift_results.csv
#
# outputs/day04/sift/
#   ├── target_normal_sift.jpg
#   ├── target_shift_sift.jpg
#   ├── target_rot15_sift.jpg
#   └── target_scale90_sift.jpg
# --------------------------------------------------

with report_path.open(
    "w",
    newline="",
    encoding="utf-8",
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=[
            "condition",
            "filename",
            "reference_keypoints",
            "target_keypoints",
            "good_matches",
            "time_ms",
        ],
    )

    writer.writeheader()
    writer.writerows(rows)


print()

print(
    "CSV 저장:",
    report_path,
)

print(
    "결과 이미지:",
    output_dir,
)