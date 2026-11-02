from pathlib import Path
from time import perf_counter
import csv

import cv2
import numpy as np


# --------------------------------------------------
# 1. 입력 이미지와 결과 저장 경로 설정
#
# 무엇을 하나요?
# - 비교 기준인 Reference 이미지 경로를 지정합니다.
# - 검사할 Target 이미지 폴더를 지정합니다.
# - Homography 결과 이미지와 CSV 저장 경로를 설정합니다.
# - 저장할 폴더가 없으면 자동으로 생성합니다.
#
# Reference : 기준이 되는 광천김 로고 이미지
# Target    : 위치·회전·크기를 변경한 비교 이미지
# --------------------------------------------------

reference_path = Path(
    "samples/day04/reference/reference_label.jpg"
)

target_dir = Path(
    "samples/day04/targets"
)

output_dir = Path(
    "outputs/day04/homography"
)

report_path = Path(
    "reports/day04_homography_results.csv"
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
# - 각 이미지에 동일한 SIFT + Homography를 적용합니다.
#
# 비교 조건
# normal  : 정상 이미지
# shift   : 위치가 이동한 이미지
# rot15   : 15도 회전한 이미지
# scale90 : 크기가 90%로 변한 이미지
#
# 목적:
# 이미지의 위치·회전·크기가 변하더라도
# Reference 로고의 위치를 찾을 수 있는지 확인합니다.
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
# SIFT는 이미지의 특징점을 추출하는 알고리즘입니다.
# 현재 OpenCV 환경에서 SIFT를 사용할 수 없다면
# 이후 특징점 검출과 Homography 계산을 진행할 수 없습니다.
#
# 따라서 SIFT_create가 없으면 오류를 발생시킵니다.
# --------------------------------------------------

if not hasattr(
    cv2,
    "SIFT_create",
):
    raise RuntimeError(
        "현재 OpenCV에서 SIFT_create를 사용할 수 없습니다."
    )


# --------------------------------------------------
# 4. 기준 이미지(Reference)를 읽고 흑백으로 변환
#
# 무엇을 하나요?
# - 광천김 Reference 이미지를 컬러로 읽습니다.
# - SIFT 특징점 검출을 위해 Grayscale로 변환합니다.
# - 이미지를 읽지 못하면 오류를 발생시킵니다.
#
# 컬러 원본:
# 이후 Reference 영역의 네 모서리를 정의하는 기준입니다.
#
# Grayscale:
# SIFT 특징점을 추출할 때 사용합니다.
# --------------------------------------------------

reference = cv2.imread(
    str(reference_path)
)

if reference is None:
    raise FileNotFoundError(
        reference_path
    )


reference_gray = cv2.cvtColor(
    reference,
    cv2.COLOR_BGR2GRAY,
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
# 이미지에서 특징적인 위치를 찾아내고,
# 크기나 회전 변화에도 비교할 수 있는
# 특징 정보를 생성하는 알고리즘입니다.
#
# 이번 코드에서는 SIFT 기본 설정을 사용합니다.
# --------------------------------------------------

sift = cv2.SIFT_create()


# --------------------------------------------------
# 6. Reference의 Keypoint와 Descriptor 추출
#
# 무엇을 하나요?
# - Reference 이미지에서 특징점(Keypoint)을 찾습니다.
# - 각 특징점의 Descriptor를 생성합니다.
#
# Keypoint:
# 이미지에서 특징적인 위치와 크기, 방향 등의 정보
#
# Descriptor:
# 해당 특징점 주변의 영상 특징을 표현한 숫자 데이터
#
# SIFT Descriptor는 일반적으로 128차원으로 구성됩니다.
#
# Descriptor가 없다면 Target과 비교할 수 없으므로
# 프로그램을 중단합니다.
# --------------------------------------------------

reference_keypoints, reference_descriptors = (
    sift.detectAndCompute(
        reference_gray,
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
# 7. SIFT Descriptor 비교용 Matcher 생성
#
# 무엇을 하나요?
# - Reference와 Target의 특징점을 비교할
#   BFMatcher를 준비합니다.
# - SIFT Descriptor는 L2 Distance로 비교합니다.
#
# L2 Distance:
# 두 Descriptor 사이의 유클리드 거리입니다.
#
# 거리가 작을수록 두 특징이 비슷하다는 의미입니다.
#
# crossCheck=False:
# 이후 knnMatch()와 Ratio Test를 사용하기 위한 설정입니다.
# --------------------------------------------------

matcher = cv2.BFMatcher(
    cv2.NORM_L2,
    crossCheck=False,
)


# --------------------------------------------------
# 8. Reference 이미지의 네 모서리 좌표 준비
#
# 무엇을 하나요?
# - Reference 이미지의 가로·세로 크기를 구합니다.
# - 이미지의 네 모서리 좌표를 만듭니다.
#
# 네 모서리:
# ① 왼쪽 위
# ② 오른쪽 위
# ③ 오른쪽 아래
# ④ 왼쪽 아래
#
# 왜 필요한가요?
# 이후 Homography를 계산하면
# 이 네 모서리가 Target 이미지에서
# 어디에 위치하는지 변환할 수 있기 때문입니다.
#
# 이를 이용하여 Target에 로고 위치를
# 사각형으로 표시합니다.
# --------------------------------------------------

reference_h, reference_w = (
    reference_gray.shape[:2]
)


reference_corners = np.float32([
    [0, 0],
    [reference_w - 1, 0],
    [reference_w - 1, reference_h - 1],
    [0, reference_h - 1],
]).reshape(
    -1,
    1,
    2,
)


# 각 Target의 측정 결과를 저장할 리스트
rows = []


# --------------------------------------------------
# 9. 네 개의 Target 이미지를 순서대로 읽기
#
# 무엇을 하나요?
# - 준비한 Target 이미지 4장을 하나씩 검사합니다.
# - Target을 컬러로 읽고 Grayscale로 변환합니다.
# - 처리시간 측정을 시작합니다.
#
# for 반복문 내부에서는
# 10번부터 20번까지의 과정을 Target마다 수행합니다.
# --------------------------------------------------

for condition, target_path in target_settings:

    target = cv2.imread(
        str(target_path)
    )

    if target is None:
        raise FileNotFoundError(
            target_path
        )


    target_gray = cv2.cvtColor(
        target,
        cv2.COLOR_BGR2GRAY,
    )


    start = perf_counter()


    # --------------------------------------------------
    # 10. Target의 SIFT Keypoint와 Descriptor 추출
    #
    # 무엇을 하나요?
    # - Target에서도 SIFT 특징점을 찾습니다.
    # - 각 특징점의 Descriptor를 만듭니다.
    #
    # Reference와 Target에 동일한 SIFT를 적용하여
    # 서로 대응되는 특징점을 찾을 준비를 합니다.
    #
    # 다음 변수도 초기화합니다.
    #
    # good_matches : Ratio Test를 통과한 특징점 대응
    # inlier_count : RANSAC이 선택한 Inlier 개수
    # inlier_ratio : Good Match 중 Inlier 비율
    # status       : Homography 계산 성공 여부
    # preview      : 결과를 표시할 이미지
    # --------------------------------------------------

    target_keypoints, target_descriptors = (
        sift.detectAndCompute(
            target_gray,
            None,
        )
    )


    good_matches = []

    inlier_count = 0
    inlier_ratio = 0.0

    status = "NOT_FOUND"

    preview = target.copy()


    # --------------------------------------------------
    # 11. Reference와 Target의 Descriptor 비교
    #
    # 무엇을 하나요?
    # - BFMatcher의 knnMatch()를 사용합니다.
    # - Reference의 각 Descriptor에 대해
    #   가까운 Target Descriptor 후보 2개를 찾습니다.
    #
    # k=2:
    # 가장 가까운 후보와 두 번째로 가까운 후보를 구합니다.
    #
    # 두 후보의 거리를 비교하여
    # 다음 단계에서 Good Match를 선별합니다.
    #
    # Target Descriptor가 없다면 Matching을 생략합니다.
    # --------------------------------------------------

    if target_descriptors is not None:

        match_candidates = matcher.knnMatch(
            reference_descriptors,
            target_descriptors,
            k=2,
        )


        # --------------------------------------------------
        # 12. Ratio Test로 Good Match 선별
        #
        # 무엇을 하나요?
        # - 첫 번째 후보와 두 번째 후보의 거리를 비교합니다.
        # - 첫 번째 후보가 충분히 더 가까운 경우만 남깁니다.
        #
        # 조건:
        # 첫 번째 거리 < 두 번째 거리 × 0.75
        #
        # 조건을 만족하면 Good Match에 추가합니다.
        #
        # 왜 필요한가요?
        # 비슷한 특징점이 여러 곳에 나타날 수 있기 때문입니다.
        #
        # Ratio Test는 모호한 대응을 줄여
        # 비교적 신뢰할 만한 Match 후보를 선택합니다.
        # --------------------------------------------------

        for candidate in match_candidates:

            if len(candidate) < 2:
                continue


            m, n = candidate


            if m.distance < 0.75 * n.distance:

                good_matches.append(
                    m
                )


    # --------------------------------------------------
    # 13. Homography 계산에 필요한 대응점 준비
    #
    # 무엇을 하나요?
    # - Good Match가 최소 4개인지 확인합니다.
    # - Reference 특징점의 좌표를 추출합니다.
    # - 대응되는 Target 특징점의 좌표를 추출합니다.
    #
    # 왜 최소 4개인가요?
    # 일반적인 2D Homography를 계산하려면
    # 적절하게 배치된 대응점이 최소 4쌍 필요하기 때문입니다.
    #
    # 단, 대응점이 4개 이상이어도
    # 점들이 잘못 배치되거나 잘못 연결되었다면
    # Homography 계산에 실패할 수 있습니다.
    #
    # queryIdx:
    # Reference 특징점의 인덱스
    #
    # trainIdx:
    # Target 특징점의 인덱스
    # --------------------------------------------------

    if len(good_matches) >= 4:

        # Reference에서 Good Match의 좌표
        reference_points = np.float32([
            reference_keypoints[
                match.queryIdx
            ].pt
            for match in good_matches
        ]).reshape(
            -1,
            1,
            2,
        )


        # Target에서 Good Match의 좌표
        target_points = np.float32([
            target_keypoints[
                match.trainIdx
            ].pt
            for match in good_matches
        ]).reshape(
            -1,
            1,
            2,
        )


        # --------------------------------------------------
        # 14. RANSAC을 사용하여 Homography 계산
        #
        # 무엇을 하나요?
        # - Reference와 Target의 대응점으로
        #   Homography 변환 행렬을 계산합니다.
        # - RANSAC을 이용해 잘못된 대응점의 영향을 줄입니다.
        #
        # Homography:
        # 한 평면의 점들이 다른 이미지에서
        # 어디로 이동하는지 나타내는 변환 관계입니다.
        #
        # RANSAC:
        # 대응점 중에서 변환 관계를 잘 따르는 점을 찾아
        # 잘못된 대응점의 영향을 줄이는 알고리즘입니다.
        #
        # Inlier:
        # 계산한 변환 관계에 잘 맞는 대응점
        #
        # Outlier:
        # 변환 관계에 잘 맞지 않는 대응점
        #
        # 5.0:
        # RANSAC에서 사용하는 재투영 오차 허용 기준입니다.
        # 단위는 픽셀입니다.
        # --------------------------------------------------

        homography, inlier_mask = (
            cv2.findHomography(
                reference_points,
                target_points,
                cv2.RANSAC,
                5.0,
            )
        )


        if (
            homography is not None
            and inlier_mask is not None
        ):

            # --------------------------------------------------
            # 15. Inlier 개수와 Inlier Ratio 계산
            #
            # 무엇을 하나요?
            # - RANSAC이 선택한 Inlier 개수를 계산합니다.
            # - 전체 Good Match 중 Inlier 비율을 구합니다.
            #
            # Inlier Count:
            # 변환 관계에 잘 맞는 대응점 개수
            #
            # Inlier Ratio:
            # Inlier 개수 ÷ Good Match 전체 개수
            #
            # 예:
            # Good Match 40개
            # Inlier 32개
            #
            # Inlier Ratio = 32 / 40 = 0.8
            #
            # 즉 Good Match 중 80%가
            # 계산된 Homography 관계를 따른다는 뜻입니다.
            #
            # 주의:
            # Inlier Ratio가 높다고 해서
            # 반드시 올바른 로고를 찾았다는 의미는 아닙니다.
            # --------------------------------------------------

            inlier_count = int(
                inlier_mask.ravel().sum()
            )


            inlier_ratio = (
                inlier_count
                / len(good_matches)
            )


            # --------------------------------------------------
            # 16. Reference의 네 모서리를 Target 좌표로 변환
            #
            # 무엇을 하나요?
            # - 8번에서 만든 Reference의 네 모서리를 가져옵니다.
            # - 계산된 Homography 행렬을 적용합니다.
            # - 네 모서리가 Target에서 어디에 해당하는지 구합니다.
            #
            # perspectiveTransform():
            # Homography를 사용하여 점의 좌표를 변환합니다.
            #
            # 이 결과를 이용하면
            # Target 이미지에서 Reference가 대응되는 위치를
            # 사각형으로 그릴 수 있습니다.
            # --------------------------------------------------

            projected_corners = (
                cv2.perspectiveTransform(
                    reference_corners,
                    homography,
                )
            )


            # --------------------------------------------------
            # 17. 계산된 위치를 초록색 사각형으로 표시
            #
            # 무엇을 하나요?
            # - 변환한 네 모서리 좌표를 정수로 변환합니다.
            # - Target 이미지에 초록색 사각형을 그립니다.
            # - Homography 계산이 성공한 상태로 기록합니다.
            #
            # cv2.polylines():
            # 여러 점을 선으로 연결하는 함수입니다.
            #
            # (0, 255, 0):
            # OpenCV BGR 색상에서 초록색을 의미합니다.
            #
            # status = FOUND:
            # Homography 행렬이 계산되었다는 의미입니다.
            #
            # 주의:
            # FOUND는 최종 OK 판정이 아닙니다.
            # 사각형 위치가 실제 로고와 일치하는지는
            # 별도로 확인해야 합니다.
            # --------------------------------------------------

            polygon = np.int32(
                projected_corners
            )


            cv2.polylines(
                preview,
                [polygon],
                True,
                (0, 255, 0),
                3,
                cv2.LINE_AA,
            )


            status = "FOUND"


    # SIFT 추출·Matching·Homography 계산시간을 측정합니다.
    # 이미지 읽기, 결과 문자 표시, 이미지 저장시간은 제외합니다.

    elapsed_ms = (
        perf_counter() - start
    ) * 1000


    # --------------------------------------------------
    # 18. Homography 측정 결과를 이미지에 표시
    #
    # 무엇을 하나요?
    # - Target 이미지 위에 측정 결과를 문자로 표시합니다.
    #
    # 표시 항목:
    # good  : Good Match 전체 개수
    # inlier: RANSAC Inlier 개수
    # ratio : Inlier Ratio
    #
    # 예:
    # good=40 inlier=32 ratio=0.80
    #
    # 이 숫자를 보면서
    # 특징점 대응이 얼마나 기하학적으로 일치하는지
    # 비교할 수 있습니다.
    # --------------------------------------------------

    cv2.putText(
        preview,
        (
            f"good={len(good_matches)} "
            f"inlier={inlier_count} "
            f"ratio={inlier_ratio:.2f}"
        ),
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )


    # --------------------------------------------------
    # 19. Homography 결과 이미지 저장
    #
    # 무엇을 하나요?
    # - 사각형과 측정값이 표시된 이미지를 JPG로 저장합니다.
    # - Target마다 별도의 결과 파일을 생성합니다.
    #
    # 저장 파일 예시:
    # target_normal_homography.jpg
    # target_shift_homography.jpg
    # target_rot15_homography.jpg
    # target_scale90_homography.jpg
    #
    # 학생들은 각 결과 이미지를 비교하면서
    # 사각형 위치가 실제 광천김 로고와 일치하는지 확인합니다.
    # --------------------------------------------------

    output_path = (
        output_dir
        / f"{target_path.stem}_homography.jpg"
    )


    cv2.imwrite(
        str(output_path),
        preview,
    )


    # --------------------------------------------------
    # 20. Target별 Homography 측정값 기록
    #
    # 무엇을 하나요?
    # - Target마다 계산한 측정값을 rows 리스트에 저장합니다.
    #
    # 저장 항목:
    #
    # condition   : 정상·이동·회전·크기 변화 조건
    # filename    : Target 이미지 이름
    # good_matches: Ratio Test를 통과한 Match 개수
    # inliers     : RANSAC Inlier 개수
    # inlier_ratio: Inlier / Good Match
    # status      : FOUND 또는 NOT_FOUND
    # time_ms     : 처리시간(밀리초)
    #
    # 이렇게 기록한 결과는
    # 21번에서 하나의 CSV 파일로 저장합니다.
    # --------------------------------------------------

    rows.append({
        "condition": condition,
        "filename": target_path.name,
        "good_matches": len(
            good_matches
        ),
        "inliers": inlier_count,
        "inlier_ratio": f"{inlier_ratio:.6f}",
        "status": status,
        "time_ms": f"{elapsed_ms:.3f}",
    })


    print(
        f"{condition:7s}",
        f"good={len(good_matches)}",
        f"inlier={inlier_count}",
        f"ratio={inlier_ratio:.3f}",
        f"status={status}",
        f"time={elapsed_ms:.2f} ms",
    )


# --------------------------------------------------
# 21. 전체 Homography 측정 결과를 CSV로 저장
#
# 무엇을 하나요?
# - 4개 Target의 측정값을 하나의 CSV로 저장합니다.
# - Good Match, Inlier, Inlier Ratio를 비교합니다.
# - 정상·이동·회전·크기 변화에 따른
#   Homography 계산 결과를 분석합니다.
#
# 최종 산출물:
#
# reports/day04_homography_results.csv
#
# outputs/day04/homography/
#   ├── target_normal_homography.jpg
#   ├── target_shift_homography.jpg
#   ├── target_rot15_homography.jpg
#   └── target_scale90_homography.jpg
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
            "good_matches",
            "inliers",
            "inlier_ratio",
            "status",
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