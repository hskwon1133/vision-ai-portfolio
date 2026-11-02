from pathlib import Path

import cv2
import numpy as np


# --------------------------------------------------
# 1. 이미지 파일을 읽는 공통 함수
#
# 무엇을 하나요?
# - 전달받은 이미지 경로를 Path 객체로 변환합니다.
# - OpenCV로 이미지를 읽습니다.
# - 이미지가 없거나 읽을 수 없으면 오류를 발생시킵니다.
# - 정상적으로 읽었다면 이미지를 반환합니다.
#
# 왜 필요한가요?
# Reference와 Target을 읽는 코드를
# 여러 곳에서 반복하지 않기 위해서입니다.
#
# 입력: 이미지 파일 경로
# 출력: OpenCV 이미지(BGR)
# --------------------------------------------------

def read_image(path):

    path = Path(path)

    image = cv2.imread(
        str(path)
    )

    if image is None:
        raise FileNotFoundError(
            f"이미지를 읽지 못했습니다: {path}"
        )

    return image


# --------------------------------------------------
# 2. 기준 이미지(Reference)를 읽고 흑백으로 변환
#
# 무엇을 하나요?
# - 1번에서 만든 read_image()를 호출합니다.
# - 광천김 기준 로고 이미지를 읽습니다.
# - SIFT 특징점을 검출하기 위해 Grayscale로 변환합니다.
#
# 왜 필요한가요?
# Reference는 앞으로 검사할 모든 Target에서
# 동일하게 사용하는 기준 로고이기 때문입니다.
#
# 입력: Reference 이미지 경로
# --------------------------------------------------

def prepare_reference(reference_path):

    reference = read_image(
        reference_path
    )

    reference_gray = cv2.cvtColor(
        reference,
        cv2.COLOR_BGR2GRAY
    )


    # --------------------------------------------------
    # 3. 현재 OpenCV에서 SIFT 사용 가능 여부 확인
    #
    # 무엇을 하나요?
    # - OpenCV에 SIFT_create가 있는지 확인합니다.
    # - SIFT 기능이 없으면 오류를 발생시킵니다.
    #
    # 왜 필요한가요?
    # 이후 Reference와 Target에서 SIFT 특징점을
    # 추출해야 하므로 실행 가능 여부를 먼저 확인합니다.
    # --------------------------------------------------

    if not hasattr(
        cv2,
        "SIFT_create"
    ):
        raise RuntimeError(
            "현재 OpenCV에서 SIFT를 사용할 수 없습니다."
        )


    # --------------------------------------------------
    # 4. SIFT 생성 및 Reference 특징점 추출
    #
    # 무엇을 하나요?
    # - SIFT 특징점 검출기를 생성합니다.
    # - 기준 로고에서 Keypoint를 찾습니다.
    # - 각 Keypoint의 Descriptor를 생성합니다.
    #
    # Keypoint:
    # 이미지에서 특징적인 위치
    #
    # Descriptor:
    # 특징점 주변의 정보를 표현한 숫자 데이터
    #
    # nfeatures=1500:
    # SIFT에서 유지할 주요 특징점 수의
    # 목표 상한을 지정합니다.
    #
    # 특징점이 4개 미만이거나 Descriptor가 없으면
    # Homography 계산이 어려우므로 중단합니다.
    # --------------------------------------------------

    sift = cv2.SIFT_create(
        nfeatures=1500
    )


    ref_keypoints, ref_descriptors = (
        sift.detectAndCompute(
            reference_gray,
            None
        )
    )


    if (
        ref_descriptors is None
        or len(ref_keypoints) < 4
    ):
        raise RuntimeError(
            "Reference에서 충분한 SIFT 특징점을 찾지 못했습니다."
        )


    # --------------------------------------------------
    # 5. Reference 측정 정보를 딕셔너리로 반환
    #
    # 무엇을 하나요?
    # - Reference 원본 이미지와 흑백 이미지를 저장합니다.
    # - SIFT 객체를 저장합니다.
    # - Keypoint와 Descriptor를 저장합니다.
    # - Reference 이미지의 가로·세로 크기를 저장합니다.
    #
    # 왜 필요한가요?
    # 여러 Target을 검사할 때 Reference 특징점을
    # 매번 다시 계산하지 않고 재사용하기 위해서입니다.
    #
    # 반환값:
    # reference_data 딕셔너리
    # --------------------------------------------------

    return {
        "image": reference,
        "gray": reference_gray,
        "sift": sift,
        "keypoints": ref_keypoints,
        "descriptors": ref_descriptors,
        "width": reference.shape[1],
        "height": reference.shape[0],
    }


# --------------------------------------------------
# 6. Target 이미지 한 장을 읽고 흑백으로 변환
#
# 무엇을 하나요?
# - 검사할 Target 이미지 한 장을 입력받습니다.
# - read_image() 함수를 이용해 이미지를 읽습니다.
# - Grayscale로 변환합니다.
#
# 왜 필요한가요?
# Template Matching과 SIFT에 사용할
# 검사 이미지를 준비하기 위해서입니다.
#
# 입력:
# target_path   : 검사 이미지 경로
# reference_data: 5번에서 준비한 기준 정보
#
# 출력:
# 이미지 한 장의 여러 측정값
# --------------------------------------------------

def measure_target(
    target_path,
    reference_data
):

    target = read_image(
        target_path
    )

    target_gray = cv2.cvtColor(
        target,
        cv2.COLOR_BGR2GRAY
    )


    # --------------------------------------------------
    # 7. Reference에서 미리 계산한 정보 가져오기
    #
    # 무엇을 하나요?
    # - Reference의 흑백 이미지를 가져옵니다.
    # - Reference의 Keypoint와 Descriptor를 가져옵니다.
    # - 이미 생성한 SIFT 객체를 가져옵니다.
    #
    # 왜 필요한가요?
    # Target마다 Reference 특징점을 다시 계산하지 않고
    # 동일한 기준을 재사용하기 위해서입니다.
    # --------------------------------------------------

    reference_gray = (
        reference_data["gray"]
    )

    ref_keypoints = (
        reference_data["keypoints"]
    )

    ref_descriptors = (
        reference_data["descriptors"]
    )

    sift = (
        reference_data["sift"]
    )


    # --------------------------------------------------
    # 8. Template Matching으로 로고 유사도 측정
    #
    # 무엇을 하나요?
    # - Target 이미지에서 Reference와 비슷한 영역을 찾습니다.
    # - TM_CCOEFF_NORMED 방법으로 유사도를 계산합니다.
    # - 가장 높은 Template Score와 그 위치를 가져옵니다.
    #
    # template_score:
    # 가장 유사한 영역의 정규화 상관계수 점수
    #
    # template_location:
    # 가장 높은 점수가 나온 위치의 왼쪽 위 좌표
    #
    # 주의:
    # 현재 코드는 Reference 크기를 바꾸지 않고
    # Target을 검색하는 기본 Template Matching입니다.
    # Reference는 Target보다 크지 않아야 합니다.
    #
    # 아직 OK / NG를 판정하지 않습니다.
    # --------------------------------------------------

    template_result = cv2.matchTemplate(
        target_gray,
        reference_gray,
        cv2.TM_CCOEFF_NORMED
    )


    (
        _,
        template_score,
        _,
        template_location
    ) = cv2.minMaxLoc(
        template_result
    )


    # --------------------------------------------------
    # 9. Target의 SIFT Keypoint와 Descriptor 추출
    #
    # 무엇을 하나요?
    # - Target 이미지에서 SIFT 특징점을 찾습니다.
    # - 각 특징점의 Descriptor를 생성합니다.
    #
    # 왜 필요한가요?
    # Reference와 Target의 특징점을 비교하여
    # 같은 로고가 존재하는지 확인하기 위해서입니다.
    #
    # good_matches:
    # Ratio Test를 통과한 특징점 대응을 저장할 리스트
    # --------------------------------------------------

    target_keypoints, target_descriptors = (
        sift.detectAndCompute(
            target_gray,
            None
        )
    )


    good_matches = []


    # --------------------------------------------------
    # 10. BFMatcher로 Reference와 Target 특징점 비교
    #
    # 무엇을 하나요?
    # - Target Descriptor가 있는지 확인합니다.
    # - BFMatcher를 생성합니다.
    # - L2 Distance로 두 이미지의 Descriptor를 비교합니다.
    # - 가장 가까운 후보 2개를 찾습니다.
    #
    # k=2:
    # 첫 번째로 가까운 후보와
    # 두 번째로 가까운 후보를 찾는 설정입니다.
    #
    # 왜 필요한가요?
    # 다음 Ratio Test에서 두 후보의 거리를 비교하여
    # 모호한 Match를 제거하기 위해서입니다.
    # --------------------------------------------------

    if target_descriptors is not None:

        matcher = cv2.BFMatcher(
            cv2.NORM_L2
        )


        knn_matches = matcher.knnMatch(
            ref_descriptors,
            target_descriptors,
            k=2
        )


        # --------------------------------------------------
        # 11. Ratio Test로 Good Match 선별
        #
        # 무엇을 하나요?
        # - 가장 가까운 후보와 두 번째 후보를 비교합니다.
        # - 첫 번째 후보가 충분히 가까울 때만 남깁니다.
        #
        # 조건:
        # 첫 번째 거리 < 두 번째 거리 × 0.75
        #
        # 조건을 통과한 대응점을 Good Match라고 합니다.
        #
        # 왜 필요한가요?
        # 여러 위치에서 비슷하게 보이는 특징점의
        # 잘못된 대응 가능성을 줄이기 위해서입니다.
        # --------------------------------------------------

        for pair in knn_matches:

            if len(pair) < 2:
                continue


            m, n = pair


            if (
                m.distance
                < 0.75 * n.distance
            ):

                good_matches.append(
                    m
                )


    # --------------------------------------------------
    # 12. Homography 측정값 초기화
    #
    # 무엇을 하나요?
    # - Homography 계산 결과를 저장할 변수를 준비합니다.
    #
    # homography_found:
    # Homography 계산 성공 여부 (0 또는 1)
    #
    # inlier_count:
    # RANSAC을 통과한 대응점 개수
    #
    # inlier_ratio:
    # Good Match 중 Inlier가 차지하는 비율
    #
    # polygon:
    # Target에서 검출한 로고 영역의 네 모서리 좌표
    #
    # 왜 필요한가요?
    # 특징점이 부족하여 Homography 계산에 실패해도
    # 기본 측정값을 반환할 수 있도록 하기 위해서입니다.
    # --------------------------------------------------

    homography_found = 0

    inlier_count = 0

    inlier_ratio = 0.0

    polygon = None


    # --------------------------------------------------
    # 13. Homography 계산에 필요한 대응점 좌표 준비
    #
    # 무엇을 하나요?
    # - Good Match가 최소 4개인지 확인합니다.
    # - Reference의 특징점 좌표를 추출합니다.
    # - 그에 대응하는 Target 특징점 좌표를 추출합니다.
    #
    # queryIdx:
    # Reference 특징점의 인덱스
    #
    # trainIdx:
    # Target 특징점의 인덱스
    #
    # 왜 필요한가요?
    # Homography를 계산하려면 두 이미지에서
    # 서로 대응되는 점의 좌표가 필요하기 때문입니다.
    #
    # 주의:
    # 대응점이 4개 이상이어도 위치 관계가 부적절하면
    # Homography 계산에 실패할 수 있습니다.
    # --------------------------------------------------

    if len(good_matches) >= 4:

        source_points = np.float32(
            [
                ref_keypoints[
                    match.queryIdx
                ].pt
                for match in good_matches
            ]
        ).reshape(
            -1,
            1,
            2
        )


        target_points = np.float32(
            [
                target_keypoints[
                    match.trainIdx
                ].pt
                for match in good_matches
            ]
        ).reshape(
            -1,
            1,
            2
        )


        # --------------------------------------------------
        # 14. RANSAC을 이용하여 Homography 계산
        #
        # 무엇을 하나요?
        # - 두 이미지의 대응점으로 Homography를 계산합니다.
        # - RANSAC을 이용해 잘못된 대응점의 영향을 줄입니다.
        # - Homography 행렬과 Inlier Mask를 반환받습니다.
        #
        # Homography:
        # 한 평면의 위치가 다른 이미지에서
        # 어떻게 변했는지 설명하는 변환 행렬입니다.
        #
        # RANSAC:
        # 전체 대응점 중에서 기하학적으로
        # 일관된 대응점을 찾아내는 알고리즘입니다.
        #
        # 5.0:
        # RANSAC의 재투영 오차 허용 기준(픽셀)입니다.
        #
        # Homography가 계산되면
        # homography_found를 1로 설정합니다.
        # --------------------------------------------------

        homography, mask = (
            cv2.findHomography(
                source_points,
                target_points,
                cv2.RANSAC,
                5.0
            )
        )


        if (
            homography is not None
            and mask is not None
        ):

            homography_found = 1


            # --------------------------------------------------
            # 15. Inlier 개수와 Inlier Ratio 계산
            #
            # 무엇을 하나요?
            # - RANSAC이 선택한 Inlier 개수를 계산합니다.
            # - Good Match 전체에서 Inlier 비율을 계산합니다.
            #
            # Inlier Count:
            # Homography 관계에 잘 맞는 대응점 개수
            #
            # Inlier Ratio:
            # Inlier Count / Good Match Count
            #
            # 예:
            # Good Match = 100개
            # Inlier = 90개
            #
            # Inlier Ratio = 90 / 100 = 0.90
            #
            # 즉 Good Match의 90%가
            # 계산된 Homography 관계를 따른다는 뜻입니다.
            # --------------------------------------------------

            inlier_count = int(
                mask.ravel().sum()
            )


            inlier_ratio = (
                inlier_count
                / len(good_matches)
            )


            # --------------------------------------------------
            # 16. Reference 로고의 네 모서리 좌표 준비
            #
            # 무엇을 하나요?
            # - Reference의 가로·세로 크기를 가져옵니다.
            # - 기준 로고의 네 모서리 좌표를 만듭니다.
            #
            # 네 모서리:
            # ① 왼쪽 위
            # ② 오른쪽 위
            # ③ 오른쪽 아래
            # ④ 왼쪽 아래
            #
            # 왜 필요한가요?
            # 이 네 점에 Homography를 적용하면
            # Target에서 로고가 위치하는 영역을
            # 계산할 수 있기 때문입니다.
            # --------------------------------------------------

            ref_width = (
                reference_data["width"]
            )

            ref_height = (
                reference_data["height"]
            )


            corners = np.float32(
                [
                    [0, 0],

                    [
                        ref_width - 1,
                        0
                    ],

                    [
                        ref_width - 1,
                        ref_height - 1
                    ],

                    [
                        0,
                        ref_height - 1
                    ],
                ]
            ).reshape(
                -1,
                1,
                2
            )


            # --------------------------------------------------
            # 17. Reference 모서리를 Target 위치로 변환
            #
            # 무엇을 하나요?
            # - Homography를 이용해 네 모서리 좌표를 변환합니다.
            # - 변환 결과가 유한한 숫자인지 확인합니다.
            # - 정상이라면 polygon에 네 점을 저장합니다.
            #
            # cv2.perspectiveTransform():
            # Homography를 이용하여 좌표를 변환하는 함수
            #
            # np.isfinite():
            # 결과에 NaN이나 무한대가 있는지 확인하는 함수
            #
            # polygon:
            # 나중에 검사 이미지 위에 로고의 위치를
            # 사각형으로 표시할 때 사용할 좌표입니다.
            # --------------------------------------------------

            projected = (
                cv2.perspectiveTransform(
                    corners,
                    homography
                )
            )


            if np.isfinite(
                projected
            ).all():

                polygon = (
                    projected
                    .reshape(-1, 2)
                )


    # --------------------------------------------------
    # 18. 이미지 한 장의 전체 측정 결과 반환
    #
    # 무엇을 하나요?
    # - Template Matching 측정값을 반환합니다.
    # - SIFT Good Match 개수를 반환합니다.
    # - Homography 계산 성공 여부를 반환합니다.
    # - Inlier 개수와 비율을 반환합니다.
    # - 로고 영역 좌표와 원본 Target 이미지를 반환합니다.
    #
    # 왜 필요한가요?
    # 이후 Calibration과 Test에서 동일한 측정 함수를
    # 호출하여 검사값을 재사용하기 위해서입니다.
    #
    # 중요:
    # 이 함수는 측정만 수행합니다.
    # 아직 최종 OK / NG를 판정하지 않습니다.
    # --------------------------------------------------

    return {

        "target_path":
            str(target_path),

        "template_score":
            float(template_score),

        "template_x":
            int(
                template_location[0]
            ),

        "template_y":
            int(
                template_location[1]
            ),

        "sift_good_matches":
            int(
                len(good_matches)
            ),

        "homography_found":
            int(
                homography_found
            ),

        "homography_inliers":
            int(
                inlier_count
            ),

        "inlier_ratio":
            float(
                inlier_ratio
            ),

        "polygon":
            polygon,

        "target_image":
            target,
    }