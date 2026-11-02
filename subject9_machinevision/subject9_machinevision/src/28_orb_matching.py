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
# - ORB 비교 이미지와 CSV 결과를 저장할 위치를 준비합니다.
# - 저장 폴더가 없으면 자동으로 생성합니다.
# --------------------------------------------------

reference_path = Path(
    "samples/day04/reference/reference_label.jpg"
)

target_dir = Path(
    "samples/day04/targets"
)

output_dir = Path(
    "outputs/day04/orb"
)

report_path = Path(
    "reports/day04_orb_results.csv"
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
# - 각 이미지에 같은 ORB 알고리즘을 적용합니다.
#
# 비교 조건
# normal  : 정상 이미지
# shift   : 위치가 이동한 이미지
# rot15   : 15도 회전한 이미지
# scale90 : 크기가 90%로 변한 이미지
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
# 3. 기준 이미지(Reference)를 흑백으로 읽기
#
# 무엇을 하나요?
# - 광천김 로고의 기준 이미지를 불러옵니다.
# - ORB 특징점을 검출하기 위해 Grayscale로 읽습니다.
# - 파일을 읽지 못하면 오류를 발생시킵니다.
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
# 4. ORB 특징점 검출기 생성
#
# 무엇을 하나요?
# - ORB 알고리즘을 사용할 준비를 합니다.
# - nfeatures=1000은 검출할 주요 특징점 수의
#   목표 상한을 지정하는 설정입니다.
#
# ORB는 이미지의 특징적인 위치를 찾아
# 다른 이미지에서도 같은 특징이 있는지 비교합니다.
# --------------------------------------------------

orb = cv2.ORB_create(
    nfeatures=1000
)


# --------------------------------------------------
# 5. Reference의 Keypoint와 Descriptor 추출
#
# 무엇을 하나요?
# - 기준 이미지에서 특징점(Keypoint)을 찾습니다.
# - 각 특징점을 비교할 수 있는 Descriptor를 만듭니다.
#
# Keypoint   : 이미지에서 특징적인 위치
# Descriptor : 그 위치의 특징을 표현한 데이터
#
# Descriptor가 없으면 이후 비교가 불가능하므로
# 프로그램을 중단합니다.
# --------------------------------------------------

reference_keypoints, reference_descriptors = (
    orb.detectAndCompute(
        reference,
        None,
    )
)


if reference_descriptors is None:
    raise RuntimeError(
        "Reference에서 ORB Descriptor를 만들 수 없습니다."
    )


print(
    "Reference Keypoints:",
    len(reference_keypoints),
)


# --------------------------------------------------
# 6. 특징점 비교를 위한 Matcher 생성
#
# 무엇을 하나요?
# - Reference와 Target의 Descriptor를 비교할
#   BFMatcher를 준비합니다.
# - ORB는 이진 Descriptor이므로
#   Hamming Distance를 사용합니다.
#
# Hamming Distance가 작을수록
# 두 Descriptor가 비슷하다는 의미입니다.
#
# crossCheck=False:
# - 이후 knnMatch와 Ratio Test를 적용하기 위한 설정입니다.
# --------------------------------------------------

matcher = cv2.BFMatcher(
    cv2.NORM_HAMMING,
    crossCheck=False,
)


rows = []


# --------------------------------------------------
# 7. 네 개의 Target 이미지를 순서대로 읽기
#
# 무엇을 하나요?
# - Target 4장을 하나씩 반복해서 검사합니다.
# - 각 이미지를 읽고 Grayscale로 변환합니다.
# - 처리시간을 측정하기 위해 시작 시간을 기록합니다.
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
    # 8. Target의 Keypoint와 Descriptor 추출
    #
    # 무엇을 하나요?
    # - Target 이미지에서도 ORB 특징점을 찾습니다.
    # - Reference와 비교할 Descriptor를 생성합니다.
    #
    # Reference와 Target에 동일한 ORB를 적용해야
    # 특징을 비교할 수 있습니다.
    # --------------------------------------------------

    target_keypoints, target_descriptors = (
        orb.detectAndCompute(
            target_gray,
            None,
        )
    )


    good_matches = []


    # --------------------------------------------------
    # 9. Reference와 Target의 Descriptor 비교
    #
    # 무엇을 하나요?
    # - BFMatcher의 knnMatch()를 사용합니다.
    # - Reference의 각 Descriptor에 대해
    #   가장 가까운 Target 후보 2개를 찾습니다.
    #
    # k=2:
    # - 가장 비슷한 후보와 두 번째 후보를 찾습니다.
    # - 두 후보의 거리를 다음 Ratio Test에서 비교합니다.
    # --------------------------------------------------

    if target_descriptors is not None:

        match_candidates = matcher.knnMatch(
            reference_descriptors,
            target_descriptors,
            k=2,
        )


        # --------------------------------------------------
        # 10. Ratio Test로 Good Match 선별
        #
        # 무엇을 하나요?
        # - 가장 가까운 후보와 두 번째 후보의
        #   Hamming Distance를 비교합니다.
        # - 가장 가까운 후보가 충분히 더 가까울 때만
        #   신뢰할 만한 Match로 인정합니다.
        #
        # 조건:
        # 첫 번째 거리 < 두 번째 거리 × 0.75
        #
        # 조건을 만족하면 Good Match에 추가합니다.
        # --------------------------------------------------

        for candidate in match_candidates:

            if len(candidate) < 2:
                continue


            m, n = candidate


            if m.distance < 0.75 * n.distance:

                good_matches.append(
                    m
                )


    elapsed_ms = (
        perf_counter() - start
    ) * 1000


    # --------------------------------------------------
    # 11. Good Match 정렬 및 표시 개수 제한
    #
    # 무엇을 하나요?
    # - Good Match를 Hamming Distance가 작은 순서로 정렬합니다.
    # - 더 가까운 Match가 앞에 오도록 합니다.
    # - 화면이 복잡해지지 않도록 상위 30개만 표시합니다.
    #
    # 주의:
    # - 실제 Good Match 전체 개수는 유지합니다.
    # - 화면에 표시하는 개수만 최대 30개로 제한합니다.
    # --------------------------------------------------

    good_matches = sorted(
        good_matches,
        key=lambda match: match.distance,
    )


    shown_matches = good_matches[:30]


    # --------------------------------------------------
    # 12. Reference와 Target의 특징점 연결 이미지 생성
    #
    # 무엇을 하나요?
    # - Reference와 Target을 나란히 배치합니다.
    # - 서로 대응되는 특징점을 선으로 연결합니다.
    # - Good Match가 어디에서 발생했는지 확인합니다.
    #
    # drawMatches()는 두 이미지에서 대응되는
    # Keypoint를 시각적으로 보여주는 함수입니다.
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
    # 13. ORB 특징점 비교 결과 이미지 저장
    #
    # 무엇을 하나요?
    # - 연결선이 표시된 결과 이미지를 JPG로 저장합니다.
    # - Target마다 별도의 결과 파일을 생성합니다.
    #
    # 예:
    # target_normal_orb.jpg
    # target_shift_orb.jpg
    # target_rot15_orb.jpg
    # target_scale90_orb.jpg
    # --------------------------------------------------

    output_path = (
        output_dir
        / f"{target_path.stem}_orb.jpg"
    )


    cv2.imwrite(
        str(output_path),
        preview,
    )


    # --------------------------------------------------
    # 14. Target별 ORB 측정값 기록
    #
    # 무엇을 하나요?
    # - 검사 조건과 이미지 이름을 기록합니다.
    # - Reference와 Target의 특징점 개수를 기록합니다.
    # - Ratio Test를 통과한 Good Match 개수를 기록합니다.
    # - 특징 추출 및 매칭 처리시간을 기록합니다.
    #
    # rows 리스트에 저장한 값은
    # 마지막 단계에서 CSV 파일로 저장합니다.
    #
    # 주의:
    # time_ms에는 이미지 저장과 시각화 시간이 포함되지 않습니다.
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
# 15. 전체 검사 결과를 CSV 파일로 저장
#
# 무엇을 하나요?
# - 4개 Target에 대한 측정값을 하나의 CSV로 저장합니다.
# - 조건별 특징점 개수와 Good Match 수를 비교할 수 있습니다.
# - 정상, 이동, 회전, 크기 변화에 따라
#   ORB Matching 결과가 어떻게 달라지는지 분석합니다.
#
# 최종 산출물:
# reports/day04_orb_results.csv
#
# outputs/day04/orb/
#   ├── target_normal_orb.jpg
#   ├── target_shift_orb.jpg
#   ├── target_rot15_orb.jpg
#   └── target_scale90_orb.jpg
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