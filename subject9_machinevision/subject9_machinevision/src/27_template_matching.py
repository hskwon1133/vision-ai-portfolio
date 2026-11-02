from pathlib import Path
from time import perf_counter
import csv

import cv2


# --------------------------------------------------
# 1. 사용할 이미지와 결과 저장 경로를 설정합니다.
#
# template_path
# → 앞의 24번 코드에서 잘라낸 광천김 기준 로고 ROI
#
# target_dir
# → 정상·이동·회전·축소 Target 이미지 4장이 있는 폴더
#
# output_dir
# → 로고를 찾은 위치를 사각형으로 표시한 이미지 저장
#
# report_path
# → 유사도 Score와 로고 위치 등을 CSV로 저장
#
# mkdir()
# → 결과 저장 폴더가 없으면 생성합니다.
# --------------------------------------------------

template_path = Path(
    "samples/day04/reference/reference_label.jpg"
)

target_dir = Path(
    "samples/day04/targets"
)

output_dir = Path(
    "outputs/day04/template"
)

report_path = Path(
    "reports/day04_template_results.csv"
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
# 2. 비교할 Target 이미지 4장을 순서대로 지정합니다.
#
# normal
# → 원본과 동일한 정상 이미지
#
# shift
# → 위치를 이동한 이미지
#
# rot15
# → 15도 회전한 이미지
#
# scale90
# → 원본의 90% 크기로 축소한 이미지
#
# 동일한 기준 로고를 네 이미지에서 각각 찾아
# 검사 조건에 따른 결과 차이를 비교합니다.
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
# 3. 광천김 기준 로고 Template을 Gray로 읽습니다.
#
# cv2.IMREAD_GRAYSCALE
# → 이미지를 Gray 형태로 읽습니다.
#
# Template
# → 전체 제품 이미지에서 찾으려는 작은 기준 로고
#
# 이미지를 읽지 못하면 오류를 발생시킵니다.
#
# template_h → 기준 로고의 높이
# template_w → 기준 로고의 너비
#
# 이 크기는 나중에 검출 위치에
# 사각형을 그릴 때 사용합니다.
# --------------------------------------------------

template = cv2.imread(
    str(template_path),
    cv2.IMREAD_GRAYSCALE,
)

if template is None:
    raise FileNotFoundError(
        template_path
    )


template_h, template_w = (
    template.shape[:2]
)

print(
    "Template size:",
    template_w,
    "x",
    template_h,
)


# --------------------------------------------------
# 4. 네 이미지의 검사 결과를 저장할 리스트를 만듭니다.
#
# rows = []
# → 처음에는 데이터가 없는 빈 리스트
#
# 각 Target에서 측정한
# Score, Location, 처리시간 등을
# 하나씩 추가합니다.
#
# 마지막 11번에서 이 리스트를 CSV로 저장합니다.
# --------------------------------------------------

rows = []


# --------------------------------------------------
# 5. Target 이미지 4장을 하나씩 읽고 Gray로 변환합니다.
#
# for 반복문
# → normal → shift → rot15 → scale90
#   순서로 하나씩 처리합니다.
#
# cv2.imread()
# → 전체 제품 이미지를 읽습니다.
#
# cv2.cvtColor()
# → 컬러 BGR 이미지를 Gray로 변환합니다.
#
# Template과 Target을 모두 Gray로 준비해야
# 같은 방식으로 밝기 패턴을 비교할 수 있습니다.
#
# Template이 Target보다 크면
# Template Matching을 실행할 수 없으므로
# 크기를 확인합니다.
# --------------------------------------------------

for condition, target_path in target_settings:

    target = cv2.imread(
        str(target_path)
    )

    if target is None:
        raise FileNotFoundError(
            target_path
        )


    gray = cv2.cvtColor(
        target,
        cv2.COLOR_BGR2GRAY,
    )


    target_h, target_w = (
        gray.shape[:2]
    )


    if (
        template_w > target_w
        or template_h > target_h
    ):
        raise ValueError(
            f"Template이 Target보다 큽니다: "
            f"{target_path.name}"
        )


    # --------------------------------------------------
    # 6. Template Matching으로 유사도와 위치를 찾습니다.
    #
    # 이번 코드의 핵심 기능입니다.
    #
    # cv2.matchTemplate()
    # → 작은 기준 로고를 전체 Target 이미지의
    #   여러 위치에 대입하여 유사도를 계산합니다.
    #
    # cv2.TM_CCOEFF_NORMED
    # → 정규화된 상관계수 방식으로 비교합니다.
    #
    # score_map
    # → 각 위치에서 계산한 유사도 점수의 모음
    #
    # cv2.minMaxLoc()
    # → 가장 높은 Score와 해당 위치를 찾습니다.
    #
    # max_score
    # → 기준 로고와 가장 비슷한 정도
    #
    # max_loc
    # → 가장 높은 Score가 나온 왼쪽 위 좌표
    #
    # perf_counter()
    # → 처리시간을 측정합니다.
    # --------------------------------------------------

    start = perf_counter()


    score_map = cv2.matchTemplate(
        gray,
        template,
        cv2.TM_CCOEFF_NORMED,
    )


    _, max_score, _, max_loc = (
        cv2.minMaxLoc(
            score_map
        )
    )


    elapsed_ms = (
        perf_counter() - start
    ) * 1000


    # --------------------------------------------------
    # 7. 가장 비슷한 로고 영역의 사각형 좌표를 계산합니다.
    #
    # max_loc
    # → 검출된 영역의 왼쪽 위 좌표
    #
    # top_left
    # → 사각형의 왼쪽 위
    #
    # bottom_right
    # → Template 너비와 높이를 더한 오른쪽 아래
    #
    # center_x, center_y
    # → 찾은 영역의 중심 좌표
    #
    # 이 좌표로 Target 이미지에서
    # 로고가 어디에 있는지 표시할 수 있습니다.
    # --------------------------------------------------

    top_left = max_loc

    bottom_right = (
        top_left[0] + template_w,
        top_left[1] + template_h,
    )


    center_x = (
        top_left[0]
        + template_w / 2
    )

    center_y = (
        top_left[1]
        + template_h / 2
    )


    # --------------------------------------------------
    # 8. 찾은 로고 위치를 Target 이미지에 표시합니다.
    #
    # target.copy()
    # → 원본을 변경하지 않기 위해 복사합니다.
    #
    # cv2.rectangle()
    # → 가장 비슷하다고 판단한 위치에
    #   초록색 사각형을 그립니다.
    #
    # cv2.putText()
    # → 사각형 주변에 Score를 표시합니다.
    #
    # 주의:
    # 초록색 사각형이 그려졌다고 해서
    # 실제 로고를 정확하게 찾았다는 뜻은 아닙니다.
    #
    # 결과를 열어 사각형이 올바른 로고를
    # 감싸는지 직접 확인해야 합니다.
    # --------------------------------------------------

    preview = target.copy()


    cv2.rectangle(
        preview,
        top_left,
        bottom_right,
        (0, 255, 0),
        3,
    )


    cv2.putText(
        preview,
        f"score={max_score:.3f}",
        (
            top_left[0],
            max(
                30,
                top_left[1] - 10,
            ),
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )


    # --------------------------------------------------
    # 9. 검출 결과 이미지를 파일로 저장합니다.
    #
    # 각 Target에서 찾은 로고의 위치와
    # 유사도 Score가 표시된 이미지를 저장합니다.
    #
    # normal_template.jpg
    # shift_template.jpg
    # rot15_template.jpg
    # scale90_template.jpg
    #
    # 이 결과들은 학생이 직접 열어
    # 검출 위치가 맞는지 확인하는 데 사용합니다.
    # --------------------------------------------------

    output_path = (
        output_dir
        / f"{condition}_template.jpg"
    )


    cv2.imwrite(
        str(output_path),
        preview,
    )


    # --------------------------------------------------
    # 10. 측정한 Score와 Location을 리스트에 기록합니다.
    #
    # condition
    # → 정상 / 이동 / 회전 / 축소 조건
    #
    # filename
    # → 사용한 Target 파일명
    #
    # template_score
    # → 가장 비슷한 영역의 유사도 점수
    #
    # x, y
    # → 찾은 영역의 왼쪽 위 좌표
    #
    # center_x, center_y
    # → 찾은 영역의 중심 좌표
    #
    # time_ms
    # → Template Matching 처리시간(ms)
    #
    # rows.append()
    # → 현재 Target의 측정 결과를 리스트에 추가
    #
    # print()
    # → 터미널에 현재 결과를 출력
    #
    # for문이 반복되면서
    # 네 Target의 결과가 모두 rows에 저장됩니다.
    # --------------------------------------------------

    rows.append({
        "condition": condition,
        "filename": target_path.name,
        "template_score": f"{max_score:.6f}",
        "x": top_left[0],
        "y": top_left[1],
        "center_x": f"{center_x:.2f}",
        "center_y": f"{center_y:.2f}",
        "time_ms": f"{elapsed_ms:.3f}",
    })


    print(
        f"{condition:7s}",
        f"score={max_score:.4f}",
        f"location={top_left}",
        f"time={elapsed_ms:.2f} ms",
    )


# --------------------------------------------------
# 11. 네 개 Target의 측정 결과를 CSV 파일로 저장합니다.
#
# for문이 모두 끝난 후 실행됩니다.
#
# csv.DictWriter()
# → 딕셔너리 형태의 데이터를 CSV로 저장
#
# writer.writeheader()
# → CSV의 첫 번째 행에 항목 이름 작성
#
# writer.writerows(rows)
# → normal / shift / rot15 / scale90
#   네 조건의 측정 결과를 저장
#
# 최종 CSV:
# reports/day04_template_results.csv
#
# 저장한 CSV에서는
# 조건별 Score와 Location을 비교할 수 있습니다.
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
            "template_score",
            "x",
            "y",
            "center_x",
            "center_y",
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