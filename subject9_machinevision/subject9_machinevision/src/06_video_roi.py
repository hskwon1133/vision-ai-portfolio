# 동영상 프레임에서 ROI 적용
from pathlib import Path

import cv2


video_path = Path("samples/sample.mp4")

output_dir = Path("outputs/video_roi")
output_dir.mkdir(parents=True, exist_ok=True)


# 검사할 ROI 좌표
# 실제 영상에 맞게 수정합니다.
x1 = 200
y1 = 100
x2 = 700
y2 = 500


cap = cv2.VideoCapture(str(video_path))

if not cap.isOpened():
    raise RuntimeError(
        f"영상을 열 수 없습니다: {video_path}"
    )


frame_count = 0
saved_count = 0


while True:

    ret, frame = cap.read()

    if not ret:
        break


    frame_count += 1

    height, width = frame.shape[:2]


    # ROI 좌표가 영상 밖으로 나가지 않도록 조정
    cx1 = max(0, min(x1, width))
    cx2 = max(0, min(x2, width))

    cy1 = max(0, min(y1, height))
    cy2 = max(0, min(y2, height))


    if cx1 >= cx2 or cy1 >= cy2:
        print("잘못된 ROI 좌표입니다.")
        break


    # 30 Frame마다 결과 저장
    if frame_count % 30 == 0:

        # ROI 자르기
        roi = frame[
            cy1:cy2,
            cx1:cx2
        ]


        # ROI 이미지 저장
        roi_path = (
            output_dir
            / f"frame_{frame_count:05d}_roi.jpg"
        )

        cv2.imwrite(
            str(roi_path),
            roi
        )


        # 원본 Frame 복사
        preview = frame.copy()


        # ROI 위치 표시
        cv2.rectangle(
            preview,
            (cx1, cy1),
            (cx2, cy2),
            (0, 255, 0),
            3
        )


        # ROI 표시된 전체 Frame 저장
        preview_path = (
            output_dir
            / f"frame_{frame_count:05d}_preview.jpg"
        )

        cv2.imwrite(
            str(preview_path),
            preview
        )


        saved_count += 1

        print(
            "저장:",
            roi_path.name
        )


cap.release()


print("전체 Frame:", frame_count)
print("ROI 저장 Frame:", saved_count)