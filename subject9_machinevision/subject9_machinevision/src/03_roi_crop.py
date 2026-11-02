# ROI 지정하고 잘라내기
from pathlib import Path

import cv2


image_path = Path("samples/day01/sample01.jpg")
output_dir = Path("outputs/roi")

image = cv2.imread(str(image_path))

if image is None:
    raise FileNotFoundError(image_path)

height, width = image.shape[:2]

# 자신의 이미지에 맞게 수정합니다.
x1 = 200
y1 = 100
x2 = 700
y2 = 500

# 좌표가 이미지 범위를 벗어나지 않도록 제한합니다.
x1 = max(0, min(x1, width))
x2 = max(0, min(x2, width))
y1 = max(0, min(y1, height))
y2 = max(0, min(y2, height))

if x1 >= x2 or y1 >= y2:
    raise ValueError(
        f"잘못된 ROI 좌표입니다: {(x1, y1, x2, y2)}"
    )

roi = image[y1:y2, x1:x2]

output_dir.mkdir(parents=True, exist_ok=True)

roi_path = output_dir / "sample01_roi.jpg"
preview_path = output_dir / "sample01_roi_preview.jpg"

# 잘라낸 ROI 저장
cv2.imwrite(str(roi_path), roi)

# 원본에서 ROI 위치를 확인할 Preview 저장
preview = image.copy()
cv2.rectangle(
    preview,
    (x1, y1),
    (x2, y2),
    (0, 255, 0),
    3,
)

cv2.imwrite(str(preview_path), preview)

print("원본 크기 :", image.shape)
print("ROI 좌표  :", (x1, y1, x2, y2))
print("ROI 크기  :", roi.shape)
print("ROI 저장  :", roi_path)
print("Preview   :", preview_path)