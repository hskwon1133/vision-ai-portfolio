# 이미지 한 장 불러오기
from pathlib import Path

import cv2


image_path = Path("samples/day01/sample01.jpg")
output_path = Path("outputs/read_image_check.jpg")

image = cv2.imread(str(image_path))

if image is None:
    raise FileNotFoundError(
        f"이미지를 읽지 못했습니다: {image_path}"
    )

output_path.parent.mkdir(parents=True, exist_ok=True)

saved = cv2.imwrite(str(output_path), image)

print("입력 :", image_path)
print("저장 :", output_path)
print("저장 성공:", saved)