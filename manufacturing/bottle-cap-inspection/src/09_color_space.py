# 색공간 비교
# openCV 내부에서는 기본값 = BGR

'''
BGR / GRAY / HSV 
정상 이미지 몇 장

불량 이미지 몇 장

밝은 조건

어두운 조건

반사가 있는 조건

제품 위치가 조금 다른 조건

cap_on_reference.jpg
→ 기준 영상

challenge.jpg
→ 어둡고 저대비 조건

glare.jpg
→ 강한 조명 조건

dark.jpg
low_contrast.jpg
noise.jpg
blur.jpg
'''

from pathlib import Path

import cv2


image_path = Path("samples/day02/cap_on_reference.jpg")
output_dir = Path("outputs/day02/color")
output_dir.mkdir(parents=True, exist_ok=True)

image = cv2.imread(str(image_path))

if image is None:
    raise FileNotFoundError(
        f"이미지를 읽지 못했습니다: {image_path}"
    )

gray = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2GRAY,
)

hsv = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2HSV,
)

h, s, v = cv2.split(hsv)

cv2.imwrite(
    str(output_dir / "original_bgr.jpg"),
    image,
)

cv2.imwrite(
    str(output_dir / "gray.jpg"),
    gray,
)

cv2.imwrite(
    str(output_dir / "h_channel.jpg"),
    h,
)

cv2.imwrite(
    str(output_dir / "s_channel.jpg"),
    s,
)

cv2.imwrite(
    str(output_dir / "v_channel.jpg"),
    v,
)

print("입력 :", image_path)
print("저장 :", output_dir)
print("Color Space 결과 저장 완료")