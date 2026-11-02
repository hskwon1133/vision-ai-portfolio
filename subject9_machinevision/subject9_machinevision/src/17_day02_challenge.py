# %% 0. 준비
import cv2
import matplotlib
matplotlib.use("Agg")          # 창을 띄우지 않고 파일로만 저장하는 모드
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path("outputs/day02/challenge")
OUT.mkdir(parents=True, exist_ok=True)

# %% 1. 두 이미지를 흑백으로 읽기
on  = cv2.imread("samples/day02/cap_on_reference.jpg",  cv2.IMREAD_GRAYSCALE)
off = cv2.imread("samples/day02/cap_off_reference.jpg", cv2.IMREAD_GRAYSCALE)

if on is None or off is None:
    raise FileNotFoundError("이미지를 못 찾았어요. 프로젝트 최상위 폴더에서 실행했는지 확인하세요.")

print("ON 크기:", on.shape)
print("OFF 크기:", off.shape)

# %% 2. 나란히 비교 이미지 저장
plt.figure(figsize=(8, 4))
for i, (img, title) in enumerate([(on, "ON"), (off, "OFF")]):
    plt.subplot(1, 2, i + 1)
    plt.imshow(img, cmap="gray", vmin=0, vmax=255)
    plt.title(title)
    plt.axis("off")
plt.tight_layout()
plt.savefig(OUT / "step1_gray.png")
plt.close()
print("저장:", OUT / "step1_gray.png")