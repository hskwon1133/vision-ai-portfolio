## python opencv 실행 환경 확인
import sys

import cv2
import numpy as np


print("Python :", sys.version.split()[0])
print("OpenCV:", cv2.__version__)
print("NumPy :", np.__version__)
print("Environment OK")