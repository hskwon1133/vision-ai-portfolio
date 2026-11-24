import tempfile
import time

import cv2
from django.conf import settings


class CameraError(Exception):
    """카메라 수집 실패 (사용자에게 보여줄 메시지)"""


def capture_clip() -> str:
    url = settings.IP_WEBCAM_URL
    seconds = settings.CAPTURE_SECONDS
    if not url:
        raise CameraError("카메라 주소가 설정되지 않았습니다")

    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        raise CameraError("카메라에 연결할 수 없습니다")

    frames = []
    fail_count = 0
    start = time.monotonic()
    try:
        # 정해진 시간(초) 동안만 읽고 멈춤
        while time.monotonic() - start < seconds:
            ok, frame = cap.read()
            if not ok:
                fail_count += 1
                if fail_count > 30:
                    raise CameraError("카메라 연결이 끊겼습니다")
                continue
            # 메모리 절약: 가로 640으로 줄임 (YOLO도 640으로 처리함)
            h, w = frame.shape[:2]
            frame = cv2.resize(frame, (640, int(h * 640 / w)))
            frames.append(frame)
    finally:
        cap.release()

    elapsed = time.monotonic() - start
    if len(frames) < 30:
        raise CameraError(f"수집된 프레임이 부족합니다 ({len(frames)}장)")

    # 실제로 모인 프레임 수로 FPS 계산
    fps = len(frames) / elapsed
    h, w = frames[0].shape[:2]

    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp.close()
    writer = cv2.VideoWriter(tmp.name, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames:
        writer.write(f)
    writer.release()
    return tmp.name
