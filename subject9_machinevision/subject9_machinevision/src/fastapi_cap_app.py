
"""스마트폰 IP Webcam → OpenCV 검사 → FastAPI 실시간 웹 화면."""

# ① 라이브러리 가져오기 · 프로젝트 경로 지정
# FastAPI, OpenCV, Pydantic 및 검사 모듈을 불러옵니다.
from contextlib import asynccontextmanager
import ipaddress
from pathlib import Path
import threading
import time
from urllib.parse import urlsplit

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from src.cap_inspector import CapRule, inspect_roi, load_rule

ROOT = Path(__file__).resolve().parents[1]
REFERENCE_DIR = ROOT / "samples" / "day03" / "reference"
HTML_FILE = ROOT / "web" / "cap_inspection.html"


# ② 대기 화면 이미지 준비
# 스마트폰 영상이 연결되기 전 보여 줄 JPEG를 만듭니다.
def placeholder_jpg() -> bytes:
    frame = np.full((480, 640, 3), (38, 43, 50), dtype=np.uint8)
    cv2.putText(frame, "IP WEBCAM - WAITING", (65, 227),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (230, 230, 230), 2)
    return cv2.imencode(".jpg", frame)[1].tobytes()


WAITING_JPG = placeholder_jpg()


# ③ 실시간 검사 상태를 관리하는 클래스
# 카메라 주소, ROI, 기준값, 판정 결과, 최신 프레임 등을 보관합니다.
class LiveState:
    """한 대의 카메라를 단일 스레드에서 읽고 여러 브라우저에 공유합니다."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.url: str | None = None
        self.roi: tuple[float, float, float, float] | None = None
        self.rule: CapRule | None = None
        self.rule_error = "기준 이미지 읽기 전"
        self.camera_status = "NOT_CONNECTED"
        self.message = "스마트폰 IP Webcam 주소를 연결하세요."
        self.decision = "WAIT"
        self.reason = ""
        self.ratio: float | None = None
        self.frame_size: list[int] | None = None
        self.latest_jpg: bytes = WAITING_JPG
        self.last_frame_time = 0.0
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None

    # ④ 기준 이미지로 검사 규칙 준비/갱신
    # cap_on·cap_off 이미지에서 판정 기준값을 계산합니다.
    def reload_rule(self) -> dict:
        try:
            rule = load_rule(REFERENCE_DIR)
        except (ValueError, FileNotFoundError) as exc:
            with self.lock:
                self.rule = None
                self.rule_error = str(exc)
                self.decision = "WAIT"
            return {"ok": False, "error": str(exc)}
        with self.lock:
            self.rule = rule
            self.rule_error = ""
            self.decision = "WAIT"
        return {"ok": True, "threshold": rule.threshold,
                "cap_on_ratio": rule.cap_on_ratio,
                "cap_off_ratio": rule.cap_off_ratio}

    # ⑤ 카메라 작업 시작·종료
    # 별도의 스레드를 시작하고 서버 종료 시 정리합니다.
    def start(self) -> None:
        self.stop_event.clear()
        self.reload_rule()
        self.thread = threading.Thread(target=self.camera_loop, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=3)

    # ⑥ 카메라 주소와 ROI 선택값 저장
    # 카메라를 바꾸면 기존 ROI와 검사 결과를 초기화합니다.
    def set_camera(self, url: str) -> None:
        with self.lock:
            self.url = url
            self.roi = None  # 새 영상에 이전 ROI를 적용하지 않습니다.
            self.camera_status = "CONNECTING"
            self.message = "카메라 연결 중입니다."
            self.decision = "WAIT"
            self.ratio = None
            self.latest_jpg = WAITING_JPG
            self.last_frame_time = 0.0

    def set_roi(self, roi: tuple[float, float, float, float] | None) -> None:
        with self.lock:
            self.roi = roi
            self.decision = "WAIT"
            self.ratio = None
            self.reason = ""

    # ⑦ 현재 검사 상태 조회
    # 카메라 연결, 판정값, Area Ratio, 기준값 등을 반환합니다.
    def get_status(self) -> dict:
        with self.lock:
            elapsed = (time.monotonic() - self.last_frame_time
                       if self.last_frame_time else None)
            camera_status = self.camera_status
            if camera_status == "ONLINE" and elapsed is not None and elapsed > 5:
                camera_status = "DISCONNECTED"
            rule = self.rule
            return {
                "camera_status": camera_status,
                "message": self.message,
                "decision": self.decision if camera_status == "ONLINE" else "WAIT",
                "reason": self.reason,
                "ratio": self.ratio,
                "roi": self.roi,
                "frame_size": self.frame_size,
                "rule_ready": rule is not None,
                "rule_error": self.rule_error,
                "threshold": round(rule.threshold, 5) if rule else None,
                "on_ratio": round(rule.cap_on_ratio, 5) if rule else None,
                "off_ratio": round(rule.cap_off_ratio, 5) if rule else None,
            }

    # ⑧ IP Webcam 연결 및 연결 실패 처리
    # OpenCV로 스마트폰 영상 URL에 접속하고 재연결을 시도합니다.
    def camera_loop(self) -> None:
        while not self.stop_event.is_set():
            with self.lock:
                url = self.url
            if not url:
                self.stop_event.wait(0.1)
                continue

            # 한 스레드에서만 VideoCapture를 엽니다.
            cap = cv2.VideoCapture(url)
            if not cap.isOpened():
                cap.release()
                with self.lock:
                    if self.url == url:
                        self.camera_status = "DISCONNECTED"
                        self.message = "IP Webcam 연결 실패: URL과 Wi-Fi를 확인하세요."
                self.stop_event.wait(1.0)
                continue

            try:
                while not self.stop_event.is_set():
                    with self.lock:
                        if self.url != url:
                            break
                        roi_coords = self.roi
                        rule = self.rule

                    # ⑨ 실시간 Frame 읽기
                    # 새 프레임을 읽고, 실패하면 상태를 갱신한 뒤 재연결합니다.
                    ok, frame = cap.read()
                    if not ok or frame is None:
                        with self.lock:
                            if self.url == url:
                                self.camera_status = "DISCONNECTED"
                                self.message = "프레임 수신 중단: 재연결을 시도합니다."
                                self.decision = "WAIT"
                                self.latest_jpg = WAITING_JPG
                        break

                    h, w = frame.shape[:2]
                    image = frame.copy()
                    decision = "WAIT"
                    reason = ""
                    ratio = None

                    # ⑩ ROI Crop 후 핵심 검사 함수 호출
                    # 선택한 영역을 잘라 cap_inspector.inspect_roi()로 판정합니다.
                    if roi_coords is not None:
                        nx1, ny1, nx2, ny2 = roi_coords
                        x1, x2 = round(nx1 * w), round(nx2 * w)
                        y1, y2 = round(ny1 * h), round(ny2 * h)
                        x1, x2 = max(0, x1), min(w, x2)
                        y1, y2 = max(0, y1), min(h, y2)
                        if x2 > x1 and y2 > y1:
                            region = frame[y1:y2, x1:x2]
                            if rule is not None:
                                try:
                                    result = inspect_roi(region, rule)
                                    decision = result["decision"]
                                    reason = result["reason"]
                                    ratio = result["ratio"]
                                except (ValueError, cv2.error):
                                    decision = "CHECK"
                                    reason = "ROI_PROCESS_FAILED"
                            color = ((0, 190, 0) if decision == "OK"
                                     else (0, 0, 225) if decision == "NG"
                                     else (0, 200, 255))
                            cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
                    else:
                        cv2.putText(image, "SELECT ROI ON WEB PAGE", (15, 40),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 230, 255), 2)

                    # ⑪ 검사 화면 표시 · JPEG 변환 · 최신 결과 저장
                    # 판정 문자와 ROI 표시가 포함된 Frame을 브라우저용으로 준비합니다.
                    line = f"{decision}  Area Ratio: {ratio:.3f}" if ratio is not None else decision
                    cv2.putText(image, line, (12, h - 18),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 245, 255), 2)
                    success, encoded = cv2.imencode(
                        ".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 77]
                    )
                    if success:
                        with self.lock:
                            # 웹에서 ROI/카메라를 방금 변경했다면 이전 프레임 결과는 폐기합니다.
                            if self.url == url and self.roi == roi_coords and self.rule is rule:
                                self.latest_jpg = encoded.tobytes()
                                self.camera_status = "ONLINE"
                                self.message = "영상 수신 중"
                                self.decision = decision
                                self.reason = reason
                                self.ratio = ratio
                                self.frame_size = [w, h]
                                self.last_frame_time = time.monotonic()
            finally:
                cap.release()
            self.stop_event.wait(0.1)


# ⑫ 서버 실행 준비 및 종료 처리(Lifespan)
# FastAPI가 시작되면 카메라 스레드를 시작하고, 종료 시 중지합니다.
state = LiveState()


@asynccontextmanager
async def lifespan(_: FastAPI):
    state.start()
    yield
    state.stop()


app = FastAPI(title="교과 9 병뚜껑 실시간 검사", lifespan=lifespan)


# ⑬ HTTP 요청 Body 검증(Pydantic)
# 카메라 URL과 ROI 좌표의 자료형·허용 범위를 확인합니다.
class CameraRequest(BaseModel):
    url: str


class ROIRequest(BaseModel):
    x1: float = Field(ge=0.0, le=1.0)
    y1: float = Field(ge=0.0, le=1.0)
    x2: float = Field(ge=0.0, le=1.0)
    y2: float = Field(ge=0.0, le=1.0)


# ⑭ GET / : 브라우저에 HTML 화면 전달
@app.get("/")
def index():
    return FileResponse(HTML_FILE, media_type="text/html")


# ⑮ POST /api/camera : 카메라 URL 확인 후 연결 시도
# HTTP 형식, IP 주소, 포트 등을 검사합니다.
@app.post("/api/camera")
def connect_camera(body: CameraRequest):
    url = body.url.strip()
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise HTTPException(400, "http(s)://IP:PORT/video 형식으로 입력하세요.")
    if parts.username or parts.password:
        raise HTTPException(400, "사용자 인증이 없는 교육용 IP Webcam 주소만 사용하세요.")
    try:
        addr = ipaddress.ip_address(parts.hostname)
        if not (addr.is_private or addr.is_loopback):
            raise ValueError("사설 IP 주소가 아닙니다")
        _ = parts.port  # 숫자가 아닌 포트는 거부
    except ValueError:
        raise HTTPException(400, "스마트폰의 사설 IP 주소(예: 192.168.x.x)를 사용하세요.")
    state.set_camera(url)
    return {"ok": True, "message": "연결을 시도합니다."}


# ⑯ POST /api/roi : 브라우저에서 선택한 ROI 좌표 저장
# 너무 작은 ROI는 거부합니다.
@app.post("/api/roi")
def set_roi(body: ROIRequest):
    if body.x2 - body.x1 < 0.03 or body.y2 - body.y1 < 0.03:
        raise HTTPException(400, "ROI가 너무 작습니다. 화면에서 더 넓게 드래그하세요.")
    state.set_roi((body.x1, body.y1, body.x2, body.y2))
    return {"ok": True, "roi": body.model_dump()}


# ⑰ DELETE /api/roi : ROI 선택 초기화
@app.delete("/api/roi")
def reset_roi():
    state.set_roi(None)
    return {"ok": True}


# ⑱ POST /api/reload-rule : 정상·불량 기준값 다시 계산
@app.post("/api/reload-rule")
def reload_rule():
    result = state.reload_rule()
    if not result["ok"]:
        raise HTTPException(400, result["error"])
    return result


# ⑲ GET /api/status : 최신 상태를 JSON으로 전달
@app.get("/api/status")
def get_status():
    return state.get_status()


# ⑳ GET /stream.mjpg : HTTP MJPEG 실시간 영상 전송
# 새 JPEG가 준비될 때마다 브라우저에 연속 응답합니다.
@app.get("/stream.mjpg")
def stream_video():
    def frames():
        while True:
            with state.lock:
                image_bytes = state.latest_jpg
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n"
                   b"Cache-Control: no-cache\r\n\r\n"
                   + image_bytes + b"\r\n")
            time.sleep(0.10)  # 브라우저에 최대 약 10회/초 전송

    return StreamingResponse(
        frames(), media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"},
    )




