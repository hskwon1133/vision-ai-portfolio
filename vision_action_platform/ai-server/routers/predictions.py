import os
import shutil
import tempfile
import time

import cv2
from fastapi import APIRouter, File, HTTPException, Request, UploadFile

from services.action_inference import predict_tracks

router = APIRouter()

ALLOWED_EXT = {".mp4", ".mov", ".avi"}


@router.post("/predict/video")
def predict_video(request: Request, file: UploadFile = File(...)):
    started = time.perf_counter()

    # 1) 확장자 검사
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in ALLOWED_EXT:
        raise HTTPException(status_code=400, detail=f"지원하지 않는 형식입니다: {ext}")

    # 2) 임시 파일로 저장
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        # 3) 영상 열기 확인
        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="영상을 읽을 수 없습니다")
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()

        # 4) 사람별 추적 + Feature
        tracks = request.app.state.pose.extract_tracks(tmp_path, fps)

        # 5) 사람별 예측
        predictions = predict_tracks(tracks, request.app.state.bundle)

        message = None
        if not tracks:
            message = "영상에서 사람을 찾지 못했습니다"
        elif not predictions:
            message = "사람은 찾았지만 예측에 필요한 프레임이 부족합니다"

        return {
            "filename": file.filename,
            "frame_count": frame_count,
            "fps": round(fps, 2),
            "person_count": len(tracks),
            "predictions": predictions,
            "message": message,
            "processing_time_sec": round(time.perf_counter() - started, 2),
        }
    finally:
        os.remove(tmp_path)
