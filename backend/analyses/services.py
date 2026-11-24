import os

import requests
from django.conf import settings


class AIServiceError(Exception):
    """FastAPI 호출 실패 (이유를 사용자에게 보여줄 메시지로 담음)"""


def request_video_prediction(video_path: str) -> dict:
    url = f"{settings.FASTAPI_URL}/predict/video"
    filename = os.path.basename(video_path)

    try:
        with open(video_path, "rb") as f:
            response = requests.post(
                url,
                files={"file": (filename, f, "video/mp4")},
                timeout=300,   # 최대 5분 기다림
            )
    except requests.ConnectionError:
        raise AIServiceError("AI 서버에 연결할 수 없습니다")
    except requests.Timeout:
        raise AIServiceError("AI 서버 응답 시간이 초과되었습니다")

    if response.status_code != 200:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        raise AIServiceError(f"AI 분석 실패 ({response.status_code}): {detail}")

    return response.json()
