import os

from celery import shared_task
from django.db import transaction
from django.utils import timezone

from .camera_capture import capture_clip
from .models import AnalysisJob, PersonPrediction
from .services import request_video_prediction


@shared_task
def run_analysis(job_id: int):
    job = AnalysisJob.objects.get(id=job_id)
    job.status = AnalysisJob.Status.PROCESSING
    job.save(update_fields=["status"])

    temp_path = None
    try:
        # 1) 영상 확보: 업로드 파일 또는 웹캠 수집
        if job.input_type == AnalysisJob.InputType.UPLOAD:
            video_path = job.video.path
        else:
            temp_path = capture_clip()
            video_path = temp_path

        # 2) FastAPI 추론 (업로드·웹캠 공통)
        result = request_video_prediction(video_path)

        # 3) 결과 저장
        with transaction.atomic():
            PersonPrediction.objects.bulk_create([
                PersonPrediction(
                    job=job,
                    track_id=p["track_id"],
                    action_label=p["action_label"],
                    confidence=p["confidence"],
                    window_count=1,
                )
                for p in result.get("predictions", [])
            ])
            job.status = AnalysisJob.Status.COMPLETED
            job.error_message = result.get("message") or ""
            job.completed_at = timezone.now()
            job.save(update_fields=["status", "error_message", "completed_at"])

    except Exception as e:
        job.status = AnalysisJob.Status.FAILED
        job.error_message = str(e)
        job.completed_at = timezone.now()
        job.save(update_fields=["status", "error_message", "completed_at"])

    finally:
        # 웹캠 임시 영상은 분석 후 삭제
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
