from django.conf import settings
from django.db import models


class AnalysisJob(models.Model):
    """분석 요청 1건 (영상 업로드 1번 또는 웹캠 수집 1번)"""

    class InputType(models.TextChoices):
        UPLOAD = "UPLOAD", "영상 업로드"
        WEBCAM = "WEBCAM", "IP Webcam"

    class Status(models.TextChoices):
        PENDING = "PENDING", "대기"
        PROCESSING = "PROCESSING", "처리 중"
        COMPLETED = "COMPLETED", "완료"
        FAILED = "FAILED", "실패"

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,        # Django 기본 User
        on_delete=models.CASCADE,
        related_name="analysis_jobs",
    )
    title = models.CharField(max_length=100, blank=True)
    memo = models.TextField(blank=True)
    input_type = models.CharField(max_length=10, choices=InputType.choices)
    video = models.FileField(upload_to="videos/", blank=True)
    status = models.CharField(
        max_length=20, choices=Status.choices, default=Status.PENDING
    )
    celery_task_id = models.CharField(max_length=255, blank=True)
    error_message = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    completed_at = models.DateTimeField(null=True, blank=True)

    def __str__(self):
        return f"Job {self.id} ({self.status})"


class PersonPrediction(models.Model):
    """분석 작업 안에서 사람 1명의 예측 결과"""

    job = models.ForeignKey(
        AnalysisJob,
        on_delete=models.CASCADE,
        related_name="predictions",
    )
    track_id = models.IntegerField()
    action_label = models.CharField(max_length=50)
    confidence = models.FloatField()
    window_count = models.IntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Job {self.job_id} / Track {self.track_id}: {self.action_label}"