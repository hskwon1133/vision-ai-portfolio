from django.db import transaction

from .tasks import run_analysis

from rest_framework import viewsets
from rest_framework.decorators import action
from rest_framework.response import Response

from .models import AnalysisJob
from .serializers import AnalysisJobSerializer, AnalysisJobUpdateSerializer


class AnalysisJobViewSet(viewsets.ModelViewSet):
    # PUT(전체 수정)은 막고 PATCH(부분 수정)만 허용
    http_method_names = ["get", "post", "patch", "delete", "head", "options"]

    def get_queryset(self):
        # 로그인한 사용자의 작업만 조회
        return (
            AnalysisJob.objects
            .filter(user=self.request.user)
            .prefetch_related("predictions")
            .order_by("-created_at")
        )

    def get_serializer_class(self):
        if self.action == "partial_update":
            return AnalysisJobUpdateSerializer
        return AnalysisJobSerializer

    def perform_create(self, serializer):
        job = serializer.save(user=self.request.user)

        # DB 저장이 "확정된 뒤"에 Celery 작업 등록
        def enqueue():
            result = run_analysis.delay(job.id)
            job.celery_task_id = result.id
            job.save(update_fields=["celery_task_id"])

        transaction.on_commit(enqueue)

    @action(detail=True, methods=["get"], url_path="status")
    def job_status(self, request, pk=None):
        job = self.get_object()
        return Response({
            "id": job.id,
            "status": job.status,
            "error_message": job.error_message,
            "is_done": job.status in ("COMPLETED", "FAILED"),
        })