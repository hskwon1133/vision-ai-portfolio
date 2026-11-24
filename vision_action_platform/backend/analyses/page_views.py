from django.contrib.admin.views.decorators import staff_member_required
from django.contrib.auth.decorators import login_required
from django.db.models import Count
from django.shortcuts import render

from .models import AnalysisJob, PersonPrediction


@login_required
def index(request):
    """분석 화면 (로그인한 사람만)"""
    return render(request, "analyses/index.html")


@staff_member_required
def dashboard(request):
    """관리자 대시보드 (is_staff만)"""
    # 동작별 예측 건수
    action_counts = (
        PersonPrediction.objects
        .values("action_label")
        .annotate(count=Count("id"))
        .order_by("action_label")
    )
    # 작업 상태별 건수
    status_counts = (
        AnalysisJob.objects
        .values("status")
        .annotate(count=Count("id"))
        .order_by("status")
    )
    recent_jobs = AnalysisJob.objects.select_related("user").order_by("-created_at")[:10]

    chart_data = {
        "actions": {
            "labels": [r["action_label"] for r in action_counts],
            "counts": [r["count"] for r in action_counts],
        },
        "statuses": {
            "labels": [r["status"] for r in status_counts],
            "counts": [r["count"] for r in status_counts],
        },
    }
    return render(request, "analyses/dashboard.html", {
        "chart_data": chart_data,
        "recent_jobs": recent_jobs,
    })
