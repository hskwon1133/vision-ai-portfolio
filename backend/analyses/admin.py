from django.contrib import admin

from .models import AnalysisJob, PersonPrediction


class PersonPredictionInline(admin.TabularInline):
    """작업 상세 화면 아래에 사람별 예측을 표로 같이 보여줌"""
    model = PersonPrediction
    extra = 0
    readonly_fields = ("track_id", "action_label", "confidence", "window_count", "created_at")


@admin.register(AnalysisJob)
class AnalysisJobAdmin(admin.ModelAdmin):
    list_display = ("id", "user", "input_type", "status", "created_at", "completed_at")
    list_filter = ("status", "input_type")
    search_fields = ("user__username", "title")
    inlines = [PersonPredictionInline]


@admin.register(PersonPrediction)
class PersonPredictionAdmin(admin.ModelAdmin):
    list_display = ("id", "job", "track_id", "action_label", "confidence", "created_at")
    list_filter = ("action_label",)
    search_fields = ("job__user__username",)