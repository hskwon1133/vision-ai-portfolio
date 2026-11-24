import os

from rest_framework import serializers

from .models import AnalysisJob, PersonPrediction

ALLOWED_EXT = {".mp4", ".mov", ".avi"}


class PersonPredictionSerializer(serializers.ModelSerializer):
    """사람 1명의 예측 결과 (읽기 전용)"""

    class Meta:
        model = PersonPrediction
        fields = ["track_id", "action_label", "confidence", "window_count", "created_at"]


class AnalysisJobSerializer(serializers.ModelSerializer):
    """분석 작업 생성·조회용"""

    predictions = PersonPredictionSerializer(many=True, read_only=True)

    class Meta:
        model = AnalysisJob
        fields = [
            "id", "title", "memo", "input_type", "video",
            "status", "error_message", "created_at", "completed_at",
            "predictions",
        ]
        # 사용자가 직접 바꿀 수 없는 칸 (서버가 채움)
        read_only_fields = ["status", "error_message", "created_at", "completed_at"]

    def validate(self, attrs):
        input_type = attrs.get("input_type")
        video = attrs.get("video")

        if input_type == AnalysisJob.InputType.UPLOAD:
            if not video:
                raise serializers.ValidationError({"video": "영상 업로드 방식은 영상 파일이 필요합니다."})
            ext = os.path.splitext(video.name)[1].lower()
            if ext not in ALLOWED_EXT:
                raise serializers.ValidationError({"video": f"지원하지 않는 형식입니다: {ext}"})

        if input_type == AnalysisJob.InputType.WEBCAM and video:
            raise serializers.ValidationError({"video": "IP Webcam 방식은 영상 파일을 보내지 않습니다."})

        return attrs


class AnalysisJobUpdateSerializer(serializers.ModelSerializer):
    """수정(PATCH)용: 제목과 메모만 바꿀 수 있음"""

    class Meta:
        model = AnalysisJob
        fields = ["title", "memo"]
