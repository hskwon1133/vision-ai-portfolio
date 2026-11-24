from src.realtime_utils import TimedFeatureBuffer, predict_from_window

# 이 값보다 확률이 낮으면 "UNKNOWN"(판단 불확실)으로 표시
CONFIDENCE_THRESHOLD = 0.5


def predict_tracks(tracks: dict, bundle) -> list:
    steps = bundle.training_info["sequence_steps"]          # 30
    min_frames = bundle.training_info["min_valid_frames"]   # 10

    predictions = []
    for track_id, data in tracks.items():
        times = data["times"]
        features = data["features"]

        # 학습 기준: 유효 프레임이 10개 미만이면 예측하지 않음
        if len(features) < min_frames:
            continue

        # 영상 전체를 담도록 window를 영상 길이보다 길게 설정
        window = (times[-1] - times[0]) + 1.0
        buffer = TimedFeatureBuffer(window_seconds=window)
        for t, f in zip(times, features):
            buffer.append(t, f)

        try:
            label, raw_label, confidence, probs = predict_from_window(
                bundle.model, buffer, steps, CONFIDENCE_THRESHOLD
            )
        except ValueError:
            continue

        predictions.append({
            "track_id": track_id,
            "action_label": label,
            "raw_label": raw_label,
            "confidence": round(confidence, 4),
            "valid_frames": len(features),
            "probabilities": {
                c: round(float(p), 4)
                for c, p in zip(bundle.model.classes_, probs)
            },
        })
    return predictions
