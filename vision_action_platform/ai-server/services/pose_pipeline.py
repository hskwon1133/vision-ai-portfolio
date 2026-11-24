from collections import defaultdict

from ultralytics import YOLO

from src.multi_person_utils import feature_from_keypoints


class PosePipeline:
    """영상 → 사람별(track_id) Feature 목록"""

    def __init__(self, project_config: dict, base_dir):
        # project.json의 설정을 학습 때와 똑같이 사용
        self.model = YOLO(str(base_dir / project_config["pose_model"]))
        self.imgsz = project_config["image_size"]
        self.person_conf = project_config["person_confidence"]

    def extract_tracks(self, video_path: str, fps: float) -> dict:
        tracks = defaultdict(lambda: {"times": [], "features": []})

        # stream=True: 프레임을 하나씩 처리 (메모리 절약)
        # 호출할 때마다 추적기가 새로 시작되어 영상끼리 track_id가 섞이지 않음
        results = self.model.track(
            source=video_path,
            stream=True,
            imgsz=self.imgsz,
            conf=self.person_conf,
            verbose=False,
        )

        for frame_idx, r in enumerate(results):
            # 이 프레임에 추적된 사람이 없으면 건너뜀
            if r.boxes is None or r.boxes.id is None or r.keypoints is None:
                continue

            timestamp = frame_idx / fps
            track_ids = r.boxes.id.int().cpu().tolist()
            keypoints_all = r.keypoints.xy.cpu().numpy()   # (사람 수, 17, 2)

            for track_id, keypoints in zip(track_ids, keypoints_all):
                feature = feature_from_keypoints(keypoints)
                if feature is None:
                    continue
                tracks[track_id]["times"].append(timestamp)
                tracks[track_id]["features"].append(feature)

        return dict(tracks)
