# FastAPI Beans ResNet18 동기·비동기 평가 프로젝트

콩잎 이미지를 `angular_leaf_spot` / `bean_rust` / `healthy`로 분류하는 ResNet18 Fine-tuning 모델을 FastAPI 동기 API와 Celery 비동기 작업 큐로 서빙한다.

## 1. 환경

- Windows + WSL2 Ubuntu, Docker Desktop (WSL Integration) / Docker Compose
- Docker 서비스: `web`(FastAPI :8000), `worker`(Celery), `redis`(:6379), `postgres`(:5432)
- 학습: WSL `.venv` (Python 3.14) / 서빙 컨테이너: Python 3.11
- 추론 장치: CPU

## 2. 데이터 준비

1. [AI-Lab-Makerere/beans](https://huggingface.co/datasets/AI-Lab-Makerere/beans)의 `data/train.zip`, `validation.zip`, `test.zip`을 받는다.
2. 원본 `validation/`의 각 클래스 이미지를 코드 실행용 `dataset/val/`에 배치한다.

```text
dataset/
├── train/{angular_leaf_spot, bean_rust, healthy}/
├── val/{angular_leaf_spot, bean_rust, healthy}/
└── test/{angular_leaf_spot, bean_rust, healthy}/
```

3. 개수 확인: `for d in dataset/*/*; do echo "$d $(ls "$d" | wc -l)"; done`
4. 데이터셋은 공개·배포 권한을 확인한 뒤 공유한다 (이 저장소에는 포함하지 않음).

## 3. 모델 준비

```bash
python -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements.txt
python -m training.finetune --data dataset --output artifacts/beans_resnet18 --epochs 10 --batch-size 8 --lr 0.001 --freeze-backbone
```

- 결과: `artifacts/beans_resnet18/` (`model.safetensors`, `config.json`, `preprocessor_config.json`, `training_report.json`)
- Test 평가: `python -m training.evaluate_test --model artifacts/beans_resnet18 --data dataset/test`
- `.env`: `MODEL_PATH=artifacts/beans_resnet18` (`.env.example` 참고)

## 4. 구동 순서

```bash
# 1. Docker Desktop 실행 후
docker compose up -d --build
docker compose ps                              # web, worker, redis, postgres 확인
docker compose exec redis redis-cli ping       # PONG
docker compose logs --tail=30 worker           # celery@... ready.
curl http://127.0.0.1:8000/health              # {"status":"ok"}
```

> 로컬에서 `uvicorn`이 8000번 포트를 쓰고 있으면 web 컨테이너가 시작되지 않으므로 먼저 종료한다.

| 주소 | 내용 |
|---|---|
| http://127.0.0.1:8000/vision/sync/ | 동기 분류 화면 |
| http://127.0.0.1:8000/vision/async/ | 비동기 분류 화면 |
| http://127.0.0.1:8000/docs | API 문서 |

## 5. 확인할 응답

- `/health`: `{"status":"ok"}`
- 동기 POST `/api/sync/classify/`: 200, `top1`, `all`, `inference_ms`, `cache_hit`
- 비동기 POST `/api/async/classify/`: 202, `job_id`, `check_url` → GET `/api/async/jobs/{id}/`: `status=done`
- PostgreSQL `image_jobs`: 비동기 작업 기록 (DBeaver: Host 127.0.0.1, Port 5432)

## 6. 테스트·측정

```bash
docker compose exec web python -m pytest -q tests/test_api.py          # 모의 테스트
python scripts/measure_timings.py dataset/test/healthy/healthy_test.40.jpg   # 성능 측정 → reports/data/timings.csv
```

## 7. 변경·종료

```bash
docker compose up -d --build                          # 코드·Dockerfile·템플릿·tests 변경 시
docker compose up -d --force-recreate web worker      # .env / MODEL_PATH 변경 시 (web·worker 둘 다)
docker compose logs -f --tail=100 web worker          # 로그
docker compose stop                                   # 정상 중지 (볼륨 보존)
docker compose up -d                                  # 재시작
```

`docker compose down -v`는 PostgreSQL·Redis 볼륨 데이터를 지우므로 사용하지 않는다. `media/jobs/`와 `artifacts/`는 별도로 보관한다.

## 8. 결과물 위치

| 내용 | 위치 |
|---|---|
| 산출물 1~8 | `reports/01_*.md` ~ `reports/08_*.md` |
| 원본 데이터 | `reports/data/` (`timings.csv`, `image_jobs_rows.csv`, 실험별 `training_report.json`, `test_eval_beans_resnet18.txt`) |
| 실행 증빙 | `reports/evidence/` |
| 학습 결과 | `artifacts/beans_resnet18/training_report.json` |

## 9. 제출 및 보안

- `.env`, 비밀번호, API 키는 GitHub에 올리지 않는다 (`.env.example`만 포함).
- 데이터셋·모델 가중치 같은 대용량 파일은 허용된 제출 경로로 따로 전달하고, 받는 사람이 `dataset/`과 `artifacts/`에 배치하는 절차는 위 2·3장을 따른다.
