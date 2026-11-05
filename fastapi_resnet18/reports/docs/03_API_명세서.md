# 03. API 명세서

> 기준 서버: `http://127.0.0.1:8000` (Docker `web` 컨테이너, FastAPI)
> 모델: `artifacts/beans_resnet18` (콩잎 3클래스: `angular_leaf_spot`, `bean_rust`, `healthy`)

## 1. API 목록

| API ID | Method | URL                         | 요청 형태                           | 성공 HTTP | 주요 응답                                                                                       | 실패 HTTP                                             |
| ------ | ------ | --------------------------- | ------------------------------- | ------: | ------------------------------------------------------------------------------------------- | --------------------------------------------------- |
| API-01 | GET    | `/health`                   | 없음                              |     200 | `status`                                                                                    | 서버 미실행 시 HTTP 응답 없음 (`curl: (7) Failed to connect`) |
| API-02 | POST   | `/api/sync/classify/`       | `multipart/form-data` / `image` |     200 | `model_id`, `top1`, `all`, `inference_ms`, `cache_hit`                                      | 400, 413                                            |
| API-03 | POST   | `/api/async/classify/`      | `multipart/form-data` / `image` |     202 | `job_id`, `status`, `check_url`                                                             | 400, 413, 503                                       |
| API-04 | GET    | `/api/async/jobs/{job_id}/` | 정수형 `job_id`                    |     200 | `job_id`, `status`, `model_id`, `top1`, `all`, `inference_ms`, `cache_hit`, `error_message` | 404                                                 |
| WEB-01 | GET    | `/`                         | 없음                              |     307 | `/vision/sync/`로 이동                                                                         | —                                                   |
| WEB-02 | GET    | `/vision/sync/`             | 없음                              |     200 | 동기 분류 HTML                                                                                  | —                                                   |
| WEB-03 | GET    | `/vision/async/`            | 없음                              |     200 | 비동기 분류 HTML (1초 간격 상태 조회)                                                                   | —                                                   |
| DOC-01 | GET    | `/docs`                     | 없음                              |     200 | Swagger UI                                                                                  | —                                                   |

**의미 구분:** `202`는 **작업 접수**이며 분석 완료가 아니다. 완료 여부는 API-04 응답의 `status`가 `done`(또는 `failed`)인지로 판단한다.

---

## 2. API-02 동기 분류

| 항목 | 내용 |
|---|---|
| URL | `/api/sync/classify/` |
| Method | POST |
| Content-Type | `multipart/form-data` |
| 요청 필드 | `image`: JPEG / PNG / WEBP, 5MB 이하, 최대 1,200만 픽셀 |
| 처리 | `router_sync.classify_sync()` → `cache_service.classify_with_cache()` → MISS 시 `services.classify_image_bytes()` |
| 성공 | 200 OK — 추론이 끝난 뒤 결과 포함 |
| 오류 | 400 이미지가 아니거나 형식 불가 / 413 5MB 초과 |
| DB 기록 | 없음 (`image_jobs`에 저장하지 않음, 결과는 Redis 캐시 600초) |

**요청 (실제 실행)**

```bash
curl -i -X POST -F "image=@dataset/test/healthy/healthy_test.40.jpg" http://127.0.0.1:8000/api/sync/classify/
```

**성공 응답 (실제, 2026-11-04 16:05)**

```text
HTTP/1.1 200 OK
server: uvicorn
content-length: 295
content-type: application/json
```

```json
{
  "model_id": "artifacts/beans_resnet18",
  "top1": {"label": "healthy", "score": 0.6759311556816101},
  "all": [
    {"label": "healthy", "score": 0.6759311556816101},
    {"label": "bean_rust", "score": 0.21172815561294556},
    {"label": "angular_leaf_spot", "score": 0.11234068870544434}
  ],
  "inference_ms": 21.71,
  "cache_hit": false
}
```

**캐시 HIT 응답 형태 (실제 측정, `timings.csv` sync HIT)**: `"inference_ms": null`, `"cache_hit": true` — 모델 추론을 생략했다는 뜻이며 0ms가 아니다.

---

## 3. API-03 비동기 작업 접수

| 항목 | 내용 |
|---|---|
| URL | `/api/async/classify/` |
| Method | POST |
| Content-Type | `multipart/form-data` |
| 요청 필드 | `image` (API-02와 같은 검증) |
| 처리 | 검증 → `media/jobs/{uuid}.bin` 저장 → `image_jobs` 행 생성(`queued`) → `classify_job.delay(job_id)` |
| 성공 | **202 Accepted** — 접수만 완료 |
| 오류 | 400 / 413 / **503** (Redis 브로커 연결 실패로 큐 등록 불가, 해당 행은 `failed` 기록) |

**요청·응답 (실제, job 6)**

```bash
curl -i -X POST -F "image=@dataset/test/healthy/healthy_test.40.jpg" http://127.0.0.1:8000/api/async/classify/
```

```text
HTTP/1.1 202 Accepted
server: uvicorn
content-length: 63
content-type: application/json
```

```json
{"job_id": 6, "status": "queued", "check_url": "/api/async/jobs/6/"}
```

---

## 4. API-04 작업 상태 조회

| 항목 | 내용 |
|---|---|
| URL | `/api/async/jobs/{job_id}/` |
| Method | GET |
| 경로 변수 | `job_id` (정수) |
| 성공 | 200 OK — `status`는 `queued` / `running` / `done` / `failed` |
| 결과 필드 | `done`일 때만 `top1`, `all` 채움 / `failed`일 때만 `error_message` 채움 |
| 오류 | 404 없는 `job_id` |

**완료 응답 (실제, job 6 — 동기 요청이 저장한 캐시를 재사용)**

```json
{
  "job_id": 6,
  "status": "done",
  "model_id": "artifacts/beans_resnet18",
  "top1": {"label": "healthy", "score": 0.6759311556816101},
  "all": [
    {"label": "healthy", "score": 0.6759311556816101},
    {"label": "bean_rust", "score": 0.21172815561294556},
    {"label": "angular_leaf_spot", "score": 0.11234068870544434}
  ],
  "inference_ms": null,
  "cache_hit": true,
  "error_message": null
}
```

**완료 응답 (실제, job 1 — worker 첫 추론)**

```json
{
  "job_id": 1,
  "status": "done",
  "model_id": "artifacts/beans_resnet18",
  "top1": {"label": "bean_rust", "score": 0.7002041935920715},
  "all": [
    {"label": "bean_rust", "score": 0.7002041935920715},
    {"label": "angular_leaf_spot", "score": 0.27035287022590637},
    {"label": "healthy", "score": 0.029442884027957916}
  ],
  "inference_ms": 2148.14,
  "cache_hit": false,
  "error_message": null
}
```

> 응답 필드명 `all`은 API 응답 이름이며, DB 컬럼명은 `all_scores`다. 두 이름을 바꿔 쓰지 않는다.

---

## 5. 오류 응답

| API | 상황 | HTTP | 응답 본문 (코드 기준) | 실환경 확인 |
|---|---|---:|---|---|
| API-02/03 | 이미지가 아닌 파일 (`.txt`) | 400 | `{"detail": "올바른 이미지 파일이 아닙니다."}` | [실행 후 기입] |
| API-02/03 | 5MB 초과 | 413 | `{"detail": "5MB 이하 파일만 허용합니다."}` | [실행 후 기입] |
| API-03 | Redis 브로커 연결 실패 | 503 | `{"detail": "작업 큐 등록 실패: Redis 연결을 확인하세요."}` | 미실시 (모의 테스트로만 확인) |
| API-04 | 없는 `job_id` | 404 | `{"detail": "해당 작업이 없습니다."}` | [실행 후 기입] |

실환경 확인 명령:

```bash
echo "hello" > /tmp/bad.txt
curl -s -w "\nHTTP %{http_code}\n" -X POST -F "image=@/tmp/bad.txt" http://127.0.0.1:8000/api/sync/classify/
curl -s -w "\nHTTP %{http_code}\n" http://127.0.0.1:8000/api/async/jobs/999999/
head -c 6000000 /dev/urandom > /tmp/big.bin
curl -s -w "\nHTTP %{http_code}\n" -X POST -F "image=@/tmp/big.bin" http://127.0.0.1:8000/api/sync/classify/
```

> 400·413·404·503은 모의 테스트(`tests/test_api.py`, 4 passed)에서 상태코드를 검증했다. 모의 테스트는 실환경 통합 테스트와 구분해 기록한다 → [산출물 7](07_기능테스트_오류분석.md).

---

## 6. 실행 근거표

| 검증 ID | 테스트 입력 | 실제 상태코드 | 결과 | 첨부 증빙 |
|---|---|---:|---|---|
| API-01-T01 | `GET /health` | 200 | `{"status":"ok"}` | `evidence/health.png` |
| API-02-T01 | `healthy_test.40.jpg` | 200 | healthy 0.6759, `cache_hit false`, 21.71ms | `evidence/api_sync_200.png` |
| API-02-T02 | 같은 이미지 재요청 (측정 스크립트) | 200 | `cache_hit true`, `inference_ms null` | `data/timings.csv` |
| API-03-T01 | `healthy_test.40.jpg` | 202 | `job_id 6`, `queued` | `evidence/api_async_202.png` |
| API-04-T01 | `job_id=6` | 200 | `done`, healthy, `cache_hit true` | `evidence/cache_hit.png` |
| API-04-T02 | `job_id=1` | 200 | `done`, bean_rust 0.7002 | `evidence/api_job_done.png` |
| API-04-T03 | `job_id=999999` | [ ] | [ ] | `evidence/api_errors.png` |
| API-02-T03 | `.txt` 파일 | [ ] | [ ] | `evidence/api_errors.png` |
| API-02-T04 | 6MB 파일 | [ ] | [ ] | `evidence/api_errors.png` |
