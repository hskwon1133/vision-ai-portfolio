# 01. ResNet18 콩잎 3클래스 Fine-tuning 결과보고서

| 항목                          | 내용                                                                      |
| --------------------------- | ----------------------------------------------------------------------- |
| 평가 대상                       | 콩잎 질병 이미지 3클래스 분류 (Image Classification)                                |
| 기준 모델                       | Hugging Face `microsoft/resnet-18` (ImageNet 사전학습 가중치)                  |
| 출력층                         | ImageNet 1,000클래스 → 3클래스로 교체 (`ignore_mismatched_sizes=True`)           |
| 최종 학습 방식                    | **Head-only Fine-tuning** (Backbone 고정, 분류 헤드만 학습, `--freeze-backbone`) |
| 학습 코드                       | `training/finetune.py` (01 교재 제공 코드, 수정 없음) — 실행 옵션만 변경                 |
| 입력 데이터                      | `dataset/train` (학습), `dataset/val` (검증·best 모델 선택)                     |
| 보관 데이터                      | `dataset/test` (학습·모델 선택에 사용하지 않음)                                      |
| 최종 모델 위치                    | `artifacts/beans_resnet18/`                                             |
| 최종 Best Validation Accuracy | **0.9398** (125 / 133, epoch 6)                                         |
| 실험 상태                       | 실행 완료                                                                   |

---

## 1. 데이터 구성

### 1-1. 원본 데이터와 실행용 데이터 매핑

| 구분     | 내용                                                                                                                                 |
| ------ | ---------------------------------------------------------------------------------------------------------------------------------- |
| 출처     | [AI-Lab-Makerere/beans](https://huggingface.co/datasets/AI-Lab-Makerere/beans) `data/` 의 `train.zip`, `validation.zip`, `test.zip` |
| 원본 폴더  | `train/` · `validation/` · `test/` (각각 3개 클래스 폴더) — 원본은 보존                                                                         |
| 실행용 폴더 | `dataset/train/` · `dataset/val/` · `dataset/test/`                                                                                |
| 매핑     | `train → dataset/train`, **`validation → dataset/val`** (제공 코드가 `val` 경로를 읽음), `test → dataset/test`                               |
| 정답 라벨  | 폴더 이름을 그대로 따름 (학습 코드가 `sorted()`로 폴더명을 클래스로 읽음)                                                                                    |

| 클래스 | 의미 | 모델 id |
|---|---|---:|
| `angular_leaf_spot` | 각진 반점병 | 0 |
| `bean_rust` | 녹병 | 1 |
| `healthy` | 건강한 잎 | 2 |

### 1-2. 클래스별 이미지 수 (실제 학습에 사용된 기준)

확인 명령:

```bash
for d in dataset/*/*; do echo "$d $(ls "$d" | wc -l)"; done
```

| 클래스               |     Train | Validation |    Test |
| ----------------- | --------: | ---------: | ------: |
| angular_leaf_spot |       345 |         44 |      43 |
| bean_rust         |       348 |         45 |      43 |
| healthy           |       341 |         44 |      42 |
| **합계**            | **1,034** |    **133** | **128** |

- `training_report.json`의 `train_count: 1034`, `val_count: 133`과 일치한다.
- 클래스별 장수 차이가 작아 클래스 균형은 양호하다.

### 1-3. 데이터 점검 결과

| 점검 | 방법 | 결과 |
|---|---|---|
| 비이미지 파일 | `find dataset -type f ! -iname "*.jpg" ! -iname "*.jpeg" ! -iname "*.png" ! -iname "*.webp"` | `train/healthy/healthy_train.120tore` 1개 발견 → 조치 (아래) |
| 손상 이미지 | Pillow `Image.verify()`로 전체 파일 검사 | 손상 **0건** (조치 후) |
| 이미지 형식 | Pillow로 크기·색 형식 집계 | 전체 1,295장 모두 **500×500, RGB** |
| 중복·누출 | `find dataset -type f -exec md5sum {} + \| sort \| uniq -w32 -dD` | 중복 **0건** (train·val·test 간 누출 없음) |
| 라벨 육안 확인 | 클래스별 일부 이미지 직접 열람 | 라벨 이상 없음. 초점 흐림, 조명에 따른 색 차이, 잎 모양·구도 차이 존재 → 실제 촬영 환경의 다양성으로 판단해 데이터 유지 |

**비이미지 파일 처리 기록**

1. 폴더 파일 수(train/healthy 342개, train 합계 1,035개)와 학습 리포트 `train_count: 1034`가 1개 다른 것을 확인.
2. 확장자가 깨진 파일 `dataset/train/healthy/healthy_train.120tore`를 찾음 (학습 코드의 확장자 필터 `.jpg/.jpeg/.png/.webp`에 걸리지 않아 이미 제외되던 파일).
3. `.jpg`로 이름을 바꿔 학습을 시도 → `PIL.UnidentifiedImageError: cannot identify image file 'dataset/train/healthy/healthy_train.120.jpg'` 발생.
4. 이미지로 열리지 않는 손상 파일로 판단하고 삭제 → train 1,034장. 실험 A부터 동일한 1,034장으로 학습되었으므로 실험 간 데이터 조건은 같다.

---

## 2. 실행 설정

### 2-1. 공통 설정 (제공 코드 기준)

| 파라미터            | 값                                         | 비고                                         |
| --------------- | ----------------------------------------- | ------------------------------------------ |
| base_model      | `microsoft/resnet-18`                     | 코드 상수 `BASE_MODEL`                         |
| batch_size      | 8                                         |                                            |
| optimizer       | AdamW                                     | 코드 고정, `requires_grad=True`인 파라미터만 갱신      |
| seed            | 42                                        | `random.seed(42)`, `torch.manual_seed(42)` |
| best 모델 저장 규칙   | Validation Accuracy가 **이전 최고값보다 클 때만** 저장 | 동점이면 먼저 도달한 epoch 유지                       |
| freeze_backbone | True일 때 `classifier.*`만 학습                | Head-only                                  |

### 2-2. 최종 모델 실행 명령

```bash
python -m training.finetune --data dataset --output artifacts/beans_resnet18 --epochs 10 --batch-size 8 --lr 0.001 --freeze-backbone
```

| 파라미터 | 값 |
|---|---|
| epochs | 10 |
| learning_rate | 0.001 |
| freeze_backbone | True (Head-only) |
| output | `artifacts/beans_resnet18` |

> 제공 코드는 수정하지 않았고 `--epochs`, `--lr` 실행 옵션만 바꿨다.

### 2-3. 실행 환경

| 항목                   | 값                                                                    |
| -------------------- | -------------------------------------------------------------------- |
| OS                   | Windows + WSL2 Ubuntu 26.04 LTS                                      |
| Python               | 3.14 (`.venv`)                                                       |
| torch / transformers | Name: torch / Version: 2.15.0 / Name: transformers / Version: 4.57.6 |
| 장치                   | GPU - NVIDIA GeForce RTX 3060                                        |

---

## 3. 성능 개선 실험 (Validation 기준)

같은 데이터(train 1,034 / val 133), 같은 seed(42), batch 8에서 **한 번에 한 가지만 바꿔** 비교했다.

| 실험    | 방식            | 바꾼 것                | epochs |        lr | 출력 폴더                  | Best Val Acc |        맞힌 수 | 학습 시간    |
| ----- | ------------- | ------------------- | -----: | --------: | ---------------------- | -----------: | ----------: | -------- |
| A     | Head-only     | 기준선 (제공 명령)         |      3 |    0.0002 | `beans_head_e3`        |       0.7669 |     102/133 | 28.4s    |
| B     | Head-only     | epochs 3→10         |     10 |    0.0002 | `beans_head_e10`       |       0.8947 |     119/133 | 1m 18.8s |
| **C** | **Head-only** | **lr 0.0002→0.001** | **10** | **0.001** | `beans_head_e10_lr1e3` |   **0.9398** | **125/133** | 1m 19.8s |
| D     | Full (선택 실험)  | Backbone 고정 해제      |     10 |    0.0002 | `beans_full_e10`       |       1.0000 |     133/133 | 1m 24.5s |

- 실험 A 결과 폴더는 최종 모델 생성 전 `artifacts/beans_resnet18` → `artifacts/beans_head_e3`로 이름을 바꿔 보관했다.
- 각 실험의 원본 기록: [`data/A_head_e3_training_report.json`](data/A_head_e3_training_report.json), [`data/B_head_e10_training_report.json`](data/B_head_e10_training_report.json), [`data/C_head_e10_lr1e3_training_report.json`](data/C_head_e10_lr1e3_training_report.json), [`data/D_full_e10_training_report.json`](data/D_full_e10_training_report.json)

> A와 B는 epochs 외 설정이 같고 seed가 고정되어 epoch 1~3 결과가 동일하다.

### 3-1. 실험별 해석

![[01_learning_curves.png]]

![[02_best_val_bar.png]]
#### ▶ 실험별 해석 요약

| 비교    | 근거 (수치)                                                            | 해석                                             |
| ----- | ------------------------------------------------------------------ | ---------------------------------------------- |
| A → B | A의 loss가 0.99 → 0.71로 계속 감소하던 중 종료                                 | 3 epoch로는 학습 부족                                |
| B     | epoch 6~10 val 0.88 / 0.88 / 0.89 / 0.89 / 0.87                    | lr 0.0002 Head-only는 약 0.89에서 한계               |
| B → C | 마지막 loss 0.4942 → 0.3746                                           | 학습률을 키워 같은 epoch 안에 더 많이 학습                    |
| C 변동  | epoch2~3 0.80 → 0.78, epoch9 0.89로 하락                              | 학습률이 커서 흔들림. epoch4 이후 대부분 0.92~0.94           |
| D     | epoch1부터 0.9398, epoch2~10은 0.962~1.000                            | Backbone까지 학습하면 가장 높음                          |
| D 주의  | best 1.0은 val 133장 중 최고 epoch를 고른 값. 1장 ≈ 0.75%p, e6~10 평균 ≈ 0.973 | 낙관적으로 해석하지 않음                                  |
| 시간    | A 28.4s → B·C 약 80초, D 84.5초                                       | 시간은 epoch 수에 비례. lr은 시간에 영향 없음. Full도 5초만 더 걸림 |
| 최종    | 가이드 Head-only 기준                                                   | C로 서빙(Test 0.8984), D는 비교 자료                   |

### 3-2. 최종 모델 선택 근거

- 선택 기준은 **Validation Accuracy만** 사용했다. Test 데이터는 모델·학습률 선택에 사용하지 않았다.
- Head-only 실험(A·B·C) 중 C가 가장 높아 C 설정을 최종으로 선택하고, 같은 명령을 `--output artifacts/beans_resnet18`로 다시 실행했다.

---

## 4. 최종 모델 학습 결과

출처: `artifacts/beans_resnet18/training_report.json`

| Epoch | Train Loss | Validation Accuracy |
| ----: | ---------: | ------------------: |
|     1 |     0.8257 |              0.8271 |
|     2 |     0.5820 |              0.8045 |
|     3 |     0.4980 |              0.7820 |
|     4 |     0.4592 |              0.9248 |
|     5 |     0.4095 |              0.9323 |
| **6** | **0.3835** |     **0.9398** ← 저장 |
|     7 |     0.4081 |              0.9398 |
|     8 |     0.3737 |              0.9398 |
|     9 |     0.4054 |              0.8947 |
|    10 |     0.3746 |              0.9323 |

- **Best Validation Accuracy: 0.9398** (125 / 133)
- epoch 6~8이 동점이며, 저장 규칙(기록 경신 시에만 저장)에 따라 처음 도달한 **epoch 6의 모델**이 저장되었다.
- 실험 C와 10개 epoch 기록이 소수점까지 동일하게 재현되었다 (seed 42 고정).
- Test 결과는 **5. Test Set 평가** 참고 (별도 평가 스크립트로 측정).

---

## 5. Test Set 평가

제공 학습 코드에는 Test 평가 절차가 없어 평가 스크립트 `training/evaluate_test.py`를 **별도로 작성**했다. 학습 코드는 수정하지 않았다.

- 평가 시점: Validation으로 최종 모델(C 설정)을 **선택한 뒤** 1회 평가. Test는 모델·학습률 선택에 사용하지 않았다.
- 평가 대상: `artifacts/beans_resnet18` / `dataset/test` 128장
- 원본 출력: [`data/test_eval_beans_resnet18.txt`](data/test_eval_beans_resnet18.txt)

```bash
python -m training.evaluate_test --model artifacts/beans_resnet18 --data dataset/test | tee reports/data/test_eval_beans_resnet18.txt
```

**Test Accuracy: 0.8984 (115 / 128)**

혼동행렬 (행 = 정답, 열 = 예측)

| 정답 \ 예측           | angular_leaf_spot | bean_rust | healthy |  합계 |
| ----------------- | ----------------: | --------: | ------: | --: |
| angular_leaf_spot |            **37** |         4 |       2 |  43 |
| bean_rust         |                 5 |    **38** |       0 |  43 |
| healthy           |                 1 |         1 |  **40** |  42 |
| 예측 합계             |                43 |        43 |      42 | 128 |

| 클래스 | Precision | Recall |
|---|---:|---:|
| angular_leaf_spot | 0.860 | 0.860 |
| bean_rust | 0.884 | 0.884 |
| healthy | 0.952 | 0.952 |

**해석**
- Test Accuracy(0.8984)는 Validation Best(0.9398)보다 4.14%p 낮다. Validation은 best epoch를 고르는 데 쓰여 낙관적이고, Test는 선택에 쓰지 않은 데이터이므로 Test 값이 실제 성능에 더 가깝다.
- 틀린 13장 중 **9장이 angular_leaf_spot ↔ bean_rust** 사이 혼동이다. 두 병해의 반점 모양·색이 비슷한 것으로 보인다.
- healthy는 Precision·Recall 0.952로 가장 잘 구분되었다.
- 이번 결과는 클래스마다 예측 수와 정답 수가 같아(43/43/42) Precision과 Recall 값이 우연히 같다.

---

## 6. 생성 파일 확인

```bash
ls -lh artifacts/beans_resnet18
```

| 파일 | 역할 | 확인 |
|---|---|:---:|
| `model.safetensors` | 모델 가중치 (학습 결과) | ✓ |
| `config.json` | 모델 설정, `id2label` 클래스 이름 | ✓ |
| `preprocessor_config.json` | 전처리 규칙 (크기·정규화) | ✓ |
| `training_report.json` | 학습 기록 (클래스, 데이터 수, history, best) | ✓ |

---

## 7. 서비스 연결 확인

| 항목 | 값 |
|---|---|
| `.env` | `MODEL_PATH=artifacts/beans_resnet18` |
| 재시작 | FastAPI [ ] / Celery Worker [ ] (02 단계부터 둘 다) |

동기 API 결과 (`POST /api/sync/classify/`, test 폴더 이미지 사용):

| 입력 파일 (폴더 정답)                   | HTTP | top1.label | top1.score | inference_ms | cache_hit | 일치  |
| ------------------------------- | ---: | ---------- | ---------: | -----------: | --------- | :-: |
| `healthy_test.40.jpg` (healthy) |  미기록 | healthy    |     0.6759 |        19.74 | false     |  ✓  |

- 실행 환경: 로컬 `.venv`에서 `uvicorn main:app --reload` (01 단계, Redis 캐시 미연결)
- 실행 장치: CPU (서버 로그 `Device set to use cpu`)
- `inference_ms 19.74`는 같은 이미지의 두 번째 요청 값(모델 첫 로딩 이후)
- HTTP 상태코드와 Docker 서빙 결과(200, 21.71ms)는 산출물 3 참고

> 위 결과는 서빙 동작 확인용이며 Test 정확도를 대신하지 않는다. score는 모델 출력 확률로, 보정된 신뢰도라고 단정하지 않는다.

---

## 8. 결과 판정

| 구분    | 내용                                                                                                                   |
| ----- | -------------------------------------------------------------------------------------------------------------------- |
| 완료    | 데이터 매핑·점검, Head-only Fine-tuning(최종 C 설정), 모델 파일 4종 생성, 실행 옵션 비교 실험 A·B·C, 선택 실험 D(Full), 최종 모델 Test 평가              |
| 개선 결과 | Head-only 기준선 0.7669 → 최종 0.9398 (**+17.29%p**, Validation 기준)                                                       |
| 최종 성능 | **Test Accuracy 0.8984 (115/128)**                                                                                   |
| 미측정   | 실험 A·B·D의 Test 성능 (Test는 최종 모델에만 적용)                                                                                 |
| 한계    | val 133장·test 128장으로 작아 1장 차이가 약 0.75~0.78%p; angular_leaf_spot ↔ bean_rust 혼동이 오류의 대부분; lr 0.001에서 epoch별 val 변동 관찰 |

### 증빙

| 파일                                                                                                                                                           | 내용                               |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------ | -------------------------------- |
| `data/01_data_count.txt`                                                                                                                                     | 클래스별 이미지 수 출력                    |
| `캡처 없음. 당시 오류 원문: `PIL.UnidentifiedImageError: cannot identify image file 'dataset/train/healthy/healthy_train.120.jpg'` (finetune.py line 32 `Image.open`)` | `UnidentifiedImageError` 화면      |
| `data/01_data_check.txt`                                                                                                                                     | 손상 0 / 500×500 RGB / 중복 0 출력     |
| `data/01_train_log_final.txt`                                                                                                                                | 최종 학습 터미널 출력                     |
| `data/01_model_folder.txt`                                                                                                                                   | `artifacts/beans_resnet18` 파일 목록 |
| `data/01_test_eval_beans_resnet18.txt`                                                                                                                       | Test 평가 출력 (Accuracy, 혼동행렬)      |

---

## 9. 고도화 제안 (미구현)

아래는 **제안이며 이번 코드에 구현하지 않았고 성능 개선 여부도 검증하지 않았다.**

- 오류가 몰린 angular_leaf_spot ↔ bean_rust 구분을 위해 Full Fine-tuning 모델(실험 D)을 같은 Test 절차로 평가해 서빙 후보로 검토한다.
- 학습률 스케줄(초반 크게 → 후반 작게)을 적용해 lr 0.001에서 보인 val 변동을 줄인다.
- 촬영 환경 다양성(초점 흐림, 조명 색 차이, 구도)을 고려해 밝기·색·흐림·회전·크롭 데이터 증강을 실험한다.
- HSV 색공간 기반 병반 후보 면적비를 CNN 점수와 함께 기록해, 신뢰도가 낮거나 결과가 엇갈리는 이미지를 `확인 필요`로 분리하는 규칙 후처리를 설계한다.
