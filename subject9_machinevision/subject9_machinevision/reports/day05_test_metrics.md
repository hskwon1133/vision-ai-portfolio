# Day05 Test Metrics

## Baseline

- Rule Version: day05-logo-v1.0
- Positive Class: NG
- Test Images: 6

## Confusion Matrix

| Actual / Predicted | NG | OK |
|---|---:|---:|
| NG | TP = 3 | FN = 0 |
| OK | FP = 3 | TN = 0 |

## Metrics

- Accuracy: 0.5000
- NG Precision: 0.5000
- NG Recall: 1.0000
- FP: 3
- FN: 0

## Interpretation

- FP: 실제 OK인데 NG로 잘못 판정
- FN: 실제 NG인데 OK로 잘못 통과
