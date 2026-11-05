"""예: python -m training.evaluate_test --model artifacts/beans_resnet18 --data dataset/test"""
import argparse
from pathlib import Path
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

EXTS = {".jpg", ".jpeg", ".png", ".webp"}

def main(args):
    processor = AutoImageProcessor.from_pretrained(args.model)
    model = AutoModelForImageClassification.from_pretrained(args.model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    labels = [model.config.id2label[i] for i in range(len(model.config.id2label))]

    # 혼동행렬: confusion[정답][예측] = 장수
    confusion = {t: {p: 0 for p in labels} for t in labels}
    for true_label in labels:
        for path in sorted((Path(args.data) / true_label).iterdir()):
            if path.suffix.lower() not in EXTS:
                continue
            with Image.open(path) as raw:
                pixels = processor(images=raw.convert("RGB"), return_tensors="pt")["pixel_values"].to(device)
            with torch.no_grad():
                pred = model(pixel_values=pixels).logits.argmax(-1).item()
            confusion[true_label][labels[pred]] += 1

    total = sum(sum(row.values()) for row in confusion.values())
    correct = sum(confusion[l][l] for l in labels)
    print(f"모델: {args.model}")
    print(f"Test Accuracy: {correct}/{total} = {correct/total:.4f}\n")
    print("정답 \\ 예측".ljust(20) + "".join(l[:12].rjust(14) for l in labels))
    for t in labels:
        print(t.ljust(20) + "".join(str(confusion[t][p]).rjust(14) for p in labels))
    print()
    for l in labels:
        tp = confusion[l][l]
        pred_n = sum(confusion[t][l] for t in labels)
        true_n = sum(confusion[l].values())
        precision = tp / pred_n if pred_n else 0
        recall = tp / true_n if true_n else 0
        print(f"{l:20s} precision={precision:.3f} recall={recall:.3f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", default="dataset/test")
    main(parser.parse_args())
