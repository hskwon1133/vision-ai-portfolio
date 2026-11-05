"""학습 결과 그래프: python -m training.plot_results"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")                      # 화면 없이 파일로만 저장
import matplotlib.pyplot as plt

OUT = Path("reports/figures")
OUT.mkdir(parents=True, exist_ok=True)

RUNS = {   # 이름: (폴더, 색)
    "A Head-only e3 lr0.0002": ("artifacts/beans_head_e3", "#2a78d6"),
    "B Head-only e10 lr0.0002": ("artifacts/beans_head_e10", "#eb6834"),
    "C Head-only e10 lr0.001 (final)": ("artifacts/beans_head_e10_lr1e3", "#1baf7a"),
    "D Full e10 lr0.0002 (compare)": ("artifacts/beans_full_e10", "#eda100"),
}

def load(folder):
    r = json.loads((Path(folder) / "training_report.json").read_text(encoding="utf-8"))
    h = r["history"]
    return [x["epoch"] for x in h], [x["loss"] for x in h], [x["val_accuracy"] for x in h]

# ① 실험 C: loss / val accuracy 두 칸
ep, loss, acc = load("artifacts/beans_head_e10_lr1e3")
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
a1.plot(ep, loss, marker="o", color="#1baf7a")
a1.set_title("C: Train Loss (lower is better)")
a1.set_xlabel("Epoch"); a1.set_xticks(ep); a1.grid(alpha=0.3)
a2.plot(ep, acc, marker="o", color="#1baf7a")
b = acc.index(max(acc))
a2.scatter(ep[b], acc[b], s=120, facecolors="none", edgecolors="black")
a2.annotate(f"best {acc[b]:.4f} (epoch {ep[b]})", (ep[b], acc[b]),
            xytext=(-30, -30), textcoords="offset points")
a2.set_title("C: Validation Accuracy (higher is better)")
a2.set_xlabel("Epoch"); a2.set_xticks(ep); a2.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "C_learning_curve.png", dpi=150)

# ② A·B·C·D val accuracy 비교 한 칸
fig, ax = plt.subplots(figsize=(10, 5))
for name, (folder, color) in RUNS.items():
    if not Path(folder, "training_report.json").exists():
        print("건너뜀(없음):", folder)
        continue
    e, _, a = load(folder)
    style = dict(linewidth=6, linestyle="--", alpha=0.6) if name.startswith("A") else {}
    ax.plot(e, a, marker="o", color=color, label=f"{name}  best {max(a):.4f}", **style)
ax.set_title("Validation Accuracy by experiment (train 1034 / val 133 / seed 42)")
ax.set_xlabel("Epoch"); ax.set_ylabel("Validation Accuracy")
ax.set_xticks(range(1, 11)); ax.grid(alpha=0.3); ax.legend(loc="lower right")
fig.tight_layout()
fig.savefig(OUT / "experiments_val_compare.png", dpi=150)
print("저장:", OUT / "C_learning_curve.png", ",", OUT / "experiments_val_compare.png")
