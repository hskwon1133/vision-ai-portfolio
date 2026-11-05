"""동기/비동기 x 캐시 MISS/HIT 응답시간 측정 → reports/data/timings.csv
예: python scripts/measure_timings.py dataset/test/healthy/healthy_test.40.jpg
"""
import csv, statistics, subprocess, sys, time
from pathlib import Path
import httpx

BASE = "http://127.0.0.1:8000"
IMG = Path(sys.argv[1] if len(sys.argv) > 1 else "dataset/test/healthy/healthy_test.40.jpg")
N = 5
OUT = Path("reports/data/timings.csv")
data = IMG.read_bytes()
client = httpx.Client(timeout=60)

def flush_cache():   # Redis DB 1(캐시)만 비움. 브로커(DB 2)·DB 기록은 건드리지 않음
    subprocess.run(["docker", "compose", "exec", "-T", "redis", "redis-cli", "-n", "1", "FLUSHDB"],
                   check=True, capture_output=True)

def sync_once():
    t0 = time.perf_counter()
    r = client.post(f"{BASE}/api/sync/classify/", files={"image": (IMG.name, data, "image/jpeg")})
    ms = (time.perf_counter() - t0) * 1000
    d = r.json()
    return dict(http_status=r.status_code, http_ms=round(ms, 2), final_ms=round(ms, 2),
                inference_ms=d.get("inference_ms"), cache_hit=d.get("cache_hit"),
                label=d["top1"]["label"], job_id="")

def async_once():
    t0 = time.perf_counter()
    r = client.post(f"{BASE}/api/async/classify/", files={"image": (IMG.name, data, "image/jpeg")})
    http_ms = (time.perf_counter() - t0) * 1000
    job = r.json()
    while True:
        s = client.get(f"{BASE}{job['check_url']}").json()
        if s["status"] in ("done", "failed"):
            break
        time.sleep(0.05)
    final_ms = (time.perf_counter() - t0) * 1000
    return dict(http_status=r.status_code, http_ms=round(http_ms, 2), final_ms=round(final_ms, 2),
                inference_ms=s.get("inference_ms"), cache_hit=s.get("cache_hit"),
                label=(s.get("top1") or {}).get("label"), job_id=job["job_id"])

rows = []
def run(path, group, fn, flush, n):
    for i in range(1, n + 1):
        if flush:
            flush_cache()
        res = fn()
        rows.append(dict(run_id=i, path=path, cache_status=group, **res))
        print(path, group, i, res)

# 워밍업(비교 제외): 각 프로세스의 모델이 메모리에 올라가도록 1회씩
run("sync", "WARMUP", sync_once, True, 1)
run("async", "WARMUP", async_once, True, 1)
# 본 측정
run("sync", "MISS", sync_once, True, N)
run("sync", "HIT", sync_once, False, N)
run("async", "MISS", async_once, True, N)
run("async", "HIT", async_once, False, N)

OUT.parent.mkdir(parents=True, exist_ok=True)
with OUT.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["run_id", "path", "cache_status", "http_status", "http_ms",
                                      "final_ms", "inference_ms", "cache_hit", "label", "job_id"])
    w.writeheader(); w.writerows(rows)

print(f"\n이미지: {IMG}  /  반복 {N}회  /  저장: {OUT}")
print(f"{'경로':6} {'캐시':6} {'HTTP 중앙값':>12} {'최종 중앙값':>12} {'추론 중앙값':>12}")
for path in ("sync", "async"):
    for g in ("MISS", "HIT"):
        sel = [r for r in rows if r["path"] == path and r["cache_status"] == g]
        inf = [r["inference_ms"] for r in sel if r["inference_ms"] is not None]
        med_inf = f"{statistics.median(inf):.2f}" if inf else "null"
        print(f"{path:6} {g:6} {statistics.median(r['http_ms'] for r in sel):12.2f} "
              f"{statistics.median(r['final_ms'] for r in sel):12.2f} {med_inf:>12}")
