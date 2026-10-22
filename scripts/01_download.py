"""Download D1 / M1 parquet (last 3 years) + meta from HF dataset raywu918daytrade/daytrade."""
import sys, os
from pathlib import Path
from huggingface_hub import hf_hub_download

REPO = "raywu918daytrade/daytrade"
ROOT = Path(__file__).resolve().parent.parent / "data" / "raw"

def months(start, end):
    y, m = start
    out = []
    while (y, m) <= end:
        out.append(f"{y}_{m:02d}")
        m += 1
        if m > 12:
            y, m = y + 1, 1
    return out

START, END = (2023, 10), (2026, 10)
jobs = []
for tf in ["d1", "m1"]:
    for ym in months(START, END):
        jobs.append((f"db/{tf}/{ym}.parquet", ROOT / tf / f"{ym}.parquet"))
for f in ["info/info.parquet", "tickers/tickers.parquet",
          "tickers/stock_universe_2000.parquet", "tickers/tick_universe.parquet"]:
    jobs.append((f"db/{f}", ROOT / "meta" / Path(f).name))

for i, (remote, local) in enumerate(jobs, 1):
    if local.exists() and local.stat().st_size > 0:
        print(f"[{i}/{len(jobs)}] skip {remote}", flush=True); continue
    p = hf_hub_download(REPO, remote, repo_type="dataset")
    local.parent.mkdir(parents=True, exist_ok=True)
    os.link(p, local) if False else Path(local).write_bytes(Path(p).read_bytes())
    print(f"[{i}/{len(jobs)}] {remote} -> {local.stat().st_size/1e6:.1f} MB", flush=True)
print("DONE")
