"""為已發布的名單產生當日 5 分 K (正式+影子的股票聯集) → HF db/signals/bars/<date>.json, 手機頁卡片「分K」用.
來源: HF db/m1/<YYYY_MM>.parquet (下載到 data/raw/m1/, 月檔會更新所以每次 force_download 當月檔).
JSON: {"date":..., "bars": {stock_id: [[t_min, o, h, l, c, v], ...]}} (t_min = 09:00 起的分鐘數, 5 分一根, 含 13:30 收盤競價那根)
只補「已發布但 bars 還沒有」的日子. 用法: HF_TOKEN=... python3 scripts/build_bars.py [--dates 2026-10-22,...]
"""
import os, sys, io, json, argparse
from pathlib import Path
import pandas as pd, requests
ROOT = Path(__file__).resolve().parent.parent; REPO = "raywu918daytrade/daytrade"
BASE = f"https://huggingface.co/datasets/{REPO}/resolve/main/db/signals/"

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dates", default=None); ap.add_argument("--dry", action="store_true"); a = ap.parse_args()
    token = os.environ.get("HF_TOKEN")
    if not token and not a.dry: sys.exit("缺 HF_TOKEN")
    from huggingface_hub import HfApi, CommitOperationAdd, hf_hub_download
    api = HfApi(token=token)
    files = set(api.list_repo_files(REPO, repo_type="dataset"))
    pub = sorted({f.split("/")[-1][:10] for f in files if f.startswith("db/signals/20") and f.endswith(".json") and "_" not in f.split("/")[-1][10:]})
    have = {f.split("/")[-1][:10] for f in files if f.startswith("db/signals/bars/")}
    todo = [d for d in (a.dates.split(",") if a.dates else pub) if d not in have]
    print(f"已發布 {len(pub)} 天, 已有分K {len(have)} 天, 待做 {len(todo)}: {todo}")
    if not todo: return
    ops = []; cache = {}
    for d in todo:
        ym = d[:7].replace("-", "_")
        if ym not in cache:
            try:
                p = hf_hub_download(REPO, f"db/m1/{ym}.parquet", repo_type="dataset", force_download=True)
            except Exception as e: print(f"  {d}: 沒有 m1 月檔 {ym} ({e})"); cache[ym] = None; continue
            m1 = pd.read_parquet(p); ts = pd.to_datetime(m1["date"]); m1["day"] = ts.dt.strftime("%Y-%m-%d"); m1["t"] = ts.dt.hour*60 + ts.dt.minute - 540
            cache[ym] = m1[(m1.t >= 0) & (m1.t <= 270)]
        m1 = cache[ym]
        if m1 is None: continue
        ids = set()
        for sfx in ("", "_base", "_sr", "_seq"):
            if f"db/signals/{d}{sfx}.json" in files:
                try: ids |= {p["stock_id"] for p in requests.get(BASE + f"{d}{sfx}.json", params={"t": 1}, timeout=30).json().get("picks", [])}
                except Exception: pass
        g = m1[(m1.day == d) & m1.stock_id.isin(ids)]
        if g.empty: print(f"  {d}: m1 沒有這天的資料 (尚未更新?)"); continue
        bars = {}
        for sid, s in g.groupby("stock_id"):
            s = s.sort_values("t"); s = s.assign(b=(s.t // 5).clip(upper=53))   # 13:30 (t=270) 併入最後一根
            k = s.groupby("b").agg(t=("t", "first"), o=("open", "first"), h=("high", "max"), l=("low", "min"), c=("close", "last"), v=("volume", "sum")).reset_index(drop=True)
            bars[sid] = [[int(r.t), round(float(r.o), 2), round(float(r.h), 2), round(float(r.l), 2), round(float(r.c), 2), int(r.v)] for r in k.itertuples()]
        js = json.dumps({"date": d, "bars": bars}, separators=(",", ":")).encode("utf-8")
        ops.append(CommitOperationAdd(f"db/signals/bars/{d}.json", js)); print(f"  {d}: {len(bars)} 檔, {len(js)/1024:.0f} KB")
    if a.dry or not ops: print("dry run 或沒有可上傳"); return
    api.create_commit(repo_id=REPO, repo_type="dataset", operations=ops, commit_message=f"bars: +{len(ops)} days")
    print(f"已上傳 {len(ops)} 天的分K")

if __name__ == "__main__":
    main()
